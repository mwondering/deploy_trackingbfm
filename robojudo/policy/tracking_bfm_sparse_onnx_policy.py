from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as sRot
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from robojudo.policy import Policy, policy_registry
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)

_COMMAND_TERM_NAMES = {"ee_pose", "base_lin_vel_b", "base_ang_vel_b", "anchor_height_w"}
_ROBOT_TERM_NAMES = {"projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"}


def _infer_static_feature_dim(shape: list[Any]) -> int:
    if not shape:
        raise ValueError("ONNX tensor shape is empty.")
    feature_dim = shape[-1]
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(f"Expected a static positive feature dimension, got {shape}.")
    return feature_dim


def _yaml_scalar(node: ScalarNode):
    if node.tag.endswith(":int"):
        return int(node.value)
    if node.tag.endswith(":float"):
        return float(node.value)
    if node.tag.endswith(":bool"):
        return node.value.lower() == "true"
    if node.tag.endswith(":null"):
        return None
    return node.value


def _yaml_node_to_python(node):
    if isinstance(node, ScalarNode):
        return _yaml_scalar(node)
    if isinstance(node, SequenceNode):
        return [_yaml_node_to_python(item) for item in node.value]
    if isinstance(node, MappingNode):
        return {_yaml_node_to_python(key): _yaml_node_to_python(value) for key, value in node.value}
    raise TypeError(f"Unsupported YAML node type: {type(node)!r}")


def _load_yaml_relaxed(path: Path) -> dict[str, Any]:
    root = yaml.compose(path.read_text())
    if root is None:
        raise ValueError(f"YAML file at {path} is empty.")
    data = _yaml_node_to_python(root)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(data)!r}")
    return data


@policy_registry.register
class TrackingBfmSparseOnnxPolicy(Policy):
    """ONNX-backed sparse actor for tracking_bfm deployment."""

    def __init__(self, cfg_policy, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)

        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading sparse ONNX policy from {cfg_policy.policy_file}")
        self.session = ort.InferenceSession(
            cfg_policy.policy_file,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) not in (1, 2):
            raise ValueError(f"Expected one or two ONNX inputs, got {len(inputs)}.")
        if len(outputs) != 1:
            raise ValueError(f"Expected exactly one ONNX output, got {len(outputs)}.")

        input_by_name = {input_info.name: input_info for input_info in inputs}
        obs_input = input_by_name.get("obs", inputs[0])
        self.input_name = obs_input.name
        self.proprio_input_name = None
        proprio_input = None
        if len(inputs) == 2:
            proprio_input = input_by_name.get("proprio")
            if proprio_input is None:
                proprio_input = next(input_info for input_info in inputs if input_info.name != self.input_name)
            self.proprio_input_name = proprio_input.name

        self.output_name = outputs[0].name
        self.obs_dim = _infer_static_feature_dim(obs_input.shape)
        self.proprio_dim = _infer_static_feature_dim(proprio_input.shape) if proprio_input is not None else None
        self.action_dim = _infer_static_feature_dim(outputs[0].shape)

        expected_obs_dim = getattr(cfg_policy, "expected_obs_dim", None)
        if expected_obs_dim is not None and expected_obs_dim != self.obs_dim:
            raise ValueError(
                f"observation dimension mismatch: config expects {expected_obs_dim}, "
                f"ONNX provides {self.obs_dim}"
            )
        expected_proprio_dim = getattr(cfg_policy, "expected_proprio_dim", None)
        if expected_proprio_dim is not None and expected_proprio_dim != self.proprio_dim:
            raise ValueError(
                f"proprio observation dimension mismatch: config expects {expected_proprio_dim}, "
                f"ONNX provides {self.proprio_dim}"
            )
        if self.cfg_action_dof.num_dofs != self.action_dim:
            raise ValueError(
                f"action dimension mismatch: cfg_action_dof has {self.cfg_action_dof.num_dofs}, "
                f"ONNX provides {self.action_dim}"
            )

        model_meta = self.session.get_modelmeta().custom_metadata_map
        self.obs_group = cfg_policy.obs_group or model_meta.get("obs_group", "actor")
        self.proprio_obs_group = cfg_policy.proprio_obs_group or model_meta.get("proprio_obs_group")
        self.ctrl_type = cfg_policy.ctrl_type
        self._action_scales = None
        if cfg_policy.action_scales is not None:
            action_scales = np.asarray(cfg_policy.action_scales, dtype=np.float32).reshape(-1)
            if action_scales.shape[0] != self.action_dim:
                raise ValueError(
                    f"action scale dimension mismatch: expected {self.action_dim}, got {action_scales.shape[0]}"
                )
            self._action_scales = action_scales
        self._load_obs_contract(Path(cfg_policy.policy_file), cfg_policy.env_yaml_path)
        self.reset()

    def _build_obs_contract(
        self,
        env_cfg: dict[str, Any],
        yaml_path: Path,
        group_name: str,
        *,
        require_command_terms: bool,
    ) -> dict[str, Any]:
        try:
            group_cfg = env_cfg["observations"][group_name]
            terms_cfg = group_cfg["terms"]
        except KeyError as exc:
            raise KeyError(f"Observation group '{group_name}' not found in {yaml_path}") from exc

        term_order = list(terms_cfg.keys())
        command_terms = [name for name in term_order if name in _COMMAND_TERM_NAMES]
        robot_terms = [name for name in term_order if name in _ROBOT_TERM_NAMES]
        if require_command_terms and not command_terms:
            raise ValueError(f"No sparse command terms found in observation group '{group_name}'.")
        if not robot_terms:
            raise ValueError(f"No robot state terms found in observation group '{group_name}'.")

        if command_terms:
            command_ref = terms_cfg[command_terms[0]].get("params", {})
            command_history_steps = int(command_ref.get("history_steps", 0))
            command_future_steps = int(command_ref.get("future_steps", 1))
        else:
            command_history_steps = 0
            command_future_steps = 0
        command_history_length = max(1, command_history_steps + 1)

        robot_ref = terms_cfg[robot_terms[0]]
        raw_robot_history_length = int(robot_ref.get("history_length", 0) or 0)
        robot_history_length = max(1, raw_robot_history_length)

        return {
            "group_name": group_name,
            "term_order": term_order,
            "command_terms": command_terms,
            "robot_terms": robot_terms,
            "command_history_steps": command_history_steps,
            "command_future_steps": command_future_steps,
            "command_history_length": command_history_length,
            "command_total_length": command_history_steps + command_future_steps,
            "robot_history_length": robot_history_length,
            "command_buffer": deque(maxlen=command_history_length),
            "robot_buffers": {term_name: deque(maxlen=robot_history_length) for term_name in robot_terms},
        }

    def _load_obs_contract(self, onnx_path: Path, env_yaml_path: str | None) -> None:
        yaml_path = Path(env_yaml_path) if env_yaml_path is not None else onnx_path.parent / "params" / "env.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"tracking_bfm env yaml not found at {yaml_path}. "
                "Expected params/env.yaml next to the exported ONNX, or set env_yaml_path explicitly."
            )

        env_cfg = _load_yaml_relaxed(yaml_path)
        self._obs_contract = self._build_obs_contract(
            env_cfg,
            yaml_path,
            self.obs_group,
            require_command_terms=True,
        )
        self._proprio_contract = None
        if self.proprio_input_name is not None:
            if self.proprio_obs_group is None:
                raise ValueError(
                    "ONNX model has a proprio input but no proprio observation group. "
                    "Set proprio_obs_group or export metadata 'proprio_obs_group'."
                )
            self._proprio_contract = self._build_obs_contract(
                env_cfg,
                yaml_path,
                self.proprio_obs_group,
                require_command_terms=False,
            )

        self.term_order = self._obs_contract["term_order"]
        self.command_terms = self._obs_contract["command_terms"]
        self.robot_terms = self._obs_contract["robot_terms"]
        self.command_history_steps = self._obs_contract["command_history_steps"]
        self.command_future_steps = self._obs_contract["command_future_steps"]
        self.command_history_length = self._obs_contract["command_history_length"]
        self.command_total_length = self._obs_contract["command_total_length"]
        self.robot_history_length = self._obs_contract["robot_history_length"]
        self._command_buffer = self._obs_contract["command_buffer"]
        self._robot_buffers = self._obs_contract["robot_buffers"]

    def reset(self):
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        for contract in (self._obs_contract, self._proprio_contract):
            if contract is None:
                continue
            contract["command_buffer"].clear()
            for history in contract["robot_buffers"].values():
                history.clear()

    def post_step_callback(self, commands: list[str] | None = None):
        del commands
        return

    def _push_history(self, history: deque[np.ndarray], value: np.ndarray, target_length: int) -> list[np.ndarray]:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        history.append(value)
        values = list(history)
        if not values:
            values = [value]
        while len(values) < target_length:
            values.insert(0, values[0].copy())
        return values[-target_length:]

    def _current_robot_terms(self, env_data) -> dict[str, np.ndarray]:
        return {
            "projected_gravity": np.asarray(get_gravity_orientation(env_data.base_quat), dtype=np.float32).reshape(-1),
            "base_ang_vel": np.asarray(env_data.base_ang_vel, dtype=np.float32).reshape(-1),
            "joint_pos": (
                np.asarray(env_data.dof_pos, dtype=np.float32) - self.default_dof_pos.astype(np.float32)
            ).reshape(-1),
            "joint_vel": np.asarray(env_data.dof_vel, dtype=np.float32).reshape(-1),
            "actions": self.last_action.astype(np.float32).reshape(-1),
        }

    def _command_sequence(self, ctrl: dict[str, Any], contract: dict[str, Any]) -> dict[str, np.ndarray]:
        current = {
            term_name: np.asarray(ctrl[term_name], dtype=np.float32).reshape(-1)
            for term_name in contract["command_terms"]
        }
        if not current:
            return {}
        contract["command_buffer"].append(current)
        history = list(contract["command_buffer"])
        while len(history) < contract["command_history_length"]:
            history.insert(0, history[0].copy())

        sequence = {}
        for term_name in contract["command_terms"]:
            pieces = [step[term_name] for step in history[-contract["command_history_length"] :]]
            if contract["command_future_steps"] > 1:
                pieces.extend([current[term_name]] * (contract["command_future_steps"] - 1))
            sequence[term_name] = np.concatenate(pieces, dtype=np.float32)
        return sequence

    def _assemble_observation(
        self,
        contract: dict[str, Any],
        robot_current: dict[str, np.ndarray],
        ctrl: dict[str, Any],
        expected_dim: int,
    ) -> np.ndarray:
        command_terms = self._command_sequence(ctrl, contract)
        robot_terms = {
            term_name: np.concatenate(
                self._push_history(
                    contract["robot_buffers"][term_name],
                    robot_current[term_name],
                    contract["robot_history_length"],
                ),
                dtype=np.float32,
            )
            for term_name in contract["robot_terms"]
        }

        obs_parts = []
        for term_name in contract["term_order"]:
            if term_name in command_terms:
                obs_parts.append(command_terms[term_name])
            elif term_name in robot_terms:
                obs_parts.append(robot_terms[term_name])
            else:
                raise KeyError(f"Unsupported observation term '{term_name}' in sparse actor contract.")

        obs = np.concatenate(obs_parts, dtype=np.float32)
        if obs.shape[0] != expected_dim:
            raise ValueError(
                f"assembled observation dimension mismatch for group '{contract['group_name']}': "
                f"expected {expected_dim}, got {obs.shape[0]}"
            )
        return obs

    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        if self.ctrl_type not in ctrl_data:
            raise KeyError(f"Controller data '{self.ctrl_type}' not found in ctrl_data.")
        ctrl = ctrl_data[self.ctrl_type]
        robot_current = self._current_robot_terms(env_data)
        obs = self._assemble_observation(self._obs_contract, robot_current, ctrl, self.obs_dim)
        if self._proprio_contract is None:
            return obs, {}

        proprio = self._assemble_observation(
            self._proprio_contract,
            robot_current,
            ctrl,
            self.proprio_dim,
        )
        return {"obs": obs, "proprio": proprio}, {}

    def _reshape_obs(self, obs: np.ndarray, expected_dim: int, label: str) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if obs.shape[1] != expected_dim:
            raise ValueError(f"{label} dimension mismatch: expected {expected_dim}, got {obs.shape[1]}")
        return obs

    def get_action(self, obs: np.ndarray | Mapping[str, np.ndarray]) -> np.ndarray:
        if self.proprio_input_name is None:
            if isinstance(obs, Mapping):
                obs = obs["obs"]
            obs = self._reshape_obs(obs, self.obs_dim, "observation")
            feeds = {self.input_name: obs}
        else:
            if not isinstance(obs, Mapping):
                raise TypeError("Latent tracking ONNX expects observation mapping with 'obs' and 'proprio'.")
            obs_array = self._reshape_obs(obs["obs"], self.obs_dim, "observation")
            proprio_array = self._reshape_obs(obs["proprio"], self.proprio_dim, "proprio observation")
            feeds = {
                self.input_name: obs_array,
                self.proprio_input_name: proprio_array,
            }
        [action] = self.session.run([self.output_name], feeds)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = (1 - self.action_beta) * self.last_action + self.action_beta * action
        self.last_action = action.copy()
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        if self._action_scales is not None:
            return action * self._action_scales
        return action * self.action_scale

    def debug_viz(self, visualizer, env_data, ctrl_data, extras):
        del extras
        if self.ctrl_type not in ctrl_data:
            return
        base_pos = getattr(env_data, "base_pos", None)
        base_quat = getattr(env_data, "base_quat", None)
        if base_pos is None or base_quat is None:
            return

        ctrl = ctrl_data[self.ctrl_type]
        base_lin_vel_b = np.asarray(ctrl["base_lin_vel_b"], dtype=np.float32).reshape(-1)
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            base_lin_vel_b,
            color=[0.2, 0.9, 0.6, 1.0],
            scale=1.5,
            horizontal_only=True,
            id=20,
        )

        ee_pose = np.asarray(ctrl["ee_pose"], dtype=np.float32).reshape(-1)
        if ee_pose.shape[0] < 18:
            return
        rot = sRot.from_quat(base_quat)
        left_pos_b = ee_pose[:3]
        right_pos_b = ee_pose[9:12]
        left_pos_w = base_pos + rot.apply(left_pos_b)
        right_pos_w = base_pos + rot.apply(right_pos_b)
        visualizer.draw_arrow(
            left_pos_w,
            base_quat,
            [0.05, 0.0, 0.0],
            color=[0.18, 0.71, 0.98, 1.0],
            scale=1.0,
            id=21,
        )
        visualizer.draw_arrow(
            right_pos_w,
            base_quat,
            [0.05, 0.0, 0.0],
            color=[1.0, 0.45, 0.24, 1.0],
            scale=1.0,
            id=22,
        )
