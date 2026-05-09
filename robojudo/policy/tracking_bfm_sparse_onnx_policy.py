from __future__ import annotations

import logging
import os
from collections import deque
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
        if len(inputs) != 1:
            raise ValueError(f"Expected exactly one ONNX input, got {len(inputs)}.")
        if len(outputs) != 1:
            raise ValueError(f"Expected exactly one ONNX output, got {len(outputs)}.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.obs_dim = _infer_static_feature_dim(inputs[0].shape)
        self.action_dim = _infer_static_feature_dim(outputs[0].shape)

        expected_obs_dim = getattr(cfg_policy, "expected_obs_dim", None)
        if expected_obs_dim is not None and expected_obs_dim != self.obs_dim:
            raise ValueError(
                f"observation dimension mismatch: config expects {expected_obs_dim}, "
                f"ONNX provides {self.obs_dim}"
            )
        if self.cfg_action_dof.num_dofs != self.action_dim:
            raise ValueError(
                f"action dimension mismatch: cfg_action_dof has {self.cfg_action_dof.num_dofs}, "
                f"ONNX provides {self.action_dim}"
            )

        model_meta = self.session.get_modelmeta().custom_metadata_map
        self.obs_group = cfg_policy.obs_group or model_meta.get("obs_group", "actor")
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

    def _load_obs_contract(self, onnx_path: Path, env_yaml_path: str | None) -> None:
        yaml_path = Path(env_yaml_path) if env_yaml_path is not None else onnx_path.parent / "params" / "env.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"tracking_bfm env yaml not found at {yaml_path}. "
                "Expected params/env.yaml next to the exported ONNX, or set env_yaml_path explicitly."
            )

        env_cfg = _load_yaml_relaxed(yaml_path)
        try:
            group_cfg = env_cfg["observations"][self.obs_group]
            terms_cfg = group_cfg["terms"]
        except KeyError as exc:
            raise KeyError(f"Observation group '{self.obs_group}' not found in {yaml_path}") from exc

        self.term_order = list(terms_cfg.keys())
        self.command_terms = [name for name in self.term_order if name in _COMMAND_TERM_NAMES]
        self.robot_terms = [name for name in self.term_order if name in _ROBOT_TERM_NAMES]
        if not self.command_terms:
            raise ValueError(f"No sparse command terms found in observation group '{self.obs_group}'.")
        if not self.robot_terms:
            raise ValueError(f"No robot state terms found in observation group '{self.obs_group}'.")

        command_ref = terms_cfg[self.command_terms[0]].get("params", {})
        self.command_history_steps = int(command_ref.get("history_steps", 0))
        self.command_future_steps = int(command_ref.get("future_steps", 1))
        self.command_history_length = max(1, self.command_history_steps + 1)
        self.command_total_length = self.command_history_steps + self.command_future_steps

        robot_ref = terms_cfg[self.robot_terms[0]]
        raw_robot_history_length = int(robot_ref.get("history_length", 0) or 0)
        self.robot_history_length = max(1, raw_robot_history_length)

        self._command_buffer = deque(maxlen=self.command_history_length)
        self._robot_buffers = {
            term_name: deque(maxlen=self.robot_history_length) for term_name in self.robot_terms
        }

    def reset(self):
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._command_buffer.clear()
        for history in self._robot_buffers.values():
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
            "joint_pos": (np.asarray(env_data.dof_pos, dtype=np.float32) - self.default_dof_pos.astype(np.float32)).reshape(-1),
            "joint_vel": np.asarray(env_data.dof_vel, dtype=np.float32).reshape(-1),
            "actions": self.last_action.astype(np.float32).reshape(-1),
        }

    def _command_sequence(self, ctrl: dict[str, Any]) -> dict[str, np.ndarray]:
        current = {
            term_name: np.asarray(ctrl[term_name], dtype=np.float32).reshape(-1) for term_name in self.command_terms
        }
        self._command_buffer.append(current)
        history = list(self._command_buffer)
        while len(history) < self.command_history_length:
            history.insert(0, history[0].copy())

        sequence = {}
        for term_name in self.command_terms:
            pieces = [step[term_name] for step in history[-self.command_history_length :]]
            if self.command_future_steps > 1:
                pieces.extend([current[term_name]] * (self.command_future_steps - 1))
            sequence[term_name] = np.concatenate(pieces, dtype=np.float32)
        return sequence

    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        if self.ctrl_type not in ctrl_data:
            raise KeyError(f"Controller data '{self.ctrl_type}' not found in ctrl_data.")
        ctrl = ctrl_data[self.ctrl_type]
        command_terms = self._command_sequence(ctrl)
        robot_current = self._current_robot_terms(env_data)

        robot_terms = {
            term_name: np.concatenate(
                self._push_history(self._robot_buffers[term_name], robot_current[term_name], self.robot_history_length),
                dtype=np.float32,
            )
            for term_name in self.robot_terms
        }

        obs_parts = []
        for term_name in self.term_order:
            if term_name in command_terms:
                obs_parts.append(command_terms[term_name])
            elif term_name in robot_terms:
                obs_parts.append(robot_terms[term_name])
            else:
                raise KeyError(f"Unsupported observation term '{term_name}' in sparse actor contract.")

        obs = np.concatenate(obs_parts, dtype=np.float32)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"assembled observation dimension mismatch: expected {self.obs_dim}, got {obs.shape[0]}"
            )
        return obs, {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"observation dimension mismatch: expected {self.obs_dim}, got {obs.shape[1]}"
            )
        [action] = self.session.run([self.output_name], {self.input_name: obs})
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
