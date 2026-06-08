from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.tracking_bfm_sparse_onnx_policy import (
    _infer_static_feature_dim,
    _load_yaml_relaxed,
)
from robojudo.tools.tracking_bfm_sparse_command import RetargetMotionSnapshot, quat_xyzw_to_wxyz
from robojudo.tools.tracking_bfm_wbteleop_command import (
    DEFAULT_WBTELEOP_LIMB_ANCHOR_BODY_NAME,
    DEFAULT_WBTELEOP_LIMB_BODY_NAMES,
    extract_limb_pose_b_from_snapshot,
)
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)

_WBTELEOP_TERM_DIMS = {
    "command": 58,
    "ref_limb_ee_pose_b": 36,
    "motion_ref_ang_vel": 3,
    "robot_limb_ee_pose_b": 36,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "joint_pos": 29,
    "joint_vel": 29,
    "actions": 29,
}


@policy_registry.register
class WbTeleopOnnxPolicy(Policy):
    """ONNX-backed actor for the tracking_bfm wbteleop observation contract."""

    def __init__(self, cfg_policy, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug("Loading wbteleop ONNX policy from %s", cfg_policy.policy_file)
        self.session = ort.InferenceSession(cfg_policy.policy_file, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input for wbteleop actor, got {len(inputs)}.")
        if len(outputs) != 1:
            raise ValueError(f"Expected exactly one ONNX output, got {len(outputs)}.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.obs_dim = _infer_static_feature_dim(inputs[0].shape)
        self.action_dim = _infer_static_feature_dim(outputs[0].shape)

        expected_obs_dim = getattr(cfg_policy, "expected_obs_dim", None)
        if expected_obs_dim is not None and expected_obs_dim != self.obs_dim:
            raise ValueError(
                f"observation dimension mismatch: config expects {expected_obs_dim}, ONNX provides {self.obs_dim}"
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
            self._action_scales = np.asarray(cfg_policy.action_scales, dtype=np.float32).reshape(-1)
            if self._action_scales.shape[0] != self.action_dim:
                raise ValueError(
                    f"action scale dimension mismatch: expected {self.action_dim}, got {self._action_scales.shape[0]}"
                )

        self._load_obs_contract(Path(cfg_policy.policy_file), cfg_policy.env_yaml_path)
        self._default_ref_limb_ee_pose_b = self._build_default_ref_limb_pose_b()
        self.reset()

    def _load_obs_contract(self, onnx_path: Path, env_yaml_path: str | None) -> None:
        yaml_path = Path(env_yaml_path) if env_yaml_path is not None else onnx_path.parent / "params" / "env.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"tracking_bfm wbteleop env yaml not found at {yaml_path}. "
                "Expected params/env.yaml next to the exported ONNX, or set env_yaml_path explicitly."
            )

        env_cfg = _load_yaml_relaxed(yaml_path)
        try:
            terms_cfg = env_cfg["observations"][self.obs_group]["terms"]
        except KeyError as exc:
            raise KeyError(f"Observation group '{self.obs_group}' not found in {yaml_path}") from exc

        self.term_order = list(terms_cfg.keys())
        unsupported = [term_name for term_name in self.term_order if term_name not in _WBTELEOP_TERM_DIMS]
        if unsupported:
            raise KeyError(f"Unsupported wbteleop observation terms: {unsupported}")
        self.term_cfgs = terms_cfg
        self._term_buffers = {
            term_name: deque(maxlen=max(1, int((term_cfg or {}).get("history_length", 0) or 0)))
            for term_name, term_cfg in terms_cfg.items()
        }

    def reset(self):
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._default_pose_mode = False
        self._hold_default_pose = True
        for buffer in self._term_buffers.values():
            buffer.clear()

    def set_default_pose_mode(self, enabled: bool):
        self._default_pose_mode = bool(enabled)
        self._hold_default_pose = bool(enabled)
        if enabled:
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)

    def post_step_callback(self, commands: list[str] | None = None):
        del commands
        return

    def _push_term_history(self, term_name: str, value: np.ndarray) -> np.ndarray:
        history = self._term_buffers[term_name]
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        history.append(value)
        values = list(history)
        while len(values) < history.maxlen:
            values.insert(0, values[0].copy())
        return np.concatenate(values[-history.maxlen :], dtype=np.float32)

    def _build_default_ref_limb_pose_b(self) -> np.ndarray:
        term_cfg = self.term_cfgs.get("ref_limb_ee_pose_b", {})
        params = term_cfg.get("params", {})
        body_names = tuple(params.get("body_names", DEFAULT_WBTELEOP_LIMB_BODY_NAMES))
        anchor_body_name = params.get("anchor_body_name", DEFAULT_WBTELEOP_LIMB_ANCHOR_BODY_NAME)
        snapshot_body_names = (anchor_body_name, *body_names)

        try:
            import mujoco

            from robojudo.config import ASSETS_DIR

            model = mujoco.MjModel.from_xml_path((ASSETS_DIR / "robots/g1/g1_29dof_rev_1_0.xml").as_posix())
            data = mujoco.MjData(model)
            data.qpos[-self.num_dofs :] = np.asarray(self.default_dof_pos, dtype=np.float64)
            mujoco.mj_forward(model, data)

            body_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for body_name in snapshot_body_names
            ]
            missing = [name for name, body_id in zip(snapshot_body_names, body_ids, strict=True) if body_id < 0]
            if missing:
                raise KeyError(f"Missing G1 body names for default wbteleop limb pose: {missing}")

            snapshot = RetargetMotionSnapshot(
                body_names=snapshot_body_names,
                body_pos_w=np.asarray([data.xpos[body_id] for body_id in body_ids], dtype=np.float32),
                body_quat_w=np.asarray([data.xquat[body_id] for body_id in body_ids], dtype=np.float32),
                body_lin_vel_w=np.zeros((len(snapshot_body_names), 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((len(snapshot_body_names), 3), dtype=np.float32),
                timestamp_ns=0,
            )
            return extract_limb_pose_b_from_snapshot(
                snapshot,
                body_names=body_names,
                anchor_body_name=anchor_body_name,
            )
        except Exception as exc:
            logger.warning("Failed to build wbteleop default limb reference from MuJoCo: %s", exc)
            return np.zeros(_WBTELEOP_TERM_DIMS["ref_limb_ee_pose_b"], dtype=np.float32)

    def _fk_limb_pose_b(self, env_data, term_cfg: dict[str, Any]) -> np.ndarray:
        fk_info = getattr(env_data, "fk_info", None)
        if fk_info is None:
            return np.zeros(_WBTELEOP_TERM_DIMS["robot_limb_ee_pose_b"], dtype=np.float32)

        params = term_cfg.get("params", {})
        body_names = tuple(params.get("body_names", DEFAULT_WBTELEOP_LIMB_BODY_NAMES))
        anchor_body_name = params.get("anchor_body_name", DEFAULT_WBTELEOP_LIMB_ANCHOR_BODY_NAME)
        snapshot_body_names = (anchor_body_name, *body_names)
        try:
            body_pos_w = np.asarray([fk_info[name]["pos"] for name in snapshot_body_names], dtype=np.float32)
            body_quat_w = np.asarray(
                [quat_xyzw_to_wxyz(fk_info[name]["quat"]) for name in snapshot_body_names],
                dtype=np.float32,
            )
        except KeyError:
            return np.zeros(_WBTELEOP_TERM_DIMS["robot_limb_ee_pose_b"], dtype=np.float32)

        snapshot = RetargetMotionSnapshot(
            body_names=snapshot_body_names,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=np.zeros((len(snapshot_body_names), 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((len(snapshot_body_names), 3), dtype=np.float32),
            timestamp_ns=0,
        )
        return extract_limb_pose_b_from_snapshot(
            snapshot,
            body_names=body_names,
            anchor_body_name=anchor_body_name,
        )

    def _current_terms(self, env_data, ctrl: dict[str, Any]) -> dict[str, np.ndarray]:
        if self._hold_default_pose:
            command = np.concatenate(
                [
                    self.default_dof_pos.astype(np.float32),
                    np.zeros(self.num_dofs, dtype=np.float32),
                ],
                dtype=np.float32,
            )
            ref_limb_ee_pose_b = self._default_ref_limb_ee_pose_b.copy()
            motion_ref_ang_vel = np.zeros(3, dtype=np.float32)
        else:
            command = np.asarray(ctrl.get("command", np.zeros(58, dtype=np.float32)), dtype=np.float32)
            ref_limb_ee_pose_b = np.asarray(
                ctrl.get("ref_limb_ee_pose_b", np.zeros(36, dtype=np.float32)),
                dtype=np.float32,
            )
            motion_ref_ang_vel = np.asarray(
                ctrl.get("motion_ref_ang_vel", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            )

        terms = {
            "command": command,
            "ref_limb_ee_pose_b": ref_limb_ee_pose_b,
            "motion_ref_ang_vel": motion_ref_ang_vel,
            "projected_gravity": np.asarray(get_gravity_orientation(env_data.base_quat), dtype=np.float32),
            "base_ang_vel": np.asarray(env_data.base_ang_vel, dtype=np.float32),
            "joint_pos": np.asarray(env_data.dof_pos, dtype=np.float32) - self.default_dof_pos.astype(np.float32),
            "joint_vel": np.asarray(env_data.dof_vel, dtype=np.float32),
            "actions": self.last_action.astype(np.float32),
        }
        if "robot_limb_ee_pose_b" in ctrl:
            terms["robot_limb_ee_pose_b"] = np.asarray(ctrl["robot_limb_ee_pose_b"], dtype=np.float32)
        else:
            terms["robot_limb_ee_pose_b"] = self._fk_limb_pose_b(env_data, self.term_cfgs["robot_limb_ee_pose_b"])
        return terms

    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        if self.ctrl_type not in ctrl_data:
            raise KeyError(f"Controller data '{self.ctrl_type}' not found in ctrl_data.")
        ctrl = ctrl_data[self.ctrl_type]
        self._hold_default_pose = self._default_pose_mode or ctrl.get("state") != "active"
        if self._hold_default_pose:
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        current = self._current_terms(env_data, ctrl)

        obs_parts = []
        for term_name in self.term_order:
            value = np.asarray(current[term_name], dtype=np.float32).reshape(-1)
            expected_dim = _WBTELEOP_TERM_DIMS[term_name]
            if value.shape[0] != expected_dim:
                raise ValueError(
                    f"wbteleop term '{term_name}' dimension mismatch: expected {expected_dim}, got {value.shape[0]}"
                )
            obs_parts.append(self._push_term_history(term_name, value))

        obs = np.concatenate(obs_parts, dtype=np.float32)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"assembled wbteleop observation dimension mismatch: expected {self.obs_dim}, got {obs.shape[0]}"
            )
        return obs, {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self._hold_default_pose:
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)
            return self.last_action.copy()

        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"observation dimension mismatch: expected {self.obs_dim}, got {obs.shape[1]}")
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
        del env_data, extras
        if self.ctrl_type not in ctrl_data:
            return
        ctrl = ctrl_data[self.ctrl_type]
        qpos = ctrl.get("_ref_qpos")
        if qpos is not None and hasattr(visualizer, "update_ref_motion_ghost_mesh"):
            visualizer.update_ref_motion_ghost_mesh(np.asarray(qpos, dtype=np.float32))
            return

        body_pos = ctrl.get("_ref_body_pos_w")
        if body_pos is None:
            return
        body_names = ctrl.get("_ref_body_names")
        if hasattr(visualizer, "update_ref_motion_ghost"):
            visualizer.update_ref_motion_ghost(
                np.asarray(body_pos, dtype=np.float32),
                body_names=body_names,
            )
        else:
            visualizer.update_rg_view(
                np.asarray(body_pos, dtype=np.float32),
                np.zeros((len(body_pos), 4)),
                humanoid_id=1,
            )
