from __future__ import annotations

from typing import Any

import numpy as np

from robojudo.tools.tracking_bfm_sparse_command import (
    RetargetMotionSnapshot,
    _body_index,
    _relative_pose_wxyz,
    rot6d_from_quat_wxyz,
)

DEFAULT_WBTELEOP_LIMB_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
DEFAULT_WBTELEOP_LIMB_ANCHOR_BODY_NAME = "pelvis"
DEFAULT_WBTELEOP_COMMAND_ANCHOR_BODY_NAME = "torso_link"
DEFAULT_WBTELEOP_MOTION_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def extract_limb_pose_b_from_snapshot(
    snapshot: RetargetMotionSnapshot,
    *,
    body_names: tuple[str, ...] = DEFAULT_WBTELEOP_LIMB_BODY_NAMES,
    anchor_body_name: str = DEFAULT_WBTELEOP_LIMB_ANCHOR_BODY_NAME,
) -> np.ndarray:
    anchor_idx = _body_index(snapshot, anchor_body_name)
    anchor_pos_w = np.asarray(snapshot.body_pos_w[anchor_idx], dtype=np.float32)
    anchor_quat_w = np.asarray(snapshot.body_quat_w[anchor_idx], dtype=np.float32)

    pieces = []
    for body_name in body_names:
        body_idx = _body_index(snapshot, body_name)
        pos_b, quat_b = _relative_pose_wxyz(
            anchor_pos_w,
            anchor_quat_w,
            snapshot.body_pos_w[body_idx],
            snapshot.body_quat_w[body_idx],
        )
        pieces.extend([pos_b, rot6d_from_quat_wxyz(quat_b)])
    return np.concatenate(pieces, dtype=np.float32)


class WbTeleopRetargetCommandExtractor:
    """Extract wbteleop actor command terms from consecutive retargeted qpos snapshots."""

    def __init__(
        self,
        *,
        joint_dof: int = 29,
        limb_body_names: tuple[str, ...] = DEFAULT_WBTELEOP_LIMB_BODY_NAMES,
        limb_anchor_body_name: str = DEFAULT_WBTELEOP_LIMB_ANCHOR_BODY_NAME,
        command_anchor_body_name: str = DEFAULT_WBTELEOP_COMMAND_ANCHOR_BODY_NAME,
    ):
        self.joint_dof = int(joint_dof)
        self.limb_body_names = tuple(limb_body_names)
        self.limb_anchor_body_name = limb_anchor_body_name
        self.command_anchor_body_name = command_anchor_body_name
        self._last_joint_pos: np.ndarray | None = None
        self._last_timestamp_ns: int | None = None

    def reset(self):
        self._last_joint_pos = None
        self._last_timestamp_ns = None

    def _joint_pos_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.shape[0] < self.joint_dof:
            raise ValueError(f"retarget qpos has {qpos.shape[0]} values, expected at least {self.joint_dof}")
        return qpos[-self.joint_dof :].astype(np.float32)

    def _joint_vel(self, joint_pos: np.ndarray, timestamp_ns: int) -> np.ndarray:
        joint_vel = np.zeros(self.joint_dof, dtype=np.float32)
        has_previous = self._last_joint_pos is not None and self._last_timestamp_ns is not None
        if has_previous and timestamp_ns > self._last_timestamp_ns:
            dt = max((timestamp_ns - self._last_timestamp_ns) * 1e-9, 1e-6)
            joint_vel = ((joint_pos - self._last_joint_pos) / dt).astype(np.float32)
        self._last_joint_pos = joint_pos.copy()
        self._last_timestamp_ns = int(timestamp_ns)
        return joint_vel

    def extract(self, snapshot: RetargetMotionSnapshot, *, state: str = "active") -> dict[str, Any]:
        if snapshot.qpos is None:
            raise ValueError("wbteleop command extraction requires retarget snapshot.qpos.")

        joint_pos = self._joint_pos_from_qpos(snapshot.qpos)
        joint_vel = self._joint_vel(joint_pos, int(snapshot.timestamp_ns))
        command_anchor_idx = _body_index(snapshot, self.command_anchor_body_name)

        return {
            "command": np.concatenate([joint_pos, joint_vel], dtype=np.float32),
            "ref_limb_ee_pose_b": extract_limb_pose_b_from_snapshot(
                snapshot,
                body_names=self.limb_body_names,
                anchor_body_name=self.limb_anchor_body_name,
            ),
            "motion_ref_ang_vel": np.asarray(snapshot.body_ang_vel_w[command_anchor_idx], dtype=np.float32).reshape(3),
            "state": state,
            "timestamp_ns": int(snapshot.timestamp_ns),
            "_ref_body_names": tuple(snapshot.body_names),
            "_ref_body_pos_w": np.asarray(snapshot.body_pos_w, dtype=np.float32).copy(),
            "_ref_body_quat_w": np.asarray(snapshot.body_quat_w, dtype=np.float32).copy(),
            "_ref_qpos": np.asarray(snapshot.qpos, dtype=np.float32).copy(),
        }
