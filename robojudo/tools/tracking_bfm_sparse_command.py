from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_ANCHOR_BODY_NAME = "pelvis"
DEFAULT_EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


@dataclass(frozen=True)
class RetargetMotionSnapshot:
    body_names: tuple[str, ...]
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    timestamp_ns: int
    qpos: np.ndarray | None = None


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def quat_apply_inverse_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    rot = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    return rot.inv().apply(np.asarray(vec, dtype=np.float32)).astype(np.float32)


def _body_index(snapshot: RetargetMotionSnapshot, body_name: str) -> int:
    try:
        return snapshot.body_names.index(body_name)
    except ValueError as exc:
        raise KeyError(f"Body '{body_name}' not found in retarget snapshot.") from exc


def _relative_pose_wxyz(
    anchor_pos_w: np.ndarray,
    anchor_quat_w: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_rot = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_w))
    body_rot = R.from_quat(quat_wxyz_to_xyzw(body_quat_w))
    pos_b = anchor_rot.inv().apply(np.asarray(body_pos_w, dtype=np.float32) - np.asarray(anchor_pos_w, dtype=np.float32))
    quat_b = quat_xyzw_to_wxyz((anchor_rot.inv() * body_rot).as_quat())
    return pos_b.astype(np.float32), quat_b.astype(np.float32)


def rot6d_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    rotmat = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_matrix()
    return rotmat[:, :2].reshape(-1).astype(np.float32)


def extract_tracking_bfm_sparse_command(
    snapshot: RetargetMotionSnapshot,
    *,
    anchor_body_name: str = DEFAULT_ANCHOR_BODY_NAME,
    ee_body_names: tuple[str, str] = DEFAULT_EE_BODY_NAMES,
    state: str = "active",
) -> dict[str, Any]:
    """Extract RoboJuDo command terms with the UNICTL tracking_bfm training semantics."""
    anchor_idx = _body_index(snapshot, anchor_body_name)
    ee_indices = [_body_index(snapshot, name) for name in ee_body_names]

    anchor_pos_w = np.asarray(snapshot.body_pos_w[anchor_idx], dtype=np.float32)
    anchor_quat_w = np.asarray(snapshot.body_quat_w[anchor_idx], dtype=np.float32)
    anchor_lin_vel_w = np.asarray(snapshot.body_lin_vel_w[anchor_idx], dtype=np.float32)
    anchor_ang_vel_w = np.asarray(snapshot.body_ang_vel_w[anchor_idx], dtype=np.float32)

    ee_parts = []
    for body_idx in ee_indices:
        pos_b, quat_b = _relative_pose_wxyz(
            anchor_pos_w,
            anchor_quat_w,
            snapshot.body_pos_w[body_idx],
            snapshot.body_quat_w[body_idx],
        )
        ee_parts.extend([pos_b, rot6d_from_quat_wxyz(quat_b)])

    return {
        "ee_pose": np.concatenate(ee_parts, dtype=np.float32),
        "base_lin_vel_b": quat_apply_inverse_wxyz(anchor_quat_w, anchor_lin_vel_w),
        "base_ang_vel_b": quat_apply_inverse_wxyz(anchor_quat_w, anchor_ang_vel_w),
        "anchor_height_w": np.array([anchor_pos_w[2]], dtype=np.float32),
        "state": state,
        "timestamp_ns": int(snapshot.timestamp_ns),
        "_commands": [],
    }


class MujocoRetargetSnapshotBuilder:
    """Build body-field snapshots from retargeted MuJoCo qpos.

    MuJoCo does not expose training-side reference velocities for arbitrary
    offline qpos streams, so this builder estimates body velocities from
    consecutive FK results. If an upstream retargeter exposes exact
    ``body_lin_vel_w``/``body_ang_vel_w`` fields, prefer those fields instead.
    """

    def __init__(self, model, data, body_names: tuple[str, ...]):
        self.model = model
        self.data = data
        self.body_names = tuple(body_names)
        self.body_ids = [model.body(name).id for name in self.body_names]
        self._last_timestamp_ns: int | None = None
        self._last_body_pos_w: np.ndarray | None = None
        self._last_body_quat_w: np.ndarray | None = None

    def build(self, qpos: np.ndarray, timestamp_ns: int) -> RetargetMotionSnapshot:
        import mujoco as mj

        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos dimension mismatch: model expects {self.model.nq}, got {qpos.shape[0]}")

        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

        body_pos_w = np.asarray([self.data.xpos[body_id].copy() for body_id in self.body_ids], dtype=np.float32)
        body_quat_w = np.asarray([self.data.xquat[body_id].copy() for body_id in self.body_ids], dtype=np.float32)

        body_lin_vel_w = np.zeros_like(body_pos_w, dtype=np.float32)
        body_ang_vel_w = np.zeros_like(body_pos_w, dtype=np.float32)
        if self._last_timestamp_ns is not None and timestamp_ns > self._last_timestamp_ns:
            dt = max((timestamp_ns - self._last_timestamp_ns) * 1e-9, 1e-6)
            body_lin_vel_w = ((body_pos_w - self._last_body_pos_w) / dt).astype(np.float32)
            body_ang_vel_w = np.asarray(
                [
                    (R.from_quat(quat_wxyz_to_xyzw(quat)) * R.from_quat(quat_wxyz_to_xyzw(last_quat)).inv()).as_rotvec()
                    / dt
                    for last_quat, quat in zip(self._last_body_quat_w, body_quat_w, strict=True)
                ],
                dtype=np.float32,
            )

        self._last_timestamp_ns = int(timestamp_ns)
        self._last_body_pos_w = body_pos_w.copy()
        self._last_body_quat_w = body_quat_w.copy()

        return RetargetMotionSnapshot(
            body_names=self.body_names,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            timestamp_ns=int(timestamp_ns),
            qpos=qpos.copy(),
        )
