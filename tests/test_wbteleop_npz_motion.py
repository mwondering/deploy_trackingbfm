from __future__ import annotations

import numpy as np

from robojudo.tools.tracking_bfm_wbteleop_command import DEFAULT_WBTELEOP_MOTION_BODY_NAMES
from robojudo.tools.wbteleop_npz_motion import WbTeleopNpzMotionLoader


def _write_motion_npz(path, *, body_count: int = 30, frames: int = 2) -> None:
    joint_pos = np.tile(np.arange(29, dtype=np.float32), (frames, 1))
    joint_vel = np.tile(np.arange(100, 129, dtype=np.float32), (frames, 1))
    body_pos_w = np.zeros((frames, body_count, 3), dtype=np.float32)
    body_quat_w = np.zeros((frames, body_count, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((frames, body_count, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((frames, body_count, 3), dtype=np.float32)
    for body_idx in range(body_count):
        body_pos_w[:, body_idx, 0] = body_idx
        body_quat_w[:, body_idx, 0] = 1.0
        body_lin_vel_w[:, body_idx, 1] = body_idx + 10
        body_ang_vel_w[:, body_idx, 2] = body_idx + 20
    body_pos_w[:, 0, 2] = 0.8
    np.savez(
        path,
        fps=np.array(50, dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


def test_isaaclab_motion_is_reindexed_to_mujoco_order(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file)

    loader = WbTeleopNpzMotionLoader(motion_file.as_posix(), motion_type="isaaclab")
    snapshot = loader.snapshot_at(0, timestamp_ns=123)

    assert snapshot.body_names == DEFAULT_WBTELEOP_MOTION_BODY_NAMES
    assert snapshot.timestamp_ns == 123
    assert snapshot.qpos.shape == (36,)
    assert snapshot.joint_vel is not None
    assert snapshot.qpos[-29 + 1] == 3.0
    assert snapshot.joint_vel[1] == 103.0
    left_hip_roll_idx = snapshot.body_names.index("left_hip_roll_link")
    assert snapshot.body_pos_w[left_hip_roll_idx, 0] == 4.0


def test_mujoco_motion_keeps_existing_joint_and_body_order(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file)

    loader = WbTeleopNpzMotionLoader(motion_file.as_posix(), motion_type="mujoco")
    snapshot = loader.snapshot_at(0, timestamp_ns=123)

    assert snapshot.qpos[-29 + 1] == 1.0
    left_hip_roll_idx = snapshot.body_names.index("left_hip_roll_link")
    assert snapshot.body_pos_w[left_hip_roll_idx, 0] == 2.0


def test_motion_loader_requires_tracking_bfm_npz_keys(tmp_path) -> None:
    motion_file = tmp_path / "bad_motion.npz"
    np.savez(motion_file, fps=np.array(50, dtype=np.int64), joint_pos=np.zeros((1, 29), dtype=np.float32))

    try:
        WbTeleopNpzMotionLoader(motion_file.as_posix())
    except KeyError as exc:
        assert "joint_vel" in str(exc)
    else:
        raise AssertionError("expected missing npz key to raise KeyError")
