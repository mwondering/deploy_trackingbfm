from __future__ import annotations

import numpy as np

from robojudo.tools.streamed_command_interpolator import (
    StreamedCommandInterpolator,
    quat_slerp_wxyz,
    quat_wxyz_from_rot6d,
    slerp_rot6d,
)
from robojudo.tools.tracking_bfm_sparse_command import rot6d_from_quat_wxyz


def _ctrl(timestamp_s, joint_pos, *, limb=None, ang_vel=None, joint_dof=3, state="active"):
    command = np.concatenate([np.asarray(joint_pos, dtype=np.float32), np.zeros(joint_dof, dtype=np.float32)])
    ctrl = {"state": state, "timestamp_ns": int(timestamp_s * 1e9), "command": command}
    if limb is not None:
        ctrl["ref_limb_ee_pose_b"] = np.asarray(limb, dtype=np.float32)
    if ang_vel is not None:
        ctrl["motion_ref_ang_vel"] = np.asarray(ang_vel, dtype=np.float32)
    return ctrl


# -- quaternion / rot6d helpers -------------------------------------------------

def test_quat_slerp_endpoints_and_midpoint_unit_norm():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = rot6d_to_quat_90z = quat_wxyz_from_rot6d(rot6d_from_quat_wxyz(_quat_z(np.pi / 2)))
    np.testing.assert_allclose(quat_slerp_wxyz(q0, q1, 0.0), q0, atol=1e-6)
    mid = quat_slerp_wxyz(q0, q1, 0.5)
    np.testing.assert_allclose(np.linalg.norm(mid), 1.0, atol=1e-5)


def test_slerp_rot6d_endpoints_round_trip():
    r0 = rot6d_from_quat_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    r1 = rot6d_from_quat_wxyz(_quat_z(np.pi / 2))
    np.testing.assert_allclose(slerp_rot6d(r0, r1, 0.0), r0, atol=1e-5)
    np.testing.assert_allclose(slerp_rot6d(r0, r1, 1.0), r1, atol=1e-5)
    # midpoint is a valid 45-degree rotation about z
    mid = slerp_rot6d(r0, r1, 0.5)
    expected = rot6d_from_quat_wxyz(_quat_z(np.pi / 4))
    np.testing.assert_allclose(mid, expected, atol=1e-4)


def _quat_z(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=np.float64)


# -- interpolator playback ------------------------------------------------------

def test_linear_interpolation_between_two_frames():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=None
    )
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0]))  # init, playback=0.0
    a = _ctrl(1.0, [1.0, 1.0, 1.0])
    interp.update(a)  # playback advances to 0.1 -> alpha 0.1 between t=0 and t=1
    np.testing.assert_allclose(a["command"][:3], np.full(3, 0.1), atol=1e-5)


def test_repeated_latest_frame_advances_playback_monotonically_without_velocity_spike():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=None
    )
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0]))
    prev = 0.0
    max_vel = 0.0
    for _ in range(15):
        c = _ctrl(1.0, [1.0, 1.0, 1.0])  # same timestamp -> held (dup rejected), playback still advances
        interp.update(c)
        pos = float(c["command"][0])
        vel = float(c["command"][3])
        assert pos >= prev - 1e-6
        prev = pos
        max_vel = max(max_vel, abs(vel))
    # converged to the held target, and the velocity never spiked like a raw step would
    np.testing.assert_allclose(prev, 1.0, atol=1e-6)
    assert max_vel < 1.0 / 0.1  # below the 1.0/dt raw finite-difference spike


def test_stale_and_out_of_order_frames_are_rejected():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=None
    )
    interp.update(_ctrl(1.0, [1.0, 1.0, 1.0]))
    interp.update(_ctrl(2.0, [2.0, 2.0, 2.0]))
    assert len(interp._buffer) == 2
    interp.update(_ctrl(1.5, [9.0, 9.0, 9.0]))  # older than newest -> rejected
    interp.update(_ctrl(2.0, [9.0, 9.0, 9.0]))  # duplicate timestamp -> rejected
    assert len(interp._buffer) == 2


def test_dropout_holds_last_frame():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=None
    )
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0]))
    interp.update(_ctrl(0.1, [1.0, 1.0, 1.0]))
    # source stalls: keep feeding the same newest frame; playback runs past it and holds
    for _ in range(10):
        c = _ctrl(0.1, [1.0, 1.0, 1.0])
        interp.update(c)
    np.testing.assert_allclose(c["command"][:3], np.full(3, 1.0), atol=1e-6)
    np.testing.assert_allclose(c["command"][3:], np.zeros(3), atol=1e-6)  # velocity decays to zero on hold


def test_catch_up_reset_bounds_latency():
    interp = StreamedCommandInterpolator(
        control_dt=0.02, joint_dof=3, target_lag_s=0.033, max_lag_s=0.2, history_s=10.0, joint_snap_threshold=None
    )
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0]))
    # large time gap (source resumes after a long stall) -> playback snaps forward
    c = _ctrl(5.0, [5.0, 5.0, 5.0])
    interp.update(c)
    # catch-up discards the stale history and restarts at the newest frame
    assert 5.0 - interp._playback_t <= 0.2 + 1e-6
    assert len(interp._buffer) == 1
    np.testing.assert_allclose(c["command"][:3], np.full(3, 5.0), atol=1e-4)


def test_teleport_guard_snaps_instead_of_blending():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=1.0
    )
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0]))
    c = _ctrl(1.0, [10.0, 10.0, 10.0])  # jump of 10 rad > 1.0 threshold
    interp.update(c)  # playback at 0.1 would blend to 1.0, but snap -> newer frame
    np.testing.assert_allclose(c["command"][:3], np.full(3, 10.0), atol=1e-6)
    np.testing.assert_allclose(c["command"][3:], np.zeros(3), atol=1e-6)  # no velocity spike across teleport


def test_reset_when_not_active():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=None
    )
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0]))
    interp.update(_ctrl(1.0, [1.0, 1.0, 1.0]))
    idle = {"state": "idle"}
    interp.update(idle)
    assert idle == {"state": "idle"}
    assert len(interp._buffer) == 0
    assert interp._playback_t is None


def test_limb_pose_position_lerp_and_rotation_slerp():
    interp = StreamedCommandInterpolator(
        control_dt=0.1, joint_dof=3, target_lag_s=0.0, max_lag_s=100.0, history_s=100.0, joint_snap_threshold=None
    )
    r0 = rot6d_from_quat_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    r1 = rot6d_from_quat_wxyz(_quat_z(np.pi / 2))
    limb0 = np.concatenate([np.zeros(3, dtype=np.float32), r0])
    limb1 = np.concatenate([np.full(3, 1.0, dtype=np.float32), r1])
    interp.update(_ctrl(0.0, [0.0, 0.0, 0.0], limb=limb0))
    c = _ctrl(1.0, [0.0, 0.0, 0.0], limb=limb1)
    interp.update(c)  # alpha 0.1
    out = c["ref_limb_ee_pose_b"]
    np.testing.assert_allclose(out[:3], np.full(3, 0.1), atol=1e-5)  # position lerp
    expected_rot = rot6d_from_quat_wxyz(_quat_z(0.1 * np.pi / 2))
    np.testing.assert_allclose(out[3:9], expected_rot, atol=1e-4)  # rotation slerp
