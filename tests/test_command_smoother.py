from __future__ import annotations

import numpy as np

from robojudo.tools.command_smoother import CriticallyDampedFilter, WbTeleopCommandSmoother


def test_filter_snaps_to_first_sample_with_zero_velocity():
    f = CriticallyDampedFilter(cutoff_hz=10.0)
    x, v = f.update([1.0, 2.0, 3.0], dt=0.02)
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(v, [0.0, 0.0, 0.0])


def test_filter_converges_to_constant_target_without_overshoot():
    f = CriticallyDampedFilter(cutoff_hz=10.0)
    f.update([0.0], dt=0.02)
    prev = 0.0
    for _ in range(200):
        x, v = f.update([1.0], dt=0.02)
        # critically damped: monotone approach, never overshoots past the target
        assert prev - 1e-5 <= x[0] <= 1.0 + 1e-5
        prev = x[0]
    np.testing.assert_allclose(x, [1.0], atol=1e-3)
    np.testing.assert_allclose(v, [0.0], atol=1e-2)


def test_filter_velocity_is_bounded_and_smooth_for_step_input():
    # A held-then-jump staircase should produce a smooth, bounded velocity
    # rather than the single huge finite-difference spike of the raw signal.
    f = CriticallyDampedFilter(cutoff_hz=10.0)
    f.update([0.0], dt=0.02)
    max_v = 0.0
    for _ in range(100):
        _, v = f.update([1.0], dt=0.02)  # 1.0 rad step held constant
        max_v = max(max_v, abs(v[0]))
    raw_spike = 1.0 / 0.02  # 50 rad/s if differenced over one control step
    assert max_v < raw_spike  # smoothed peak velocity is well below the raw spike


def test_filter_snaps_on_teleport_above_threshold():
    f = CriticallyDampedFilter(cutoff_hz=10.0, max_jump=0.5)
    f.update([0.0], dt=0.02)
    x, v = f.update([10.0], dt=0.02)  # jump of 10 > 0.5 -> snap
    np.testing.assert_allclose(x, [10.0])
    np.testing.assert_allclose(v, [0.0])


def test_command_smoother_replaces_joint_velocity_with_filter_velocity():
    smoother = WbTeleopCommandSmoother(cutoff_hz=10.0, joint_dof=3)
    raw_vel = np.full(3, 999.0, dtype=np.float32)  # spiky finite-diff velocity
    ctrl = {
        "state": "active",
        "command": np.concatenate([np.zeros(3, dtype=np.float32), raw_vel]),
    }
    smoother.smooth(ctrl, dt=0.02)
    out_vel = ctrl["command"][3:]
    # first sample snaps -> filter velocity is zero, replacing the spike
    np.testing.assert_allclose(out_vel, np.zeros(3), atol=1e-6)
    assert not np.allclose(out_vel, raw_vel)


def test_command_smoother_smooths_limb_and_angvel_fields():
    smoother = WbTeleopCommandSmoother(cutoff_hz=10.0, joint_dof=3)
    target_limb = np.ones(36, dtype=np.float32)
    target_ang = np.ones(3, dtype=np.float32)
    # prime at zero, then push toward ones
    smoother.smooth(
        {
            "state": "active",
            "command": np.zeros(6, dtype=np.float32),
            "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
            "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
        },
        dt=0.02,
    )
    ctrl = {
        "state": "active",
        "command": np.zeros(6, dtype=np.float32),
        "ref_limb_ee_pose_b": target_limb.copy(),
        "motion_ref_ang_vel": target_ang.copy(),
    }
    smoother.smooth(ctrl, dt=0.02)
    # one control step toward the target: strictly between old (0) and target (1)
    assert np.all(ctrl["ref_limb_ee_pose_b"] > 0.0)
    assert np.all(ctrl["ref_limb_ee_pose_b"] < 1.0)
    assert np.all(ctrl["motion_ref_ang_vel"] > 0.0)
    assert np.all(ctrl["motion_ref_ang_vel"] < 1.0)


def test_command_smoother_resets_when_not_active():
    smoother = WbTeleopCommandSmoother(cutoff_hz=10.0, joint_dof=3)
    smoother.smooth(
        {"state": "active", "command": np.zeros(6, dtype=np.float32)},
        dt=0.02,
    )
    # idle frame should reset internal state and leave the payload untouched
    idle = {"state": "idle"}
    smoother.smooth(idle, dt=0.02)
    assert idle == {"state": "idle"}
    # after reset, the next active sample snaps again (velocity zero)
    ctrl = {"state": "active", "command": np.concatenate([np.full(3, 5.0), np.full(3, 7.0)]).astype(np.float32)}
    smoother.smooth(ctrl, dt=0.02)
    np.testing.assert_allclose(ctrl["command"][:3], np.full(3, 5.0))
    np.testing.assert_allclose(ctrl["command"][3:], np.zeros(3), atol=1e-6)
