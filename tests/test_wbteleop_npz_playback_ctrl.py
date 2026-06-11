from __future__ import annotations

import numpy as np

from robojudo.controller.ctrl_cfgs import WbTeleopNpzPlaybackCtrlCfg
from robojudo.controller.wbteleop_npz_playback_ctrl import WbTeleopNpzPlaybackCtrl


def _write_motion_npz(path, *, frames: int = 3) -> None:
    joint_pos = np.stack([np.full(29, frame * 0.1, dtype=np.float32) for frame in range(frames)])
    joint_vel = np.stack([np.full(29, frame + 1, dtype=np.float32) for frame in range(frames)])
    body_pos_w = np.zeros((frames, 30, 3), dtype=np.float32)
    body_quat_w = np.zeros((frames, 30, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((frames, 30, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((frames, 30, 3), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_pos_w[:, 0, 2] = 0.8
    body_ang_vel_w[:, 15, :] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
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


def test_npz_playback_outputs_wbteleop_terms_and_starts_motion(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file)
    ctrl = WbTeleopNpzPlaybackCtrl(
        WbTeleopNpzPlaybackCtrlCfg(motion_file=motion_file.as_posix(), motion_type="mujoco", auto_start=True)
    )

    first = ctrl.get_data()
    first_processed, first_commands = ctrl.process_triggers(first)
    second = ctrl.get_data()
    second_processed, second_commands = ctrl.process_triggers(second)

    assert first_processed["state"] == "active"
    assert first_commands == ["[MOTION_RESET]"]
    assert second_commands == []
    assert second_processed["command"].shape == (58,)
    assert second_processed["ref_limb_ee_pose_b"].shape == (36,)
    assert second_processed["motion_ref_ang_vel"].shape == (3,)
    np.testing.assert_allclose(second_processed["command"][:29], np.zeros(29, dtype=np.float32))
    np.testing.assert_allclose(second_processed["command"][29:], np.ones(29, dtype=np.float32))
    assert second_processed["_ref_qpos"].shape == (36,)


def test_npz_playback_waits_for_external_reset_when_manual(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file)
    ctrl = WbTeleopNpzPlaybackCtrl(
        WbTeleopNpzPlaybackCtrlCfg(motion_file=motion_file.as_posix(), motion_type="mujoco")
    )

    idle = ctrl.get_data()
    idle_processed, idle_commands = ctrl.process_triggers(idle)

    assert idle_processed["state"] == "idle"
    assert idle_commands == []

    ctrl.post_step_callback(["[MOTION_RESET]"])
    active = ctrl.get_data()
    active_processed, active_commands = ctrl.process_triggers(active)

    assert active_processed["state"] == "active"
    assert active_commands == []
    np.testing.assert_allclose(active_processed["command"][:29], np.zeros(29, dtype=np.float32))
    np.testing.assert_allclose(active_processed["command"][29:], np.ones(29, dtype=np.float32))


def test_npz_playback_manual_fade_out_returns_to_idle(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file)
    ctrl = WbTeleopNpzPlaybackCtrl(
        WbTeleopNpzPlaybackCtrlCfg(motion_file=motion_file.as_posix(), motion_type="mujoco")
    )

    ctrl.post_step_callback(["[MOTION_RESET]"])
    assert ctrl.process_triggers(ctrl.get_data())[0]["state"] == "active"

    ctrl.post_step_callback(["[MOTION_FADE_OUT]"])
    idle_processed, idle_commands = ctrl.process_triggers(ctrl.get_data())

    assert idle_processed["state"] == "idle"
    assert idle_commands == []


def test_npz_playback_fades_out_after_last_frame(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file, frames=1)
    ctrl = WbTeleopNpzPlaybackCtrl(
        WbTeleopNpzPlaybackCtrlCfg(
            motion_file=motion_file.as_posix(), motion_type="mujoco", loop=False, auto_start=True
        )
    )

    ctrl.process_triggers(ctrl.get_data())
    ctrl.process_triggers(ctrl.get_data())
    done = ctrl.get_data()
    processed, commands = ctrl.process_triggers(done)

    assert processed["state"] == "done"
    assert commands == ["[MOTION_FADE_OUT]"]


def test_npz_playback_loop_restarts_motion(tmp_path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_npz(motion_file, frames=1)
    ctrl = WbTeleopNpzPlaybackCtrl(
        WbTeleopNpzPlaybackCtrlCfg(motion_file=motion_file.as_posix(), motion_type="mujoco", loop=True, auto_start=True)
    )

    ctrl.process_triggers(ctrl.get_data())
    ctrl.process_triggers(ctrl.get_data())
    restarted = ctrl.get_data()
    _processed, commands = ctrl.process_triggers(restarted)

    assert commands == ["[MOTION_RESET]"]
