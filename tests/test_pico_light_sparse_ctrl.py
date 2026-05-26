from __future__ import annotations

import unittest

import numpy as np

from robojudo.controller.ctrl_cfgs import PicoLightSparseCtrlCfg
from robojudo.controller.pico_light_sparse_ctrl import PicoLightSparseCtrl

_DEFAULT_LEFT_EE_POSE = np.array(
    [
        0.09729591,
        0.21447651,
        -0.02440158,
        0.69704987,
        -0.01552948,
        0.15418759,
        0.97962287,
        -0.7002483,
        0.2002445,
    ],
    dtype=np.float32,
)
_DEFAULT_RIGHT_EE_POSE = np.array(
    [
        0.09729591,
        -0.21446651,
        -0.02440158,
        0.69704987,
        0.01552948,
        -0.15418759,
        0.97962287,
        -0.7002483,
        -0.2002445,
    ],
    dtype=np.float32,
)


class _FakeReader:
    def __init__(self, frames: list[dict]):
        self._frames = list(frames)
        self._last = frames[-1]

    def read(self) -> dict:
        if self._frames:
            self._last = self._frames.pop(0)
        return self._last


def _frame(
    *,
    timestamp_ns: int,
    left_axis=(0.0, 0.0),
    right_axis=(0.0, 0.0),
    left_trigger: float = 0.0,
    left_grip: float = 0.0,
    left_pos=(0.0, 0.0, 0.0),
    right_pos=(0.0, 0.0, 0.0),
    left_x: bool = False,
    right_a: bool = False,
    left_axis_click: bool = False,
    right_axis_click: bool = False,
) -> dict:
    return {
        "timestamp_ns": timestamp_ns,
        "left_axis": left_axis,
        "right_axis": right_axis,
        "left_trigger": left_trigger,
        "left_grip": left_grip,
        "right_trigger": 0.0,
        "right_grip": 0.0,
        "left_pos": np.asarray(left_pos, dtype=np.float32),
        "right_pos": np.asarray(right_pos, dtype=np.float32),
        "left_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "right_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "headset_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "left_x": left_x,
        "right_a": right_a,
        "left_axis_click": left_axis_click,
        "right_axis_click": right_axis_click,
    }


class TestPicoLightSparseCtrl(unittest.TestCase):
    def test_state_machine_cycles_and_shutdown_command(self):
        cfg = PicoLightSparseCtrlCfg()
        reader = _FakeReader(
            [
                _frame(timestamp_ns=0),
                _frame(timestamp_ns=1_000_000, right_a=True),
                _frame(timestamp_ns=2_000_000, right_a=False),
                _frame(timestamp_ns=3_000_000, right_a=True),
                _frame(timestamp_ns=4_000_000, right_a=False),
                _frame(timestamp_ns=5_000_000, left_x=True),
            ]
        )
        ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg, sdk_reader=reader)

        data = ctrl.get_data()
        self.assertEqual(data["state"], "idle")

        data = ctrl.get_data()
        self.assertEqual(data["state"], "active")

        data = ctrl.get_data()
        self.assertEqual(data["state"], "active")

        data = ctrl.get_data()
        self.assertEqual(data["state"], "pause")

        data = ctrl.get_data()
        self.assertEqual(data["state"], "pause")

        data = ctrl.get_data()
        _, commands = ctrl.process_triggers(data)
        self.assertEqual(data["state"], "exit")
        self.assertEqual(commands, ["[SHUTDOWN]"])

    def test_velocity_and_height_mapping_in_active_state(self):
        cfg = PicoLightSparseCtrlCfg(vx_scale=1.0, vy_scale=0.5, wz_scale=1.0, base_height_rate=0.3)
        reader = _FakeReader(
            [
                _frame(timestamp_ns=0, right_a=True),
                _frame(
                    timestamp_ns=100_000_000,
                    left_axis=(0.5, 1.0),
                    right_axis=(0.25, 0.0),
                    left_grip=1.0,
                    left_pos=(0.0, 0.1, -0.2),
                    right_pos=(0.0, 0.2, -0.1),
                ),
            ]
        )
        ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg, sdk_reader=reader)

        ctrl.get_data()  # enter active and latch anchors
        data = ctrl.get_data()

        np.testing.assert_allclose(data["base_lin_vel_b"], [1.0, -0.25, 0.0], atol=1e-6)
        np.testing.assert_allclose(data["base_ang_vel_b"], [0.0, 0.0, -0.25], atol=1e-6)
        self.assertGreater(data["anchor_height_w"][0], cfg.base_height_init)
        self.assertEqual(data["ee_pose"].shape, (18,))
        np.testing.assert_allclose(data["ee_pose"][:3], _DEFAULT_LEFT_EE_POSE[:3] + [0.16, 0.0, 0.08], atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"][3:9], _DEFAULT_LEFT_EE_POSE[3:9], atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"][9:12], _DEFAULT_RIGHT_EE_POSE[:3] + [0.08, 0.0, 0.16], atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"][12:18], _DEFAULT_RIGHT_EE_POSE[3:9], atol=1e-6)

    def test_idle_and_active_no_move_use_training_default_wrist_pose(self):
        cfg = PicoLightSparseCtrlCfg()
        reader = _FakeReader(
            [
                _frame(timestamp_ns=0),
                _frame(timestamp_ns=1_000_000, right_a=True),
                _frame(timestamp_ns=2_000_000, right_a=False),
            ]
        )
        ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg, sdk_reader=reader)

        idle = ctrl.get_data()
        active_anchor = ctrl.get_data()
        active_no_move = ctrl.get_data()

        expected = np.concatenate([_DEFAULT_LEFT_EE_POSE, _DEFAULT_RIGHT_EE_POSE])
        np.testing.assert_allclose(idle["ee_pose"], expected, atol=1e-6)
        np.testing.assert_allclose(active_anchor["ee_pose"], expected, atol=1e-6)
        np.testing.assert_allclose(active_no_move["ee_pose"], expected, atol=1e-6)

    def test_pause_freezes_last_commands(self):
        cfg = PicoLightSparseCtrlCfg()
        reader = _FakeReader(
            [
                _frame(timestamp_ns=0, right_a=True),
                _frame(timestamp_ns=100_000_000, left_axis=(0.0, 1.0), right_axis=(0.5, 0.0)),
                _frame(timestamp_ns=200_000_000, right_a=True),
                _frame(timestamp_ns=300_000_000, left_axis=(0.0, -1.0), right_axis=(-0.5, 0.0)),
            ]
        )
        ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg, sdk_reader=reader)

        ctrl.get_data()
        active = ctrl.get_data()
        paused = ctrl.get_data()
        frozen = ctrl.get_data()

        self.assertEqual(paused["state"], "pause")
        np.testing.assert_allclose(frozen["base_lin_vel_b"], active["base_lin_vel_b"])
        np.testing.assert_allclose(frozen["base_ang_vel_b"], active["base_ang_vel_b"])
