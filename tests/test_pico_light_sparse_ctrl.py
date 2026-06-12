from __future__ import annotations

import time
import unittest
from queue import Queue
from unittest.mock import patch

import numpy as np

from robojudo.controller.ctrl_cfgs import PicoLightSparseCtrlCfg
from robojudo.controller.pico_light_sparse_ctrl import PicoLightSparseCtrl, _RetargetPicoReader

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


class _FakeEePoseSource:
    def __init__(self, ee_poses: list[np.ndarray | None]):
        self._ee_poses = list(ee_poses)
        self.calls: list[int] = []

    def get_ee_pose(self, timestamp_ns: int) -> np.ndarray | None:
        self.calls.append(timestamp_ns)
        if self._ee_poses:
            return self._ee_poses.pop(0)
        return None


class _FakeRetargetReader:
    def __init__(self, cfg):
        self.cfg = cfg

    def read(self) -> dict:
        return _frame(timestamp_ns=0)


class _FakeRetargetStreamer:
    def get_current_frame(self):
        qpos = np.zeros(29, dtype=np.float32)
        return (
            {"qpos": qpos},
            None,
            None,
            {
                "LeftController": {
                    "index_trig": 0.7,
                    "grip": 0.2,
                    "axis": (0.3, -0.4),
                    "key_one": True,
                    "axis_click": False,
                },
                "RightController": {
                    "index_trig": 0.1,
                    "grip": 0.9,
                    "axis": (-0.5, 0.6),
                    "key_one": False,
                    "axis_click": True,
                },
                "timestamp": 123_456_789,
            },
            None,
        )


class _FakeLightRetarget:
    def retarget(self, smplx_data, offset_to_ground=True):
        del offset_to_ground
        return np.asarray(smplx_data["qpos"], dtype=np.float32)


class _FakeLightSnapshotBuilder:
    def build(self, qpos, timestamp_ns):
        from robojudo.tools.tracking_bfm_sparse_command import RetargetMotionSnapshot

        del qpos
        return RetargetMotionSnapshot(
            body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
            body_pos_w=np.array(
                [
                    [0.0, 0.0, 0.8],
                    [0.2, 0.1, 1.0],
                    [-0.2, -0.1, 1.0],
                ],
                dtype=np.float32,
            ),
            body_quat_w=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            body_lin_vel_w=np.zeros((3, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((3, 3), dtype=np.float32),
            timestamp_ns=timestamp_ns,
            qpos=np.zeros(36, dtype=np.float32),
        )


class _QueueReader:
    def __init__(self):
        self.frames = Queue()

    def read(self) -> dict:
        return self.frames.get()


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


def _cfg(**kwargs) -> PicoLightSparseCtrlCfg:
    return PicoLightSparseCtrlCfg(async_read=False, **kwargs)


class TestPicoLightSparseCtrl(unittest.TestCase):
    def test_state_machine_cycles_and_shutdown_command(self):
        cfg = _cfg()
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
        cfg = _cfg(vx_scale=1.0, vy_scale=0.5, wz_scale=1.0, base_height_rate=0.3)
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

    def test_retarget_ee_pose_source_overrides_light_controller_delta_mapping(self):
        retarget_ee_pose = np.linspace(-0.4, 0.4, 18, dtype=np.float32)
        cfg = _cfg(vx_scale=1.0, vy_scale=0.5, retarget_ee_pose=True)
        reader = _FakeReader(
            [
                _frame(timestamp_ns=0, right_a=True),
                _frame(
                    timestamp_ns=100_000_000,
                    left_axis=(0.5, 1.0),
                    left_pos=(0.0, 0.1, -0.2),
                    right_pos=(0.0, 0.2, -0.1),
                ),
            ]
        )
        ee_pose_source = _FakeEePoseSource([retarget_ee_pose, retarget_ee_pose])
        ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg, sdk_reader=reader, ee_pose_source=ee_pose_source)

        ctrl.get_data()  # enter active and latch anchors
        data = ctrl.get_data()

        np.testing.assert_allclose(data["base_lin_vel_b"], [1.0, -0.25, 0.0], atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"], retarget_ee_pose, atol=1e-6)
        self.assertEqual(ee_pose_source.calls, [0, 100_000_000])

    def test_retarget_ee_pose_frame_overrides_light_controller_delta_mapping(self):
        retarget_ee_pose = np.linspace(0.4, -0.4, 18, dtype=np.float32)
        cfg = _cfg(vx_scale=1.0, vy_scale=0.5, retarget_ee_pose=True)
        active_frame = _frame(
            timestamp_ns=100_000_000,
            left_axis=(0.5, 1.0),
            left_pos=(0.0, 0.1, -0.2),
            right_pos=(0.0, 0.2, -0.1),
        )
        active_frame["_retarget_ee_pose"] = retarget_ee_pose
        reader = _FakeReader(
            [
                _frame(timestamp_ns=0, right_a=True),
                active_frame,
            ]
        )
        ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg, sdk_reader=reader)

        ctrl.get_data()  # enter active and latch anchors
        data = ctrl.get_data()

        np.testing.assert_allclose(data["base_lin_vel_b"], [1.0, -0.25, 0.0], atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"], retarget_ee_pose, atol=1e-6)

    def test_retarget_ee_pose_mode_uses_single_retarget_reader_instead_of_pico_sdk_reader(self):
        cfg = _cfg(retarget_ee_pose=True)
        with (
            patch("robojudo.controller.pico_light_sparse_ctrl._PicoSdkReader") as sdk_reader_cls,
            patch("robojudo.controller.pico_light_sparse_ctrl._RetargetPicoReader", _FakeRetargetReader),
        ):
            sdk_reader_cls.side_effect = AssertionError("_PicoSdkReader must not be initialized")
            ctrl = PicoLightSparseCtrl(cfg_ctrl=cfg)

        self.assertIsInstance(ctrl._sdk_reader, _FakeRetargetReader)

    def test_retarget_pico_reader_uses_xrobot_controller_fields_and_timestamp(self):
        cfg = _cfg(retarget_ee_pose=True)
        reader = _RetargetPicoReader(
            cfg,
            streamer=_FakeRetargetStreamer(),
            retarget=_FakeLightRetarget(),
            snapshot_builder=_FakeLightSnapshotBuilder(),
        )

        frame = reader.read()

        self.assertEqual(frame["timestamp_ns"], 123_456_789)
        self.assertEqual(frame["left_axis"], (0.3, -0.4))
        self.assertEqual(frame["right_axis"], (-0.5, 0.6))
        self.assertEqual(frame["left_trigger"], 0.7)
        self.assertEqual(frame["left_grip"], 0.2)
        self.assertEqual(frame["right_trigger"], 0.1)
        self.assertEqual(frame["right_grip"], 0.9)
        self.assertTrue(frame["left_x"])
        self.assertFalse(frame["right_a"])
        self.assertFalse(frame["left_axis_click"])
        self.assertTrue(frame["right_axis_click"])
        timings = frame["_profile_timings"]
        self.assertGreaterEqual(timings["pico_read"], 0.0)
        self.assertGreaterEqual(timings["retarget"], 0.0)
        self.assertGreaterEqual(timings["snapshot"], 0.0)
        self.assertGreaterEqual(timings["sparse_extract"], 0.0)

    def test_idle_and_active_no_move_use_training_default_wrist_pose(self):
        cfg = _cfg()
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
        cfg = _cfg()
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

    def test_async_get_data_returns_cached_output_while_reader_waits(self):
        reader = _QueueReader()
        ctrl = PicoLightSparseCtrl(cfg_ctrl=PicoLightSparseCtrlCfg(), sdk_reader=reader)

        start = time.perf_counter()
        idle = ctrl.get_data()
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.05)
        self.assertEqual(idle["state"], "idle")

    def test_async_shutdown_command_is_drained_once(self):
        reader = _QueueReader()
        ctrl = PicoLightSparseCtrl(cfg_ctrl=PicoLightSparseCtrlCfg(), sdk_reader=reader)

        reader.frames.put(_frame(timestamp_ns=1, left_x=True))
        deadline = time.time() + 1.0
        commands = []
        while time.time() < deadline:
            data = ctrl.get_data()
            _processed, commands = ctrl.process_triggers(data)
            if commands:
                break
            time.sleep(0.01)

        self.assertEqual(commands, ["[SHUTDOWN]"])
        data = ctrl.get_data()
        _processed, commands = ctrl.process_triggers(data)
        self.assertEqual(commands, [])
