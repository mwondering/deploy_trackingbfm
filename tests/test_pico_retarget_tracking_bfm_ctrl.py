from __future__ import annotations

import unittest

import numpy as np

from robojudo.controller.ctrl_cfgs import PicoRetargetTrackingBfmCtrlCfg
from robojudo.controller.pico_retarget_tracking_bfm_ctrl import PicoRetargetTrackingBfmCtrl
from robojudo.tools.tracking_bfm_sparse_command import (
    DEFAULT_SPARSE_ANCHOR_HEIGHT_W,
    DEFAULT_SPARSE_EE_POSE,
    RetargetMotionSnapshot,
)


class _FakeStreamer:
    def __init__(self, frames: list[tuple]):
        self._frames = list(frames)
        self._last = frames[-1]

    def get_current_frame(self):
        if self._frames:
            self._last = self._frames.pop(0)
        return self._last


class _FakeRetarget:
    def __init__(self):
        self.calls = []

    def retarget(self, smplx_data, offset_to_ground=True):
        self.calls.append((smplx_data, offset_to_ground))
        return np.asarray(smplx_data["qpos"], dtype=np.float32)


class _FakeSnapshotBuilder:
    def __init__(self):
        self.calls = []

    def build(self, qpos, timestamp_ns):
        self.calls.append((qpos.copy(), timestamp_ns))
        return RetargetMotionSnapshot(
            body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
            body_pos_w=np.array(
                [
                    [0.0, 0.0, 0.8 + qpos[0]],
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
            body_lin_vel_w=np.array(
                [
                    [0.1 + qpos[0], 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            body_ang_vel_w=np.array(
                [
                    [0.0, 0.0, 0.2 + qpos[0]],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            timestamp_ns=timestamp_ns,
            qpos=qpos.copy(),
        )


def _frame(*, qpos: list[float], right_a=False, left_x=False, timestamp_ns=1) -> tuple:
    smplx_data = {"qpos": np.asarray(qpos, dtype=np.float32)}
    controller_data = {
        "RightController": {"key_one": right_a},
        "LeftController": {"key_one": left_x},
        "timestamp": timestamp_ns,
    }
    return smplx_data, None, None, controller_data, None


class TestPicoRetargetTrackingBfmCtrl(unittest.TestCase):
    def test_active_state_retargets_and_outputs_sparse_tracking_bfm_command(self):
        streamer = _FakeStreamer(
            [
                _frame(qpos=[0.0], right_a=True, timestamp_ns=1),
                _frame(qpos=[0.2], right_a=False, timestamp_ns=2),
            ]
        )
        retarget = _FakeRetarget()
        builder = _FakeSnapshotBuilder()
        ctrl = PicoRetargetTrackingBfmCtrl(
            cfg_ctrl=PicoRetargetTrackingBfmCtrlCfg(),
            streamer=streamer,
            retarget=retarget,
            snapshot_builder=builder,
        )

        first = ctrl.get_data()
        second = ctrl.get_data()

        self.assertEqual(first["state"], "active")
        self.assertEqual(second["state"], "active")
        self.assertEqual(len(retarget.calls), 2)
        self.assertTrue(retarget.calls[0][1])
        self.assertEqual(second["ee_pose"].shape, (18,))
        np.testing.assert_allclose(second["base_lin_vel_b"], [0.3, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(second["base_ang_vel_b"], [0.0, 0.0, 0.4], atol=1e-6)
        np.testing.assert_allclose(second["anchor_height_w"], [1.0], atol=1e-6)
        self.assertIsInstance(builder.calls[1][1], int)

    def test_idle_output_uses_sparse_training_default_pose(self):
        streamer = _FakeStreamer([_frame(qpos=[0.0], right_a=False)])
        ctrl = PicoRetargetTrackingBfmCtrl(
            cfg_ctrl=PicoRetargetTrackingBfmCtrlCfg(),
            streamer=streamer,
            retarget=_FakeRetarget(),
            snapshot_builder=_FakeSnapshotBuilder(),
        )

        data = ctrl.get_data()

        self.assertEqual(data["state"], "idle")
        np.testing.assert_allclose(data["ee_pose"], DEFAULT_SPARSE_EE_POSE, atol=1e-6)
        np.testing.assert_allclose(data["anchor_height_w"], [DEFAULT_SPARSE_ANCHOR_HEIGHT_W], atol=1e-6)

    def test_pause_freezes_last_sparse_command(self):
        streamer = _FakeStreamer(
            [
                _frame(qpos=[0.0], right_a=True),
                _frame(qpos=[0.2], right_a=False),
                _frame(qpos=[0.5], right_a=True),
                _frame(qpos=[0.7], right_a=False),
            ]
        )
        ctrl = PicoRetargetTrackingBfmCtrl(
            cfg_ctrl=PicoRetargetTrackingBfmCtrlCfg(),
            streamer=streamer,
            retarget=_FakeRetarget(),
            snapshot_builder=_FakeSnapshotBuilder(),
        )

        active = ctrl.get_data()
        still_active = ctrl.get_data()
        paused = ctrl.get_data()
        frozen = ctrl.get_data()

        self.assertEqual(still_active["state"], "active")
        self.assertEqual(paused["state"], "pause")
        np.testing.assert_allclose(frozen["base_lin_vel_b"], still_active["base_lin_vel_b"], atol=1e-6)

    def test_left_key_requests_shutdown(self):
        streamer = _FakeStreamer([_frame(qpos=[0.0], left_x=True)])
        ctrl = PicoRetargetTrackingBfmCtrl(
            cfg_ctrl=PicoRetargetTrackingBfmCtrlCfg(),
            streamer=streamer,
            retarget=_FakeRetarget(),
            snapshot_builder=_FakeSnapshotBuilder(),
        )

        data = ctrl.get_data()
        processed, commands = ctrl.process_triggers(data)

        self.assertEqual(processed["state"], "exit")
        self.assertEqual(commands, ["[SHUTDOWN]"])


if __name__ == "__main__":
    unittest.main()
