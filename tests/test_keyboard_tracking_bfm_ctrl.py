from __future__ import annotations

import unittest

import numpy as np

from robojudo.controller.ctrl_cfgs import KeyboardTrackingBfmCtrlCfg
from robojudo.controller.keyboard_tracking_bfm_ctrl import KeyboardTrackingBfmCtrl


class _DummyKeyboardThread:
    def __init__(self, event_queue):
        self.event_queue = event_queue

    def start(self):
        return


def _event(name: str, pressed: bool) -> dict:
    return {
        "type": "keyboard",
        "name": name,
        "pressed": pressed,
        "timestamp": 0.0,
    }


class TestKeyboardTrackingBfmCtrl(unittest.TestCase):
    def test_toggle_active_pause_and_shutdown(self):
        ctrl = KeyboardTrackingBfmCtrl(
            cfg_ctrl=KeyboardTrackingBfmCtrlCfg(),
            keyboard_thread_cls=_DummyKeyboardThread,
        )

        ctrl.event_queue.put(_event("Key.space", False))
        active = ctrl.get_data()
        self.assertEqual(active["state"], "active")

        ctrl.event_queue.put(_event("Key.space", False))
        paused = ctrl.get_data()
        self.assertEqual(paused["state"], "pause")

        ctrl.event_queue.put(_event("Key.esc", False))
        exited = ctrl.get_data()
        exited, commands = ctrl.process_triggers(exited)
        self.assertEqual(exited["state"], "exit")
        self.assertEqual(commands, ["[SHUTDOWN]"])

    def test_keyboard_mapping_generates_sparse_commands(self):
        cfg = KeyboardTrackingBfmCtrlCfg(vx_scale=1.0, vy_scale=0.5, wz_scale=1.0, ee_pos_step=0.02)
        ctrl = KeyboardTrackingBfmCtrl(
            cfg_ctrl=cfg,
            keyboard_thread_cls=_DummyKeyboardThread,
        )

        ctrl.event_queue.put(_event("Key.space", False))
        ctrl.get_data()

        for key in ("w", "a", "q", "r", "t", "y", "u", "i", "o", "p"):
            ctrl.event_queue.put(_event(key, True))

        data = ctrl.get_data()
        np.testing.assert_allclose(data["base_lin_vel_b"], [1.0, 0.5, 0.0], atol=1e-6)
        np.testing.assert_allclose(data["base_ang_vel_b"], [0.0, 0.0, 1.0], atol=1e-6)
        self.assertGreater(data["anchor_height_w"][0], cfg.base_height_init)
        self.assertEqual(data["ee_pose"].shape, (18,))
        np.testing.assert_allclose(data["ee_pose"][:3], np.array(cfg.ee_neutral_left) + 0.02, atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"][9:12], np.array(cfg.ee_neutral_right) + 0.02, atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"][3:9], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(data["ee_pose"][12:18], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)

    def test_enter_reanchors_keyboard_state(self):
        cfg = KeyboardTrackingBfmCtrlCfg(ee_pos_step=0.02)
        ctrl = KeyboardTrackingBfmCtrl(
            cfg_ctrl=cfg,
            keyboard_thread_cls=_DummyKeyboardThread,
        )
        ctrl.event_queue.put(_event("Key.space", False))
        ctrl.get_data()
        ctrl.event_queue.put(_event("t", True))
        moved = ctrl.get_data()
        self.assertGreater(moved["ee_pose"][0], cfg.ee_neutral_left[0])

        ctrl.event_queue.put(_event("t", False))
        ctrl.event_queue.put(_event("Key.enter", False))
        reset = ctrl.get_data()
        np.testing.assert_allclose(reset["ee_pose"][:3], cfg.ee_neutral_left, atol=1e-6)
        self.assertEqual(reset["state"], "active")
