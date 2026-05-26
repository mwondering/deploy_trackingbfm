import unittest

import numpy as np


class TestUnitreeAmoCtrl(unittest.TestCase):
    def test_axes_map_to_amo_commands(self):
        from robojudo.controller.ctrl_cfgs import UnitreeAmoCtrlCfg
        from robojudo.controller.unitree_amo_ctrl import UnitreeAmoInputState

        cfg = UnitreeAmoCtrlCfg(enable_height_axis=True)
        state = UnitreeAmoInputState(cfg)

        data = state.transform(
            {
                "LeftY": 0.5,
                "LeftX": -0.5,
                "RightX": 1.0,
                "RightY": -1.0,
            },
            [],
        )

        np.testing.assert_allclose(data["commands"], [0.5, -0.8, 0.2, -0.3, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(data["yaw_command_mode"], "velocity")

        data = state.transform(
            {
                "LeftY": 0.0,
                "LeftX": 0.0,
                "RightX": 1.0,
                "RightY": 0.0,
            },
            [],
        )

        np.testing.assert_allclose(data["commands"], [0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_deadzone_and_height_axis_default(self):
        from robojudo.controller.ctrl_cfgs import UnitreeAmoCtrlCfg
        from robojudo.controller.unitree_amo_ctrl import UnitreeAmoInputState

        state = UnitreeAmoInputState(UnitreeAmoCtrlCfg())

        data = state.transform(
            {
                "LeftY": 0.04,
                "LeftX": -0.04,
                "RightX": 0.04,
                "RightY": 1.0,
            },
            [],
        )

        self.assertEqual(data["commands"], [0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0])

        data = state.transform(
            {
                "RightY": -1.0,
            },
            [],
        )

        self.assertEqual(data["commands"], [0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0])

    def test_buttons_adjust_torso_commands_and_clip(self):
        from robojudo.controller.ctrl_cfgs import UnitreeAmoCtrlCfg
        from robojudo.controller.unitree_amo_ctrl import UnitreeAmoInputState

        state = UnitreeAmoInputState(UnitreeAmoCtrlCfg())
        events = [
            {"type": "button", "name": "Right", "pressed": True, "timestamp": 1.0},
            {"type": "button", "name": "Right", "pressed": True, "timestamp": 1.1},
            {"type": "button", "name": "Up", "pressed": True, "timestamp": 1.2},
            {"type": "button", "name": "R1", "pressed": True, "timestamp": 1.3},
        ]

        data = state.transform({}, events)

        np.testing.assert_allclose(data["commands"][4:7], [0.1, 0.05, 0.05])

        clip_events = [
            {"type": "button", "name": "Right", "pressed": True, "timestamp": 2.0}
            for _ in range(20)
        ]
        data = state.transform({}, clip_events)

        self.assertAlmostEqual(data["commands"][4], 0.3)

    def test_upper_body_mode_events_and_y_toggles_stand_still(self):
        from robojudo.controller.ctrl_cfgs import UnitreeAmoCtrlCfg
        from robojudo.controller.unitree_amo_ctrl import UnitreeAmoInputState

        state = UnitreeAmoInputState(UnitreeAmoCtrlCfg())

        swing = state.transform(
            {},
            [
                {"type": "button", "name": "X", "pressed": True, "timestamp": 1.0},
                {"type": "button", "name": "Y", "pressed": True, "timestamp": 1.1},
            ],
        )
        self.assertEqual(swing["upper_body_mode_event"], "swing")
        self.assertEqual(swing["commands"][7], 0.0)
        self.assertEqual(swing["stand_still_event"], "toggle")

        lock = state.transform(
            {},
            [{"type": "button", "name": "B", "pressed": True, "timestamp": 2.0}],
        )
        self.assertEqual(lock["upper_body_mode_event"], "lock")

    def test_a_button_still_triggers_shutdown(self):
        from robojudo.controller.ctrl_cfgs import UnitreeAmoCtrlCfg
        from robojudo.controller.unitree_amo_ctrl import UnitreeAmoCtrl

        ctrl = UnitreeAmoCtrl.__new__(UnitreeAmoCtrl)
        ctrl.triggers = UnitreeAmoCtrlCfg().triggers.copy()
        ctrl.combination_init_buttons = []
        ctrl.onhold_buttons = set()

        ctrl_data = {
            "button_event": [
                {"type": "button", "name": "A", "pressed": True, "timestamp": 1.0},
            ],
        }

        _, commands = UnitreeAmoCtrl.process_triggers(ctrl, ctrl_data)

        self.assertEqual(commands, ["[SHUTDOWN]"])


if __name__ == "__main__":
    unittest.main()
