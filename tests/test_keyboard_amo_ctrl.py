import unittest
from pathlib import Path
from unittest.mock import patch


class TestKeyboardAmoCtrl(unittest.TestCase):
    def test_keyboard_amo_state_matches_amo_command_steps(self):
        from robojudo.controller.keyboard_amo_ctrl import KeyboardAmoInputState

        state = KeyboardAmoInputState()

        data = state.transform_events(
            [
                {"type": "keyboard", "name": "w", "pressed": True, "timestamp": 1.0},
                {"type": "keyboard", "name": "a", "pressed": True, "timestamp": 1.1},
                {"type": "keyboard", "name": "q", "pressed": True, "timestamp": 1.2},
                {"type": "keyboard", "name": "z", "pressed": True, "timestamp": 1.3},
                {"type": "keyboard", "name": "j", "pressed": True, "timestamp": 1.4},
                {"type": "keyboard", "name": "k", "pressed": True, "timestamp": 1.5},
                {"type": "keyboard", "name": "l", "pressed": True, "timestamp": 1.6},
                {"type": "keyboard", "name": "t", "pressed": True, "timestamp": 1.7},
            ]
        )

        self.assertEqual(
            data["commands"],
            [0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.0],
        )
        self.assertEqual(data["axes"], {})
        self.assertEqual(data["button_event"], [])
        self.assertEqual(data["upper_body_mode_event"], "swing")

        data = state.transform_events(
            [
                {"type": "keyboard", "name": "s", "pressed": True, "timestamp": 1.8},
                {"type": "keyboard", "name": "d", "pressed": True, "timestamp": 1.9},
                {"type": "keyboard", "name": "e", "pressed": True, "timestamp": 2.0},
                {"type": "keyboard", "name": "x", "pressed": True, "timestamp": 2.1},
                {"type": "keyboard", "name": "u", "pressed": True, "timestamp": 2.2},
                {"type": "keyboard", "name": "i", "pressed": True, "timestamp": 2.3},
                {"type": "keyboard", "name": "o", "pressed": True, "timestamp": 2.4},
                {"type": "keyboard", "name": "g", "pressed": True, "timestamp": 2.5},
            ]
        )

        self.assertEqual(
            data["commands"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0],
        )
        self.assertEqual(data["axes"], {})
        self.assertEqual(data["button_event"], [])
        self.assertEqual(data["upper_body_mode_event"], "lock")

    def test_amo_policy_prefers_direct_keyboard_commands(self):
        import numpy as np

        from robojudo.policy.amo_policy import AMOPolicy

        policy = AMOPolicy.__new__(AMOPolicy)
        policy.cmd = np.zeros(8, dtype=np.float32)

        commands = policy._get_commands(
            {
                "KeyboardAmoCtrl": {
                    "commands": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0], dtype=np.float32)
                }
            }
        )

        np.testing.assert_allclose(commands, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])

    def test_amo_policy_height_is_offset_from_075_like_original_amo(self):
        import numpy as np
        import torch

        from robojudo.policy.amo_policy import AMOPolicy

        policy = AMOPolicy.__new__(AMOPolicy)
        policy.cmd = np.zeros(8, dtype=np.float32)
        policy.commands_map = [
            [-1.0, 0.0, 1.0],
            [0.2, 0.0, -0.2],
            [0.8, 0.0, -0.8],
            [0.3, 0.75, 0.9],
        ]

        commands = policy._get_commands({})
        np.testing.assert_allclose(commands, np.zeros(8, dtype=np.float32))

        class DummyAdapter:
            def __call__(self, x):
                return torch.zeros((1, 15), dtype=torch.float32)

        policy._get_commands = lambda _: np.zeros(8, dtype=np.float32)
        policy.default_dof_pos = np.zeros(23, dtype=np.float32)
        policy.action_scale = 0.25
        policy.last_action = np.zeros(15, dtype=np.float32)
        policy.scales_ang_vel = 0.25
        policy.scales_dof_vel = 0.05
        policy.device = "cpu"
        policy._n_demo_dof = 8
        policy.n_priv = 3
        policy.n_proprio = 76
        policy.control_dt = 0.02
        policy.gait_freq = 1.3
        policy._in_place_stand_flag = True
        policy.gait_cycle = np.array([0.25, 0.25], dtype=np.float32)
        policy.proprio_history_buf = []
        policy.extra_history_buf = []
        policy.demo_obs_template = np.zeros((17,), dtype=np.float32)
        policy.demo_obs_template[14:17] = 0.75
        policy.adapter_input = torch.zeros((1, 12), dtype=torch.float32)
        policy.adapter = DummyAdapter()
        policy.input_mean = torch.zeros((12,), dtype=torch.float32)
        policy.input_std = torch.ones((12,), dtype=torch.float32)
        policy.output_mean = torch.zeros((15,), dtype=torch.float32)
        policy.output_std = torch.ones((15,), dtype=torch.float32)
        policy.adapter_output = torch.zeros((1, 15), dtype=torch.float32)

        class EnvData:
            dof_pos = np.zeros(23, dtype=np.float32)
            dof_vel = np.zeros(23, dtype=np.float32)
            base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            base_ang_vel = np.zeros(3, dtype=np.float32)

        policy.get_observation(EnvData(), {})
        self.assertAlmostEqual(float(policy.adapter_input[0, 0].item()), 0.75, places=5)

    def test_keyboard_amo_state_prints_commands_on_change(self):
        from robojudo.controller.keyboard_amo_ctrl import KeyboardAmoInputState

        state = KeyboardAmoInputState()

        with patch("builtins.print") as mock_print:
            state.transform_events(
                [
                    {"type": "keyboard", "name": "w", "pressed": True, "timestamp": 1.0},
                ]
            )
            mock_print.assert_called_once()
            printed = mock_print.call_args.args[0]
            self.assertIn("vx:", printed)
            self.assertIn("0.05", printed)

        with patch("builtins.print") as mock_print:
            state.transform_events(
                [
                    {"type": "keyboard", "name": "w", "pressed": False, "timestamp": 1.1},
                ]
            )
            mock_print.assert_not_called()

    def test_amo_policy_upper_body_mode_switches_between_swing_and_lock(self):
        import numpy as np

        from robojudo.policy.amo_policy import AMOPolicy

        policy = AMOPolicy.__new__(AMOPolicy)
        policy.timestep = 0
        policy.upper_body_default_pos = np.array([0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3], dtype=np.float32)
        policy._n_demo_dof = 8
        policy._upper_body_env_indices = np.array([15, 16, 17, 18, 22, 23, 24, 25], dtype=np.int64)
        policy._upper_body_keyframes = np.stack(
            [
                policy.upper_body_default_pos,
                policy.upper_body_default_pos + np.array([0.16, 0.10, -0.06, 0.24, 0.04, -0.02, 0.04, 0.08]),
                policy.upper_body_default_pos + np.array([0.08, -0.05, 0.10, 0.34, -0.02, 0.03, -0.03, 0.10]),
                policy.upper_body_default_pos + np.array([0.03, 0.14, -0.14, 0.18, 0.07, -0.04, 0.07, 0.05]),
            ],
            axis=0,
        ).astype(np.float32)
        policy._upper_body_target = policy.upper_body_default_pos.copy()
        policy._upper_body_mode = "locked"
        policy._upper_body_next_mode = None
        policy._upper_body_transition_start = policy.upper_body_default_pos.copy()
        policy._upper_body_transition_end = policy.upper_body_default_pos.copy()
        policy._upper_body_transition_step = 0
        policy._upper_body_transition_steps = 3
        policy._upper_body_swing_step = 0
        policy._upper_body_swing_period_steps = 20

        policy._request_upper_body_mode("swing")
        self.assertEqual(policy._upper_body_mode, "swing")

        pd_target = np.zeros(29, dtype=np.float32)
        first_target = policy.override_pd_target(pd_target)
        np.testing.assert_allclose(
            first_target[policy._upper_body_env_indices],
            policy.upper_body_default_pos,
        )
        np.testing.assert_allclose(first_target[12:15], np.zeros(3, dtype=np.float32))

        policy.post_step_callback()
        swung_target = policy.override_pd_target(np.zeros(29, dtype=np.float32))
        self.assertFalse(
            np.allclose(swung_target[policy._upper_body_env_indices], policy.upper_body_default_pos)
        )

        policy._request_upper_body_mode("lock")
        self.assertEqual(policy._upper_body_mode, "transition")
        for _ in range(policy._upper_body_transition_steps):
            policy.post_step_callback()
        locked_target = policy.override_pd_target(np.zeros(29, dtype=np.float32))
        self.assertEqual(policy._upper_body_mode, "locked")
        np.testing.assert_allclose(
            locked_target[policy._upper_body_env_indices],
            policy.upper_body_default_pos,
            atol=1e-6,
        )

    def test_amo_policy_uses_original_upper_body_default_pose(self):
        import numpy as np

        from robojudo.policy.amo_policy import AMOPolicy

        np.testing.assert_allclose(
            AMOPolicy.AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS,
            [0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3],
        )

    def test_amo_policy_does_not_overwrite_29dof_torso_rpy_when_locking_arms(self):
        import numpy as np

        from robojudo.policy.amo_policy import AMOPolicy

        policy = AMOPolicy.__new__(AMOPolicy)
        policy.upper_body_default_pos = AMOPolicy.AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS.copy()
        policy._upper_body_target = policy.upper_body_default_pos.copy()
        policy._upper_body_mode = "locked"
        policy._upper_body_env_indices = AMOPolicy.G1_29DOF_UPPER_BODY_ENV_INDICES

        pd_target = np.arange(29, dtype=np.float32)
        locked_target = policy.override_pd_target(pd_target)

        np.testing.assert_allclose(locked_target[12:15], pd_target[12:15])
        np.testing.assert_allclose(
            locked_target[policy._upper_body_env_indices],
            AMOPolicy.AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS,
        )

    def test_policy_wrapper_applies_amo_upper_body_default_to_init_pose(self):
        import numpy as np

        from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
        from robojudo.config.g1.policy.g1_amo_policy_cfg import G1AmoLowerDoF
        from robojudo.pipeline.rl_pipeline import PolicyWrapper
        from robojudo.policy.amo_policy import AMOPolicy
        from robojudo.tools.dof import DoFAdapter

        wrapper = PolicyWrapper.__new__(PolicyWrapper)
        wrapper.env_dof_cfg = G1_29DoF()
        wrapper.actions_adapter = DoFAdapter(G1AmoLowerDoF().joint_names, wrapper.env_dof_cfg.joint_names)

        class DummyPolicy:
            def get_init_dof_pos(self):
                return np.asarray(G1AmoLowerDoF().default_pos, dtype=np.float32)

            def override_pd_target(self, pd_target):
                pd_target = pd_target.copy()
                pd_target[AMOPolicy.G1_29DOF_UPPER_BODY_ENV_INDICES] = AMOPolicy.AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS
                return pd_target

        wrapper.policy = DummyPolicy()

        init_pose = wrapper.get_init_dof_pos()
        np.testing.assert_allclose(init_pose[12:15], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(
            init_pose[AMOPolicy.G1_29DOF_UPPER_BODY_ENV_INDICES],
            AMOPolicy.AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS,
        )

    def test_amo_policy_yaw_command_disables_in_place_stand(self):
        import numpy as np
        import torch

        from robojudo.policy.amo_policy import AMOPolicy

        policy = AMOPolicy.__new__(AMOPolicy)
        policy.cmd = np.zeros(8, dtype=np.float32)
        policy.default_dof_pos = np.zeros(23, dtype=np.float32)
        policy.last_action = np.zeros(15, dtype=np.float32)
        policy.action_scale = 0.25
        policy.scales_ang_vel = 0.25
        policy.scales_dof_vel = 0.05
        policy.device = "cpu"
        policy._n_demo_dof = 8
        policy.n_priv = 3
        policy.n_proprio = 76
        policy.control_dt = 0.02
        policy.gait_freq = 1.3
        policy.gait_cycle = np.array([0.25, 0.25], dtype=np.float32)
        policy.proprio_history_buf = []
        policy.extra_history_buf = []
        policy.demo_obs_template = np.zeros((17,), dtype=np.float32)
        policy.adapter_input = torch.zeros((1, 12), dtype=torch.float32)
        policy.input_mean = torch.zeros((12,), dtype=torch.float32)
        policy.input_std = torch.ones((12,), dtype=torch.float32)
        policy.output_mean = torch.zeros((15,), dtype=torch.float32)
        policy.output_std = torch.ones((15,), dtype=torch.float32)
        policy.adapter_output = torch.zeros((1, 15), dtype=torch.float32)

        class DummyAdapter:
            def __call__(self, x):
                return torch.zeros((1, 15), dtype=torch.float32)

        class EnvData:
            dof_pos = np.zeros(23, dtype=np.float32)
            dof_vel = np.zeros(23, dtype=np.float32)
            base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            base_ang_vel = np.zeros(3, dtype=np.float32)

        policy.adapter = DummyAdapter()
        policy._get_upper_body_mode_event = lambda _: None
        policy._get_commands = lambda _: np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        policy.get_observation(EnvData(), {})
        self.assertFalse(policy._in_place_stand_flag)

    def test_g1_amo_config_is_registered(self):
        from robojudo.config import cfg_registry

        cfg_class = cfg_registry.get("g1_amo")
        cfg = cfg_class()

        self.assertEqual(cfg.policy.policy_type, "AMOPolicy")
        self.assertEqual(cfg.ctrl[0].ctrl_type, "KeyboardAmoCtrl")

    def test_run_pipeline_has_no_debug_breakpoint(self):
        source = Path("scripts/run_pipeline.py").read_text()

        self.assertNotIn("pdb.set_trace()", source)
