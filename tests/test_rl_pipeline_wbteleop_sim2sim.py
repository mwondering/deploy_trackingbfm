from __future__ import annotations

import sys
import types

import numpy as np
from box import Box

from robojudo.config.g1.g1_custom_cfg import (
    G1TrackingBfmSparseDoF,
    g1_tracking_bfm_pico_light_sim,
    g1_wbteleop_real,
    g1_wbteleop_sim2sim,
)
from robojudo.pipeline.rl_pipeline import RlPipeline


class _FakeEnv:
    num_dofs = 29

    def __init__(self):
        self.stiffness = np.zeros(29, dtype=np.float32)
        self.damping = np.zeros(29, dtype=np.float32)
        self.torque_limits = np.zeros(29, dtype=np.float32)
        self.dof_pos = np.arange(29, dtype=np.float32) * 0.01

    def set_gains(self, stiffness, damping):
        self.stiffness = np.asarray(stiffness, dtype=np.float32)
        self.damping = np.asarray(damping, dtype=np.float32)


class _FakeMujocoEnv(_FakeEnv):
    def __init__(self):
        super().__init__()
        self.model = object()
        self.data = types.SimpleNamespace(
            qpos=np.zeros(7 + self.num_dofs, dtype=np.float64),
            qvel=np.ones(6 + self.num_dofs, dtype=np.float64),
            ctrl=np.ones(self.num_dofs, dtype=np.float64),
            forwarded=False,
        )
        self.data.qpos[2] = 0.793
        self.updated = False

    def update(self):
        self.updated = True
        self.dof_pos = self.data.qpos[-self.num_dofs :].astype(np.float32)

    def get_data(self):
        return Box({"dof_pos": self.dof_pos.copy()})

    def step(self, pd_target, hand_pose=None):
        del hand_pose
        self.dof_pos = np.asarray(pd_target, dtype=np.float32)


def _make_pipeline_shell(cfg):
    pipeline = RlPipeline.__new__(RlPipeline)
    pipeline.cfg = cfg
    pipeline.env = _FakeEnv()
    pipeline._policy_stiffness = np.full(29, 11.0, dtype=np.float32)
    pipeline._policy_damping = np.full(29, 0.7, dtype=np.float32)
    pipeline._policy_torque_limits = np.full(29, 22.0, dtype=np.float32)
    pipeline.hold_policy = None
    pipeline._default_pose_mode_enabled = False
    pipeline._hold_to_policy_blend_start = None
    pipeline._hold_to_policy_blend_step = 0
    pipeline._hold_to_policy_blend_steps = 0
    pipeline._last_pd_target = None
    return pipeline


def test_wbteleop_sim2sim_default_qpos_uses_training_root_height(monkeypatch) -> None:
    cfg = g1_wbteleop_sim2sim()
    pipeline = _make_pipeline_shell(cfg)
    pipeline.env = _FakeMujocoEnv()
    pipeline.policy = types.SimpleNamespace(
        default_pos=np.asarray(G1TrackingBfmSparseDoF().default_pos, dtype=np.float32)
    )

    fake_mujoco = types.SimpleNamespace(
        mj_forward=lambda model, data: setattr(data, "forwarded", True)
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    pipeline._set_wbteleop_sim2sim_default_qpos()

    assert pipeline.env.data.qpos[2] == cfg.wbteleop_default_base_height
    np.testing.assert_allclose(
        pipeline.env.data.qpos[-pipeline.env.num_dofs :],
        np.asarray(G1TrackingBfmSparseDoF().default_pos, dtype=np.float64),
    )
    np.testing.assert_allclose(pipeline.env.data.qvel, 0.0)
    np.testing.assert_allclose(pipeline.env.data.ctrl, 0.0)
    assert pipeline.env.data.forwarded
    assert pipeline.env.updated


def test_wbteleop_sim2sim_hold_gains_switch_from_deploy_to_policy() -> None:
    cfg = g1_wbteleop_sim2sim()
    pipeline = _make_pipeline_shell(cfg)

    pipeline._set_default_pose_hold_gains(True)

    np.testing.assert_allclose(pipeline.env.stiffness, cfg.env.dof.stiffness)
    np.testing.assert_allclose(pipeline.env.damping, cfg.env.dof.damping)
    np.testing.assert_allclose(pipeline.env.torque_limits, cfg.env.dof.torque_limits)

    pipeline._set_default_pose_hold_gains(False)

    np.testing.assert_allclose(pipeline.env.stiffness, np.full(29, 11.0, dtype=np.float32))
    np.testing.assert_allclose(pipeline.env.damping, np.full(29, 0.7, dtype=np.float32))
    np.testing.assert_allclose(pipeline.env.torque_limits, np.full(29, 22.0, dtype=np.float32))


def test_wbteleop_sim2sim_hold_policy_gains_take_priority_for_hold() -> None:
    cfg = g1_wbteleop_sim2sim()
    pipeline = _make_pipeline_shell(cfg)
    pipeline.hold_policy = object()
    pipeline._hold_stiffness = np.full(29, 33.0, dtype=np.float32)
    pipeline._hold_damping = np.full(29, 3.3, dtype=np.float32)
    pipeline._hold_torque_limits = np.full(29, 44.0, dtype=np.float32)

    pipeline._set_default_pose_hold_gains(True)

    np.testing.assert_allclose(pipeline.env.stiffness, np.full(29, 33.0, dtype=np.float32))
    np.testing.assert_allclose(pipeline.env.damping, np.full(29, 3.3, dtype=np.float32))
    np.testing.assert_allclose(pipeline.env.torque_limits, np.full(29, 44.0, dtype=np.float32))


def test_wbteleop_sim2sim_missing_hold_torque_limits_falls_back_to_policy_limits() -> None:
    pipeline = _make_pipeline_shell(g1_wbteleop_sim2sim())

    class _Dof:
        torque_limits = None

    class _Adapter:
        def fit(self, value, template):
            del template
            return np.asarray(value, dtype=np.float32)

    class _HoldPolicy:
        cfg_action_dof = _Dof()
        actions_adapter = _Adapter()

    pipeline.hold_policy = _HoldPolicy()

    fitted = pipeline._fit_hold_dof_property("torque_limits", pipeline._policy_torque_limits)

    np.testing.assert_allclose(fitted, pipeline._policy_torque_limits)
    assert fitted is not pipeline._policy_torque_limits


def test_non_wbteleop_sim2sim_config_does_not_switch_hold_gains() -> None:
    pipeline = _make_pipeline_shell(g1_tracking_bfm_pico_light_sim())
    before = Box(
        {
            "stiffness": pipeline.env.stiffness.copy(),
            "damping": pipeline.env.damping.copy(),
            "torque_limits": pipeline.env.torque_limits.copy(),
        }
    )

    pipeline._set_default_pose_hold_gains(True)

    np.testing.assert_allclose(pipeline.env.stiffness, before.stiffness)
    np.testing.assert_allclose(pipeline.env.damping, before.damping)
    np.testing.assert_allclose(pipeline.env.torque_limits, before.torque_limits)


def test_wbteleop_sim2sim_uses_hold_policy_only_in_default_pose_mode() -> None:
    pipeline = _make_pipeline_shell(g1_wbteleop_sim2sim())
    main_policy = object()
    hold_policy = object()
    pipeline.policy = main_policy
    pipeline.hold_policy = hold_policy

    pipeline._default_pose_mode_enabled = True
    assert pipeline._policy_for_step() is hold_policy

    pipeline._default_pose_mode_enabled = False
    assert pipeline._policy_for_step() is main_policy


def test_wbteleop_sim2sim_hold_to_policy_blend_starts_from_last_target() -> None:
    pipeline = _make_pipeline_shell(g1_wbteleop_sim2sim())
    pipeline.hold_policy = object()
    pipeline._default_pose_mode_enabled = True
    pipeline.freq = 50
    pipeline._last_pd_target = np.full(29, 0.5, dtype=np.float32)

    pipeline._start_hold_to_policy_blend()

    assert pipeline._hold_to_policy_blend_steps == 37
    np.testing.assert_allclose(pipeline._hold_to_policy_blend_start, np.full(29, 0.5, dtype=np.float32))
    blended = pipeline._apply_hold_to_policy_blend(np.full(29, 1.0, dtype=np.float32))
    assert np.all(blended > 0.5)
    assert np.all(blended < 1.0)


def test_wbteleop_real_config_does_not_switch_sim2sim_hold_gains() -> None:
    pipeline = _make_pipeline_shell(g1_wbteleop_real())
    before = Box(
        {
            "stiffness": pipeline.env.stiffness.copy(),
            "damping": pipeline.env.damping.copy(),
            "torque_limits": pipeline.env.torque_limits.copy(),
        }
    )

    pipeline._set_default_pose_hold_gains(True)

    np.testing.assert_allclose(pipeline.env.stiffness, before.stiffness)
    np.testing.assert_allclose(pipeline.env.damping, before.damping)
    np.testing.assert_allclose(pipeline.env.torque_limits, before.torque_limits)


def test_default_pose_prepare_ctrl_data_is_neutral_for_wbteleop_policy() -> None:
    pipeline = _make_pipeline_shell(g1_wbteleop_real())
    pipeline.policy = types.SimpleNamespace(ctrl_type="PicoRetargetTrackingBfmCtrl")

    ctrl_data = pipeline._default_pose_prepare_ctrl_data()

    assert ctrl_data.COMMANDS == []
    assert ctrl_data.PicoRetargetTrackingBfmCtrl.state == "idle"


def test_wbteleop_blend_in_does_not_consume_pico_controller_state() -> None:
    pipeline = _make_pipeline_shell(g1_wbteleop_real())
    pipeline.env = _FakeMujocoEnv()
    pipeline.freq = 1
    pipeline.dt = 0.0
    pipeline._prepare_seconds = 1.0
    pipeline._set_default_pose_mode = lambda enabled: None

    class _Policy:
        ctrl_type = "PicoRetargetTrackingBfmCtrl"

        def set_default_pose_mode(self, enabled):
            del enabled

        def get_observation(self, env_data, ctrl_data):
            del env_data
            assert ctrl_data.PicoRetargetTrackingBfmCtrl.state == "idle"
            assert ctrl_data.COMMANDS == []
            return np.zeros(1, dtype=np.float32), {}

        def get_pd_target(self, obs):
            del obs
            return np.ones(29, dtype=np.float32)

    class _CtrlManager:
        def get_ctrl_data(self, env_data):
            del env_data
            raise AssertionError("prepare blend-in must not read Pico controller state")

    pipeline.policy = _Policy()
    pipeline.ctrl_manager = _CtrlManager()
    pipeline._init_dof_pos = np.zeros(29, dtype=np.float32)

    pipeline._run_blend_in()

    np.testing.assert_allclose(pipeline.env.dof_pos, np.zeros(29, dtype=np.float32))


def test_prepare_ctrl_data_keeps_controller_path_for_non_default_pose_policy() -> None:
    pipeline = _make_pipeline_shell(g1_tracking_bfm_pico_light_sim())
    env_data = Box({})

    class _CtrlManager:
        def __init__(self):
            self.calls = 0

        def get_ctrl_data(self, received_env_data):
            self.calls += 1
            assert received_env_data is env_data
            return Box({"COMMANDS": ["[MOTION_RESET]"]})

    pipeline.policy = types.SimpleNamespace()
    pipeline.ctrl_manager = _CtrlManager()

    ctrl_data = pipeline._prepare_ctrl_data(env_data)

    assert ctrl_data.COMMANDS == ["[MOTION_RESET]"]
    assert pipeline.ctrl_manager.calls == 1
