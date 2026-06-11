from robojudo.config import cfg_registry
from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg
from robojudo.config.g1.policy.g1_tracking_bfm_sparse_onnx_policy_cfg import (
    G1TrackingBfmSparseDoF,
    G1TrackingBfmSparseOnnxPolicyCfg,
    G1WbTeleopOnnxPolicyCfg,
)
from robojudo.config.g1.policy.g1_unitree_policy_cfg import G1UnitreeWoGaitPolicyCfg
from robojudo.controller.ctrl_cfgs import (
    KeyboardCtrlCfg,
    KeyboardTrackingBfmCtrlCfg,
    PicoLightSparseCtrlCfg,
    PicoRetargetTrackingBfmCtrlCfg,
    UnitreeCtrlCfg,
    WbTeleopNpzPlaybackCtrlCfg,
)


def test_g1_tracking_bfm_pico_light_sim_config_is_registered() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_pico_light_sim")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1MujocoEnvCfg)
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoLightSparseCtrlCfg)
    assert cfg.ctrl[0].retarget_ee_pose is True
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.policy_type == "TrackingBfmSparseOnnxPolicy"
    assert cfg.debug.log_obs is True


def test_g1_tracking_bfm_pico_retarget_sim_config_is_registered() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_pico_retarget_sim")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1MujocoEnvCfg)
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoRetargetTrackingBfmCtrlCfg)
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "PicoRetargetTrackingBfmCtrl"
    assert cfg.ctrl[0].anchor_body_name == "pelvis"
    assert cfg.ctrl[0].left_ee_body_name == "left_wrist_yaw_link"
    assert cfg.ctrl[0].right_ee_body_name == "right_wrist_yaw_link"
    assert cfg.debug.log_obs is True


def test_g1_tracking_bfm_pico_light_real_config_is_registered_for_dev_pc_dds() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_pico_light_real")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1RealEnvCfg)
    assert cfg.env.env_type == "UnitreeCppEnv"
    assert cfg.env.unitree.net_if == "eth0"
    assert cfg.env.born_place_align is False
    assert cfg.env.odometry_type == "UNITREE"
    assert cfg.env.limit_pd_target_effort is False
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoLightSparseCtrlCfg)
    assert cfg.ctrl[0].retarget_ee_pose is True
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "PicoLightSparseCtrl"
    assert cfg.do_safety_check is True
    assert cfg.debug.log_obs is True


def test_g1_tracking_bfm_pico_retarget_real_config_is_registered_for_dev_pc_dds() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_pico_retarget_real")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1RealEnvCfg)
    assert cfg.env.env_type == "UnitreeCppEnv"
    assert cfg.env.unitree.net_if == "eth0"
    assert cfg.env.born_place_align is False
    assert cfg.env.limit_pd_target_effort is False
    assert cfg.env.clip_pd_target is False
    assert cfg.env.pd_target_max_delta is None
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoRetargetTrackingBfmCtrlCfg)
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "PicoRetargetTrackingBfmCtrl"
    assert cfg.do_safety_check is True
    assert cfg.debug.log_obs is True


def test_g1_tracking_bfm_keyboard_sim_config_is_registered() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_keyboard_sim")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1MujocoEnvCfg)
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], KeyboardTrackingBfmCtrlCfg)
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "KeyboardTrackingBfmCtrl"
    assert cfg.policy.onnx_path.endswith(".onnx")
    assert len(cfg.policy.action_scales) == cfg.policy.action_dof.num_dofs
    assert cfg.debug.log_obs is True


def test_g1_wbteleop_sim2sim_config_is_registered_for_0608_checkpoint() -> None:
    cfg_class = cfg_registry.get("g1_wbteleop_sim2sim")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1MujocoEnvCfg)
    assert cfg.env.born_place_align is False
    assert cfg.env.random_heading is True
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoRetargetTrackingBfmCtrlCfg)
    assert isinstance(cfg.policy, G1WbTeleopOnnxPolicyCfg)
    assert cfg.policy.policy_type == "WbTeleopOnnxPolicy"
    assert cfg.policy.ctrl_type == "PicoRetargetTrackingBfmCtrl"
    assert cfg.policy.onnx_path == (
        "/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/"
        "0608_ckpt_bcrl/deploy_model_16000.onnx"
    )
    assert cfg.policy.env_yaml_path == (
        "/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/"
        "0608_ckpt_bcrl/params/env.yaml"
    )
    assert cfg.policy.obs_group == "actor"
    assert cfg.policy.expected_obs_dim == 886
    assert cfg.policy.action_dof.num_dofs == 29
    assert len(cfg.policy.action_scales) == cfg.policy.action_dof.num_dofs
    assert isinstance(cfg.hold_policy, G1UnitreeWoGaitPolicyCfg)
    assert cfg.hold_to_policy_blend_seconds == 0.75
    assert cfg.debug.wbteleop_proprio_debug is True
    assert cfg.debug.wbteleop_proprio_debug_interval == 50
    assert cfg.env.dof.default_pos == G1TrackingBfmSparseDoF().default_pos
    assert cfg.env.dof.stiffness == G1_29DoF().stiffness
    assert cfg.env.dof.damping == G1_29DoF().damping


def test_g1_wbteleop_real_config_is_registered_for_dev_pc_dds() -> None:
    cfg_class = cfg_registry.get("g1_wbteleop_real")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1RealEnvCfg)
    assert cfg.env.env_type == "UnitreeCppEnv"
    assert cfg.env.unitree.net_if == "eth0"
    assert cfg.env.born_place_align is False
    assert cfg.env.limit_pd_target_effort is False
    assert cfg.env.clip_pd_target is False
    assert cfg.env.pd_target_max_delta is None
    assert cfg.env.dof.default_pos == G1TrackingBfmSparseDoF().default_pos
    assert cfg.env.dof.stiffness == G1TrackingBfmSparseDoF().stiffness
    assert cfg.env.dof.damping == G1TrackingBfmSparseDoF().damping
    assert cfg.env.dof.torque_limits == G1TrackingBfmSparseDoF().torque_limits
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoRetargetTrackingBfmCtrlCfg)
    assert isinstance(cfg.policy, G1WbTeleopOnnxPolicyCfg)
    assert cfg.policy.policy_type == "WbTeleopOnnxPolicy"
    assert cfg.policy.ctrl_type == "PicoRetargetTrackingBfmCtrl"
    assert isinstance(cfg.hold_policy, G1UnitreeWoGaitPolicyCfg)
    assert cfg.hold_to_policy_blend_seconds == 0.75
    assert cfg.policy.onnx_path == (
        "/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/"
        "0608_ckpt_bcrl/deploy_model_16000.onnx"
    )
    assert cfg.policy.env_yaml_path == (
        "/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/"
        "0608_ckpt_bcrl/params/env.yaml"
    )
    assert cfg.policy.obs_group == "actor"
    assert cfg.policy.expected_obs_dim == 886
    assert cfg.debug.wbteleop_proprio_debug is True
    assert cfg.debug.wbteleop_proprio_debug_interval == 50
    assert cfg.do_safety_check is True
    assert cfg.debug.log_obs is False


def test_g1_wbteleop_npz_play_sim2sim_config_is_registered() -> None:
    cfg_class = cfg_registry.get("g1_wbteleop_npz_play_sim2sim")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1MujocoEnvCfg)
    assert cfg.env.born_place_align is False
    assert cfg.env.random_heading is False
    assert len(cfg.ctrl) == 2
    assert isinstance(cfg.ctrl[0], KeyboardCtrlCfg)
    assert isinstance(cfg.ctrl[1], WbTeleopNpzPlaybackCtrlCfg)
    assert cfg.ctrl[1].motion_type == "isaaclab"
    assert cfg.ctrl[1].auto_start is False
    assert isinstance(cfg.policy, G1WbTeleopOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "WbTeleopNpzPlaybackCtrl"
    assert isinstance(cfg.hold_policy, G1UnitreeWoGaitPolicyCfg)
    assert cfg.wbteleop_default_base_height == 0.76


def test_g1_wbteleop_npz_play_real_config_is_registered() -> None:
    cfg_class = cfg_registry.get("g1_wbteleop_npz_play_real")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1RealEnvCfg)
    assert cfg.env.env_type == "UnitreeCppEnv"
    assert cfg.env.born_place_align is False
    assert cfg.env.limit_pd_target_effort is False
    assert cfg.env.clip_pd_target is False
    assert cfg.env.pd_target_max_delta is None
    assert cfg.env.dof.default_pos == G1TrackingBfmSparseDoF().default_pos
    assert len(cfg.ctrl) == 2
    assert isinstance(cfg.ctrl[0], UnitreeCtrlCfg)
    assert cfg.ctrl[0].triggers["Y"] == "[MOTION_RESET]"
    assert cfg.ctrl[0].triggers["X"] == "[MOTION_FADE_IN]"
    assert cfg.ctrl[0].triggers["B"] == "[MOTION_FADE_OUT]"
    assert isinstance(cfg.ctrl[1], WbTeleopNpzPlaybackCtrlCfg)
    assert cfg.ctrl[1].auto_start is False
    assert isinstance(cfg.policy, G1WbTeleopOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "WbTeleopNpzPlaybackCtrl"
    assert isinstance(cfg.hold_policy, G1UnitreeWoGaitPolicyCfg)
    assert cfg.do_safety_check is True


def test_g1_tracking_bfm_sparse_policy_uses_knees_bent_default_pose() -> None:
    cfg = G1TrackingBfmSparseOnnxPolicyCfg()

    expected = G1TrackingBfmSparseDoF().default_pos

    assert cfg.obs_dof.default_pos == expected
    assert cfg.action_dof.default_pos == expected
    assert cfg.action_dof.default_pos != cfg_registry.get("g1_tracking_bfm_keyboard_sim")().env.dof.default_pos


def test_g1_tracking_bfm_sparse_dof_uses_training_effort_limits() -> None:
    cfg = G1TrackingBfmSparseOnnxPolicyCfg()

    assert cfg.action_dof.torque_limits == [
        *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 50.0, 50.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
    ]
