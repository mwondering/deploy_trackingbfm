from robojudo.config import cfg_registry
from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg
from robojudo.config.g1.policy.g1_tracking_bfm_sparse_onnx_policy_cfg import (
    G1TrackingBfmSparseDoF,
    G1TrackingBfmSparseOnnxPolicyCfg,
)
from robojudo.controller.ctrl_cfgs import (
    KeyboardTrackingBfmCtrlCfg,
    PicoLightSparseCtrlCfg,
    PicoRetargetTrackingBfmCtrlCfg,
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


def test_g1_tracking_bfm_pico_light_real_config_is_registered_for_dev_pc_dds() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_pico_light_real")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1RealEnvCfg)
    assert cfg.env.env_type == "UnitreeCppEnv"
    assert cfg.env.unitree.net_if == "eth0"
    assert cfg.env.born_place_align is False
    assert cfg.env.odometry_type == "UNITREE"
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoLightSparseCtrlCfg)
    assert cfg.ctrl[0].retarget_ee_pose is True
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "PicoLightSparseCtrl"
    assert cfg.do_safety_check is True


def test_g1_tracking_bfm_pico_retarget_real_config_is_registered_for_dev_pc_dds() -> None:
    cfg_class = cfg_registry.get("g1_tracking_bfm_pico_retarget_real")
    cfg = cfg_class()

    assert isinstance(cfg.env, G1RealEnvCfg)
    assert cfg.env.env_type == "UnitreeCppEnv"
    assert cfg.env.unitree.net_if == "eth0"
    assert cfg.env.born_place_align is False
    assert len(cfg.ctrl) == 1
    assert isinstance(cfg.ctrl[0], PicoRetargetTrackingBfmCtrlCfg)
    assert isinstance(cfg.policy, G1TrackingBfmSparseOnnxPolicyCfg)
    assert cfg.policy.ctrl_type == "PicoRetargetTrackingBfmCtrl"
    assert cfg.do_safety_check is True


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
