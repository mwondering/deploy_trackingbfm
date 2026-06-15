import os

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    KeyboardTrackingBfmCtrlCfg,  # noqa: F401
    PicoLightSparseCtrlCfg,  # noqa: F401
    PicoRetargetTrackingBfmCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
    WbTeleopNpzPlaybackCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)
from robojudo.tools.debug_log import DebugCfg

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_env_cfg import G1_29DoF  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_tracking_bfm_sparse_onnx_policy_cfg import (  # noqa: F401
    G1TrackingBfmSparseDoF,
    G1TrackingBfmSparseOnnxPolicyCfg,
    G1WbTeleopOnnxPolicyCfg,
)
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401

# ======================== Custom Configs ======================== #
"""
Add your custom config here.
"""

_DEV_PC_UNITREE_NET_IF = os.environ.get("ROBOJUDO_UNITREE_NET_IF", "eth0")


@cfg_registry.register
class g1_dev(RlPipelineCfg):
    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()


@cfg_registry.register
class g1_tracking_bfm_pico_light_sim(RlPipelineCfg):
    """Lightweight Pico -> sparse tracking_bfm ONNX -> G1 MuJoCo."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=True)
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=True,
    )

    ctrl: list[PicoLightSparseCtrlCfg] = [
        PicoLightSparseCtrlCfg(retarget_ee_pose=True),
    ]

    policy: G1TrackingBfmSparseOnnxPolicyCfg = G1TrackingBfmSparseOnnxPolicyCfg()


@cfg_registry.register
class g1_tracking_bfm_pico_light_real(RlPipelineCfg):
    """Dev PC Pico -> sparse tracking_bfm ONNX -> real G1 through wired Unitree DDS."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=True)
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if=_DEV_PC_UNITREE_NET_IF,
        ),
        born_place_align=False,
        limit_pd_target_effort=False,
    )

    ctrl: list[PicoLightSparseCtrlCfg] = [
        PicoLightSparseCtrlCfg(retarget_ee_pose=True),
    ]

    policy: G1TrackingBfmSparseOnnxPolicyCfg = G1TrackingBfmSparseOnnxPolicyCfg()
    do_safety_check: bool = True


@cfg_registry.register
class g1_tracking_bfm_pico_retarget_sim(RlPipelineCfg):
    """Pico full-body retarget -> sparse tracking_bfm ONNX -> G1 MuJoCo."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=True)
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=True,
    )

    ctrl: list[PicoRetargetTrackingBfmCtrlCfg] = [
        PicoRetargetTrackingBfmCtrlCfg(),
    ]

    policy: G1TrackingBfmSparseOnnxPolicyCfg = G1TrackingBfmSparseOnnxPolicyCfg(
        ctrl_type="PicoRetargetTrackingBfmCtrl",
    )


@cfg_registry.register
class g1_tracking_bfm_pico_retarget_real(RlPipelineCfg):
    """Dev PC Pico full-body retarget -> sparse tracking_bfm ONNX -> real G1 through wired Unitree DDS."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=True)
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if=_DEV_PC_UNITREE_NET_IF,
        ),
        born_place_align=False,
        limit_pd_target_effort=False,
    )

    ctrl: list[PicoRetargetTrackingBfmCtrlCfg] = [
        PicoRetargetTrackingBfmCtrlCfg(),
    ]

    policy: G1TrackingBfmSparseOnnxPolicyCfg = G1TrackingBfmSparseOnnxPolicyCfg(
        ctrl_type="PicoRetargetTrackingBfmCtrl",
    )
    do_safety_check: bool = True


@cfg_registry.register
class g1_tracking_bfm_keyboard_sim(RlPipelineCfg):
    """Keyboard-only sparse tracking_bfm ONNX deployment for local MuJoCo testing."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=True)
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=True,
    )

    ctrl: list[KeyboardTrackingBfmCtrlCfg] = [
        KeyboardTrackingBfmCtrlCfg(),
    ]

    policy: G1TrackingBfmSparseOnnxPolicyCfg = G1TrackingBfmSparseOnnxPolicyCfg(
        ctrl_type="KeyboardTrackingBfmCtrl",
    )


@cfg_registry.register
class g1_wbteleop_sim2sim(RlPipelineCfg):
    """Pico full-body retarget -> wbteleop ONNX -> G1 MuJoCo sim2sim."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=False, profile_timing=False, profile_interval=50)
    _deploy_dof = G1_29DoF()
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=True,
        dof=G1TrackingBfmSparseDoF(
            stiffness=_deploy_dof.stiffness,
            damping=_deploy_dof.damping,
            torque_limits=_deploy_dof.torque_limits,
        ),
    )

    ctrl: list[PicoRetargetTrackingBfmCtrlCfg] = [
        PicoRetargetTrackingBfmCtrlCfg(),
    ]

    policy: G1WbTeleopOnnxPolicyCfg = G1WbTeleopOnnxPolicyCfg(
        ctrl_type="PicoRetargetTrackingBfmCtrl",
    )
    hold_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    hold_to_policy_blend_seconds: float = 0.75
    wbteleop_default_base_height: float | None = 0.76


@cfg_registry.register
class g1_wbteleop_real(RlPipelineCfg):
    """Dev PC Pico full-body retarget -> wbteleop ONNX -> real G1 through wired Unitree DDS."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(
        log_obs=False,
        profile_timing=False,
        profile_interval=50,
    )
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if=_DEV_PC_UNITREE_NET_IF,
        ),
        dof=G1TrackingBfmSparseDoF(),
        born_place_align=False,
        limit_pd_target_effort=False,
    )

    ctrl: list[PicoRetargetTrackingBfmCtrlCfg] = [
        PicoRetargetTrackingBfmCtrlCfg(),
    ]

    policy: G1WbTeleopOnnxPolicyCfg = G1WbTeleopOnnxPolicyCfg(
        ctrl_type="PicoRetargetTrackingBfmCtrl",
    )
    hold_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    hold_to_policy_blend_seconds: float = 0.75
    do_safety_check: bool = True


@cfg_registry.register
class g1_wbteleop_npz_play_sim2sim(RlPipelineCfg):
    """NPZ reference motion playback -> wbteleop ONNX -> G1 MuJoCo sim2sim."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(log_obs=False, profile_timing=True, profile_interval=50)
    _deploy_dof = G1_29DoF()
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=False,
        dof=G1TrackingBfmSparseDoF(
            stiffness=_deploy_dof.stiffness,
            damping=_deploy_dof.damping,
            torque_limits=_deploy_dof.torque_limits,
        ),
    )

    ctrl: list[KeyboardCtrlCfg | WbTeleopNpzPlaybackCtrlCfg] = [
        KeyboardCtrlCfg(),
        WbTeleopNpzPlaybackCtrlCfg(auto_start=False),
    ]

    policy: G1WbTeleopOnnxPolicyCfg = G1WbTeleopOnnxPolicyCfg(
        ctrl_type="WbTeleopNpzPlaybackCtrl",
    )
    hold_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    hold_to_policy_blend_seconds: float = 0.75
    wbteleop_default_base_height: float | None = 0.76


@cfg_registry.register
class g1_wbteleop_npz_play_real(RlPipelineCfg):
    """NPZ reference motion playback -> wbteleop ONNX -> real G1 through wired Unitree DDS."""

    robot: str = "g1"
    debug: DebugCfg = DebugCfg(
        log_obs=False,
        profile_timing=True,
        profile_interval=50,
    )
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if=_DEV_PC_UNITREE_NET_IF,
        ),
        dof=G1TrackingBfmSparseDoF(),
        born_place_align=False,
        limit_pd_target_effort=False,
    )

    ctrl: list[UnitreeCtrlCfg | WbTeleopNpzPlaybackCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers={
                "A": "[SHUTDOWN]",
                "Y": "[SHUTDOWN]",
                "Start": "[MOTION_RESET]",
                "X": "[MOTION_FADE_IN]",
                "B": "[MOTION_FADE_OUT]",
            }
        ),
        WbTeleopNpzPlaybackCtrlCfg(auto_start=False),
    ]

    policy: G1WbTeleopOnnxPolicyCfg = G1WbTeleopOnnxPolicyCfg(
        ctrl_type="WbTeleopNpzPlaybackCtrl",
    )
    hold_policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    hold_to_policy_blend_seconds: float = 0.75
    do_safety_check: bool = True
