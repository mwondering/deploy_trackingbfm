import os

from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.policy.policy_cfgs import TrackingBfmSparseOnnxPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

_DEFAULT_TRACKING_BFM_ONNX = os.environ.get(
    "ROBOJUDO_TRACKING_BFM_ONNX",
    (
        "/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/"
        "0529_ckpt/latent_tracking_encoder/deploy_model_4000.onnx"
    ),
)
_DEFAULT_TRACKING_BFM_ENV_YAML = os.environ.get(
    "ROBOJUDO_TRACKING_BFM_ENV_YAML",
    (
        "/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/"
        "0529_ckpt/latent_tracking_encoder/params/env.yaml"
    ),
)


class G1TrackingBfmSparseDoF(G1_29DoF):
    """tracking_bfm sparse actors are trained around the KNEES_BENT default pose."""

    default_pos: list[float] | None = [
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[0.0, 0.0, 0.0],
        *[0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
        *[0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    ]

    # PD gains exactly matching the training env.yaml actuator groups
    # (logs/rsl_rl/0508_distillation_anchor_b/params/env.yaml).
    # Order: [hip_p, hip_r, hip_y, knee, ank_p, ank_r] per leg; [waist_y, waist_r, waist_p];
    #        [sh_p, sh_r, sh_y, elbow, wr_r, wr_p, wr_y] per arm.
    stiffness: list[float] | None = [
        *[40.18, 99.10, 40.18, 99.10, 28.50, 28.50],
        *[40.18, 99.10, 40.18, 99.10, 28.50, 28.50],
        *[40.18, 28.50, 28.50],
        *[14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78],
        *[14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78],
    ]

    damping: list[float] | None = [
        *[2.56, 6.31, 2.56, 6.31, 1.81, 1.81],
        *[2.56, 6.31, 2.56, 6.31, 1.81, 1.81],
        *[2.56, 1.81, 1.81],
        *[0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07],
        *[0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07],
    ]

    torque_limits: list[float] | None = [
        *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 50.0, 50.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
    ]


class G1TrackingBfmSparseOnnxPolicyCfg(TrackingBfmSparseOnnxPolicyCfg):
    """Sparse ONNX actor config for G1 tracking_bfm deployment.

    The default path points at the current sparse 1-stage G1 tracking run so
    the existing ``run_pipeline.py`` entrypoint works without extra CLI plumbing.
    """

    robot: str = "g1"
    onnx_path: str = _DEFAULT_TRACKING_BFM_ONNX
    env_yaml_path: str | None = _DEFAULT_TRACKING_BFM_ENV_YAML
    proprio_obs_group: str | None = "proprio_actor"
    action_scales: list[float] = [
        *[
            0.5475464629911068,
            0.35066146637882434,
            0.5475464629911068,
            0.35066146637882434,
            0.43857731392336724,
            0.43857731392336724,
        ],
        *[
            0.5475464629911068,
            0.35066146637882434,
            0.5475464629911068,
            0.35066146637882434,
            0.43857731392336724,
            0.43857731392336724,
        ],
        *[0.5475464629911068, 0.43857731392336724, 0.43857731392336724],
        *[
            0.43857731392336724,
            0.43857731392336724,
            0.43857731392336724,
            0.43857731392336724,
            0.43857731392336724,
            0.07450087032950714,
            0.07450087032950714,
        ],
        *[
            0.43857731392336724,
            0.43857731392336724,
            0.43857731392336724,
            0.43857731392336724,
            0.43857731392336724,
            0.07450087032950714,
            0.07450087032950714,
        ],
    ]

    obs_dof: DoFConfig = G1TrackingBfmSparseDoF()
    action_dof: DoFConfig = G1TrackingBfmSparseDoF()
