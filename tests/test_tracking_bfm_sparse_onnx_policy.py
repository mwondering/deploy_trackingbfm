from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from box import Box

from robojudo.policy.tracking_bfm_sparse_onnx_policy import TrackingBfmSparseOnnxPolicy
from robojudo.policy.policy_cfgs import TrackingBfmSparseOnnxPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class _DummyOnnxActor(nn.Module):
    def __init__(self, obs_dim: int = 8, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _export_dummy_onnx(path: Path, obs_dim: int = 8, output_dim: int = 4) -> None:
    model = _DummyOnnxActor(obs_dim=obs_dim, output_dim=output_dim)
    model.eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, obs_dim),),
        str(path),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={},
        dynamo=False,
    )


def _make_cfg(onnx_path: str, *, obs_dim: int = 8, action_dim: int = 4):
    obs_dof = DoFConfig(
        joint_names=[f"obs_joint_{i}" for i in range(action_dim)],
        default_pos=[0.0] * action_dim,
    )
    action_dof = DoFConfig(
        joint_names=[f"act_joint_{i}" for i in range(action_dim)],
        default_pos=[0.0] * action_dim,
    )
    return TrackingBfmSparseOnnxPolicyCfg(
        robot="g1",
        onnx_path=onnx_path,
        expected_obs_dim=obs_dim,
        obs_dof=obs_dof,
        action_dof=action_dof,
    )


def _write_env_yaml(
    path: Path,
    *,
    obs_group: str = "actor",
    history_steps: int = 0,
    future_steps: int = 1,
    robot_history_length: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
observations:
  {obs_group}:
    terms:
      ee_pose:
        params:
          command_name: motion
          ee_body_names:
            - left_wrist_yaw_link
            - right_wrist_yaw_link
          anchor_body_name: pelvis
          history_steps: {history_steps}
          future_steps: {future_steps}
      base_lin_vel_b:
        params:
          command_name: motion
          anchor_body_name: pelvis
          history_steps: {history_steps}
          future_steps: {future_steps}
      base_ang_vel_b:
        params:
          command_name: motion
          anchor_body_name: pelvis
          history_steps: {history_steps}
          future_steps: {future_steps}
      anchor_height_w:
        params:
          command_name: motion
          anchor_body_name: pelvis
          history_steps: {history_steps}
          future_steps: {future_steps}
      projected_gravity:
        history_length: {robot_history_length}
      base_ang_vel:
        history_length: {robot_history_length}
      joint_pos:
        history_length: {robot_history_length}
      joint_vel:
        history_length: {robot_history_length}
      actions:
        history_length: {robot_history_length}
""".strip()
    )


def test_tracking_bfm_sparse_onnx_policy_loads_and_infers_dims() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        _export_dummy_onnx(onnx_path)
        _write_env_yaml(onnx_path.parent / "params" / "env.yaml")

        policy = TrackingBfmSparseOnnxPolicy(_make_cfg(str(onnx_path)))

        assert policy.obs_dim == 8
        assert policy.action_dim == 4


def test_tracking_bfm_sparse_onnx_policy_rejects_action_dim_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        _export_dummy_onnx(onnx_path, output_dim=4)
        _write_env_yaml(onnx_path.parent / "params" / "env.yaml")
        cfg = _make_cfg(str(onnx_path), action_dim=5)

        with pytest.raises(ValueError, match="action dimension"):
            TrackingBfmSparseOnnxPolicy(cfg)


def test_tracking_bfm_sparse_onnx_policy_rejects_obs_dim_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        _export_dummy_onnx(onnx_path, obs_dim=8)
        _write_env_yaml(onnx_path.parent / "params" / "env.yaml")
        cfg = _make_cfg(str(onnx_path), obs_dim=7)

        with pytest.raises(ValueError, match="observation dimension"):
            TrackingBfmSparseOnnxPolicy(cfg)


def test_tracking_bfm_sparse_onnx_policy_runs_inference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        _export_dummy_onnx(onnx_path)
        _write_env_yaml(onnx_path.parent / "params" / "env.yaml")

        policy = TrackingBfmSparseOnnxPolicy(_make_cfg(str(onnx_path)))
        action = policy.get_action(obs=torch.zeros(policy.obs_dim, dtype=torch.float32).numpy())

        assert action.shape == (policy.action_dim,)


def test_tracking_bfm_sparse_onnx_policy_applies_per_joint_action_scales() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        _export_dummy_onnx(onnx_path, obs_dim=8, output_dim=4)
        _write_env_yaml(onnx_path.parent / "params" / "env.yaml")

        cfg = _make_cfg(str(onnx_path))
        cfg.action_scales = [0.1, 0.2, 0.3, 0.4]
        policy = TrackingBfmSparseOnnxPolicy(cfg)

        obs = torch.zeros(policy.obs_dim, dtype=torch.float32).numpy()
        scaled_action = policy.get_action(obs)
        raw_action = policy.last_action.copy()

        assert scaled_action.tolist() == pytest.approx((raw_action * np.asarray(cfg.action_scales)).tolist())


def test_tracking_bfm_sparse_onnx_policy_builds_sparse_observation_from_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        _export_dummy_onnx(onnx_path, obs_dim=43, output_dim=4)
        _write_env_yaml(onnx_path.parent / "params" / "env.yaml")

        policy = TrackingBfmSparseOnnxPolicy(_make_cfg(str(onnx_path), obs_dim=43, action_dim=4))
        policy.reset()

        env_data = Box(
            {
                "dof_pos": torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32).numpy(),
                "dof_vel": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy(),
                "base_quat": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).numpy(),
                "base_ang_vel": torch.tensor([0.5, -0.25, 0.75], dtype=torch.float32).numpy(),
            }
        )
        ctrl_data = Box(
            {
                "PicoLightSparseCtrl": {
                    "ee_pose": torch.arange(18, dtype=torch.float32).numpy(),
                    "base_lin_vel_b": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).numpy(),
                    "base_ang_vel_b": torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32).numpy(),
                    "anchor_height_w": torch.tensor([0.7], dtype=torch.float32).numpy(),
                }
            }
        )

        obs, extras = policy.get_observation(env_data, ctrl_data)

        expected = torch.tensor(
            [
                *range(18),
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.0,
                0.0,
                -1.0,
                0.5,
                -0.25,
                0.75,
                0.1,
                -0.2,
                0.3,
                -0.4,
                1.0,
                2.0,
                3.0,
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        ).numpy()
        assert extras == {}
        assert obs.shape == (43,)
        assert policy.command_history_length == 1
        assert policy.robot_history_length == 1
        assert policy.command_terms == ["ee_pose", "base_lin_vel_b", "base_ang_vel_b", "anchor_height_w"]
        assert policy.robot_terms == ["projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"]
        assert obs.tolist() == pytest.approx(expected.tolist())
