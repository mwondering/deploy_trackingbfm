from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from box import Box

from robojudo.policy.policy_cfgs import TrackingBfmSparseOnnxPolicyCfg
from robojudo.policy.tracking_bfm_sparse_onnx_policy import TrackingBfmSparseOnnxPolicy
from robojudo.tools.tool_cfgs import DoFConfig

_FAKE_ONNX_MODELS: dict[str, tuple[int, int]] = {}


class _FakeTensorInfo:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class _FakeModelMeta:
    custom_metadata_map: dict[str, str] = {}


class _FakeSingleInputSession:
    def __init__(self, path: str, providers: list[str]):
        del providers
        self.obs_dim, self.output_dim = _FAKE_ONNX_MODELS[path]

    def get_inputs(self):
        return [_FakeTensorInfo("obs", [1, self.obs_dim])]

    def get_outputs(self):
        return [_FakeTensorInfo("action", [1, self.output_dim])]

    def get_modelmeta(self):
        return _FakeModelMeta()

    def run(self, output_names, feeds):
        del output_names
        batch_size = feeds["obs"].shape[0]
        return [np.ones((batch_size, self.output_dim), dtype=np.float32)]


@pytest.fixture(autouse=True)
def _fake_single_input_onnxruntime(monkeypatch):
    monkeypatch.setattr(
        "robojudo.policy.tracking_bfm_sparse_onnx_policy.ort.InferenceSession",
        _FakeSingleInputSession,
    )


def _export_dummy_onnx(path: Path, obs_dim: int = 8, output_dim: int = 4) -> None:
    path.write_text("fake onnx")
    _FAKE_ONNX_MODELS[str(path)] = (obs_dim, output_dim)


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
    include_proprio: bool = False,
    proprio_obs_group: str = "proprio_actor",
) -> None:
    proprio_yaml = ""
    if include_proprio:
        proprio_yaml = f"""
  {proprio_obs_group}:
    terms:
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
"""
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
{proprio_yaml}
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


def test_tracking_bfm_sparse_onnx_policy_builds_latent_obs_and_proprio_from_yaml_metadata(monkeypatch) -> None:
    class _FakeTensorInfo:
        def __init__(self, name: str, shape: list[int]):
            self.name = name
            self.shape = shape

    class _FakeModelMeta:
        custom_metadata_map = {
            "checkpoint_family": "latent_tracking",
            "obs_group": "actor",
            "proprio_obs_group": "proprio_actor",
        }

    class _FakeLatentSession:
        def __init__(self, path: str, providers: list[str]):
            assert path.endswith("latent_policy.onnx")
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                _FakeTensorInfo("obs", [1, 43]),
                _FakeTensorInfo("proprio", [1, 18]),
            ]

        def get_outputs(self):
            return [_FakeTensorInfo("actions", [1, 4])]

        def get_modelmeta(self):
            return _FakeModelMeta()

        def run(self, output_names, feeds):
            assert output_names == ["actions"]
            assert feeds["obs"].shape == (1, 43)
            assert feeds["proprio"].shape == (1, 18)
            return [np.ones((1, 4), dtype=np.float32)]

    monkeypatch.setattr(
        "robojudo.policy.tracking_bfm_sparse_onnx_policy.ort.InferenceSession",
        _FakeLatentSession,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "latent_policy.onnx"
        onnx_path.write_text("fake onnx")
        _write_env_yaml(
            onnx_path.parent / "params" / "env.yaml",
            include_proprio=True,
        )

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

        obs_payload, extras = policy.get_observation(env_data, ctrl_data)
        action = policy.get_action(obs_payload)

        assert extras == {}
        assert policy.obs_group == "actor"
        assert policy.proprio_obs_group == "proprio_actor"
        assert set(obs_payload) == {"obs", "proprio"}
        assert obs_payload["obs"].shape == (43,)
        assert obs_payload["proprio"].shape == (18,)
        assert obs_payload["proprio"].tolist() == pytest.approx(
            [
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
            ]
        )
        assert action.shape == (4,)
