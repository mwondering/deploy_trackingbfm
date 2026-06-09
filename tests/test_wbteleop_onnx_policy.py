from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from box import Box

from robojudo.config.g1.g1_custom_cfg import g1_wbteleop_real, g1_wbteleop_sim2sim
from robojudo.policy.policy_cfgs import TrackingBfmSparseOnnxPolicyCfg
from robojudo.policy.wbteleop_onnx_policy import WbTeleopOnnxPolicy
from robojudo.tools.kinematics import MujocoKinematics
from robojudo.tools.tool_cfgs import DoFConfig

_FAKE_ONNX_MODELS: dict[str, tuple[int, int]] = {}


class _FakeTensorInfo:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class _FakeModelMeta:
    custom_metadata_map = {"obs_group": "actor"}


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
def _fake_onnxruntime(monkeypatch):
    monkeypatch.setattr("robojudo.policy.wbteleop_onnx_policy.ort.InferenceSession", _FakeSingleInputSession)


def _write_env_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
observations:
  actor:
    terms:
      command:
        history_length: 0
      ref_limb_ee_pose_b:
        params:
          body_names:
            - left_wrist_yaw_link
            - right_wrist_yaw_link
            - left_ankle_roll_link
            - right_ankle_roll_link
          anchor_body_name: pelvis
        history_length: 5
      motion_ref_ang_vel:
        history_length: 0
      robot_limb_ee_pose_b:
        params:
          body_names:
            - left_wrist_yaw_link
            - right_wrist_yaw_link
            - left_ankle_roll_link
            - right_ankle_roll_link
          anchor_body_name: pelvis
        history_length: 5
      projected_gravity:
        history_length: 5
      base_ang_vel:
        history_length: 5
      joint_pos:
        history_length: 5
      joint_vel:
        history_length: 5
      actions:
        history_length: 5
""".strip()
    )


def _make_policy(
    onnx_path: str,
    env_yaml_path: str,
    default_pos: list[float] | None = None,
) -> WbTeleopOnnxPolicy:
    default_pos = default_pos if default_pos is not None else [0.0] * 29
    dof = DoFConfig(
        joint_names=[f"joint_{i}" for i in range(29)],
        default_pos=default_pos,
    )
    cfg = TrackingBfmSparseOnnxPolicyCfg(
        policy_type="WbTeleopOnnxPolicy",
        robot="g1",
        onnx_path=onnx_path,
        env_yaml_path=env_yaml_path,
        expected_obs_dim=886,
        obs_dof=dof,
        action_dof=dof,
        ctrl_type="PicoRetargetTrackingBfmCtrl",
    )
    return WbTeleopOnnxPolicy(cfg)


def _env_data() -> Box:
    fk_info = {}
    for name, pos in {
        "pelvis": [0.0, 0.0, 0.8],
        "left_wrist_yaw_link": [0.1, 0.0, 0.9],
        "right_wrist_yaw_link": [-0.1, 0.0, 0.9],
        "left_ankle_roll_link": [0.1, 0.0, 0.2],
        "right_ankle_roll_link": [-0.1, 0.0, 0.2],
    }.items():
        fk_info[name] = {
            "pos": np.asarray(pos, dtype=np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
    return Box(
        {
            "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "base_ang_vel": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "dof_pos": np.arange(29, dtype=np.float32) * 0.01,
            "dof_vel": np.arange(29, dtype=np.float32) * 0.02,
            "fk_info": fk_info,
        }
    )


def test_wbteleop_onnx_policy_assembles_yaml_ordered_886_dim_observation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        policy = _make_policy(str(onnx_path), str(env_yaml_path))
        ctrl = {
            "PicoRetargetTrackingBfmCtrl": {
                "state": "active",
                "command": np.arange(58, dtype=np.float32),
                "ref_limb_ee_pose_b": np.ones(36, dtype=np.float32),
                "motion_ref_ang_vel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "_ref_body_pos_w": np.zeros((14, 3), dtype=np.float32),
            }
        }

        obs, extras = policy.get_observation(_env_data(), ctrl)

        assert extras == {}
        assert obs.shape == (886,)
        np.testing.assert_allclose(obs[:58], np.arange(58, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obs[58:94], np.ones(36, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obs[238:241], [1.0, 2.0, 3.0], atol=1e-6)
        action = policy.get_action(obs)
        assert action.shape == (29,)


def test_wbteleop_onnx_policy_holds_default_pose_when_retarget_controller_is_idle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        default_pos = (np.arange(29, dtype=np.float32) * 0.01).tolist()
        policy = _make_policy(str(onnx_path), str(env_yaml_path), default_pos=default_pos)
        ctrl = {
            "PicoRetargetTrackingBfmCtrl": {
                "state": "idle",
                "command": np.zeros(58, dtype=np.float32),
                "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
                "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
            }
        }

        obs, _ = policy.get_observation(_env_data(), ctrl)
        action = policy.get_action(obs)

        np.testing.assert_allclose(obs[:29], np.asarray(default_pos, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obs[29:58], np.zeros(29, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(action, np.zeros(29, dtype=np.float32), atol=1e-6)


def test_wbteleop_onnx_policy_uses_default_pose_mode_until_retarget_becomes_active() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        default_pos = (np.arange(29, dtype=np.float32) * 0.02).tolist()
        policy = _make_policy(str(onnx_path), str(env_yaml_path), default_pos=default_pos)
        ctrl = {
            "command": np.zeros(58, dtype=np.float32),
            "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
            "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
        }

        policy.set_default_pose_mode(True)
        obs, _ = policy.get_observation(
            _env_data(),
            {"PicoRetargetTrackingBfmCtrl": {**ctrl, "state": "pause"}},
        )
        assert policy._hold_default_pose is True
        np.testing.assert_allclose(obs[:29], np.asarray(default_pos, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(policy.get_action(obs), np.zeros(29, dtype=np.float32), atol=1e-6)

        obs, _ = policy.get_observation(
            _env_data(),
            {"PicoRetargetTrackingBfmCtrl": {**ctrl, "state": "active"}},
        )
        assert policy._hold_default_pose is True
        np.testing.assert_allclose(obs[:29], np.asarray(default_pos, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(policy.get_action(obs), np.zeros(29, dtype=np.float32), atol=1e-6)

        policy.set_default_pose_mode(False)
        obs, _ = policy.get_observation(
            _env_data(),
            {"PicoRetargetTrackingBfmCtrl": {**ctrl, "state": "active"}},
        )
        assert policy._hold_default_pose is False
        np.testing.assert_allclose(obs[:58], np.zeros(58, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(policy.get_action(obs), np.ones(29, dtype=np.float32), atol=1e-6)


def test_wbteleop_onnx_policy_holds_when_active_reference_is_incomplete() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        default_pos = (np.arange(29, dtype=np.float32) * 0.03).tolist()
        policy = _make_policy(str(onnx_path), str(env_yaml_path), default_pos=default_pos)
        policy.set_default_pose_mode(False)

        obs, _ = policy.get_observation(
            _env_data(),
            {
                "PicoRetargetTrackingBfmCtrl": {
                    "state": "active",
                    "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
                    "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
                }
            },
        )

        assert policy._hold_default_pose is True
        np.testing.assert_allclose(obs[:29], np.asarray(default_pos, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(policy.get_action(obs), np.zeros(29, dtype=np.float32), atol=1e-6)


def test_wbteleop_onnx_policy_holds_when_active_reference_is_non_finite() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        default_pos = (np.arange(29, dtype=np.float32) * 0.03).tolist()
        policy = _make_policy(str(onnx_path), str(env_yaml_path), default_pos=default_pos)
        policy.set_default_pose_mode(False)
        command = np.zeros(58, dtype=np.float32)
        command[0] = np.nan

        obs, _ = policy.get_observation(
            _env_data(),
            {
                "PicoRetargetTrackingBfmCtrl": {
                    "state": "active",
                    "command": command,
                    "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
                    "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
                }
            },
        )

        assert policy._hold_default_pose is True
        assert np.all(np.isfinite(obs))
        np.testing.assert_allclose(obs[:29], np.asarray(default_pos, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(policy.get_action(obs), np.zeros(29, dtype=np.float32), atol=1e-6)


def test_wbteleop_onnx_policy_holds_when_fk_limb_pose_is_non_finite() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        policy = _make_policy(str(onnx_path), str(env_yaml_path))
        policy.set_default_pose_mode(False)
        env_data = _env_data()
        env_data.fk_info["left_wrist_yaw_link"]["pos"][0] = np.inf

        obs, _ = policy.get_observation(
            env_data,
            {
                "PicoRetargetTrackingBfmCtrl": {
                    "state": "active",
                    "command": np.zeros(58, dtype=np.float32),
                    "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
                    "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
                }
            },
        )

        assert policy._hold_default_pose is True
        assert np.all(np.isfinite(obs))
        np.testing.assert_allclose(policy.get_action(obs), np.zeros(29, dtype=np.float32), atol=1e-6)


def test_wbteleop_onnx_policy_resets_history_when_entering_active() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        policy = _make_policy(str(onnx_path), str(env_yaml_path))

        idle_ctrl = {"PicoRetargetTrackingBfmCtrl": {"state": "idle"}}
        policy.get_observation(_env_data(), idle_ctrl)

        active_ref = np.full(36, 7.0, dtype=np.float32)
        active_ctrl = {
            "PicoRetargetTrackingBfmCtrl": {
                "state": "active",
                "command": np.zeros(58, dtype=np.float32),
                "ref_limb_ee_pose_b": active_ref,
                "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
            }
        }

        obs, _ = policy.get_observation(_env_data(), active_ctrl)

        assert policy._hold_default_pose is False
        ref_history = obs[58 : 58 + 5 * 36].reshape(5, 36)
        np.testing.assert_allclose(ref_history, np.tile(active_ref, (5, 1)), atol=1e-6)


def test_wbteleop_onnx_policy_holds_when_robot_limb_pose_is_unavailable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        policy = _make_policy(str(onnx_path), str(env_yaml_path))
        env_data = _env_data()
        env_data.fk_info = None

        obs, _ = policy.get_observation(
            env_data,
            {
                "PicoRetargetTrackingBfmCtrl": {
                    "state": "active",
                    "command": np.zeros(58, dtype=np.float32),
                    "ref_limb_ee_pose_b": np.zeros(36, dtype=np.float32),
                    "motion_ref_ang_vel": np.zeros(3, dtype=np.float32),
                }
            },
        )

        assert policy._hold_default_pose is True
        np.testing.assert_allclose(policy.get_action(obs), np.zeros(29, dtype=np.float32), atol=1e-6)


def test_wbteleop_robot_limb_pose_matches_between_sim_and_real_fk_configs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "policy.onnx"
        onnx_path.write_text("fake onnx")
        _FAKE_ONNX_MODELS[str(onnx_path)] = (886, 29)
        env_yaml_path = onnx_path.parent / "params" / "env.yaml"
        _write_env_yaml(env_yaml_path)
        policy = _make_policy(str(onnx_path), str(env_yaml_path))
        term_cfg = policy.term_cfgs["robot_limb_ee_pose_b"]

        q = np.asarray(policy.default_dof_pos, dtype=np.float32).copy()
        q += np.linspace(-0.03, 0.03, q.shape[0], dtype=np.float32)
        base_pos = np.array([0.7, -0.4, 0.83], dtype=np.float32)
        base_quat = np.array([0.08, -0.03, 0.21, 0.974], dtype=np.float32)
        base_quat /= np.linalg.norm(base_quat)

        def compute_pose(cfg_class):
            cfg = cfg_class().env
            kinematics = MujocoKinematics(cfg.forward_kinematic)
            fk_info = kinematics.forward(
                joint_pos=q,
                base_pos=base_pos,
                base_quat=base_quat,
                base_ang_vel=np.zeros(3, dtype=np.float32),
                base_lin_vel=np.zeros(3, dtype=np.float32),
            )
            env_data = Box(
                {
                    "fk_info": fk_info,
                    "dof_pos": q,
                    "dof_vel": np.zeros_like(q),
                    "base_quat": base_quat,
                    "base_ang_vel": np.zeros(3, dtype=np.float32),
                }
            )
            return policy._fk_limb_pose_b(env_data, term_cfg)

        sim_pose = compute_pose(g1_wbteleop_sim2sim)
        real_pose = compute_pose(g1_wbteleop_real)

        assert sim_pose.shape == (36,)
        assert np.all(np.isfinite(sim_pose))
        np.testing.assert_allclose(real_pose, sim_pose, atol=1e-7)
