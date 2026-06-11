from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_run_pipeline_module():
    spec = importlib.util.spec_from_file_location("run_pipeline", Path("scripts/run_pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_motion_file_override_sets_npz_playback_controller() -> None:
    run_pipeline = _load_run_pipeline_module()
    cfg = SimpleNamespace(ctrl=[SimpleNamespace(ctrl_type="WbTeleopNpzPlaybackCtrl", motion_file="")])

    run_pipeline._apply_motion_file_override(cfg, "/tmp/motion.npz")

    assert cfg.ctrl[0].motion_file == "/tmp/motion.npz"


def test_apply_motion_file_override_rejects_configs_without_motion_file() -> None:
    run_pipeline = _load_run_pipeline_module()
    cfg = SimpleNamespace(ctrl=[SimpleNamespace(ctrl_type="KeyboardCtrl")])

    with pytest.raises(ValueError, match="does not support --motion-file"):
        run_pipeline._apply_motion_file_override(cfg, "/tmp/motion.npz")


def test_sim_default_pose_message_mentions_npz_replay_start_key() -> None:
    run_pipeline = _load_run_pipeline_module()
    cfg = SimpleNamespace(ctrl=[SimpleNamespace(ctrl_type="WbTeleopNpzPlaybackCtrl", auto_start=False)])

    message = run_pipeline._sim_default_pose_message(cfg)

    assert "|" in message
    assert "replay" in message
