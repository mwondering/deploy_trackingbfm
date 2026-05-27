from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_joint_log.py"
_SPEC = importlib.util.spec_from_file_location("plot_joint_log", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_plot_joint_log = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_plot_joint_log)

extract_joint_series = _plot_joint_log.extract_joint_series
load_joint_names = _plot_joint_log.load_joint_names
resolve_joint_index = _plot_joint_log.resolve_joint_index


def _write_log(log_dir, records):
    log_dir.mkdir()
    with open(log_dir / "log.msgpack", "wb") as f:
        for record in records:
            f.write(msgpack.packb(record, use_bin_type=True))


def test_extract_joint_series_by_index(tmp_path):
    log_dir = tmp_path / "log_case"
    _write_log(
        log_dir,
        [
            {
                "time": 10.0,
                "env_data": {
                    "dof_pos": np.array([1.0, 2.0], dtype=np.float32),
                    "dof_vel": np.array([0.1, 0.2], dtype=np.float32),
                },
                "pd_target": np.array([1.5, 2.5], dtype=np.float32),
            },
            {
                "time": 10.02,
                "env_data": {
                    "dof_pos": np.array([3.0, 4.0], dtype=np.float32),
                    "dof_vel": np.array([0.3, 0.4], dtype=np.float32),
                },
                "pd_target": np.array([3.5, 4.5], dtype=np.float32),
            },
        ],
    )

    series = extract_joint_series(log_dir, joint_index=1)

    np.testing.assert_allclose(series["time_s"], [0.0, 0.02])
    np.testing.assert_allclose(series["pos"], [2.0, 4.0])
    np.testing.assert_allclose(series["vel"], [0.2, 0.4])
    np.testing.assert_allclose(series["target"], [2.5, 4.5])


def test_resolve_joint_name_from_config(tmp_path):
    log_dir = tmp_path / "log_case"
    log_dir.mkdir()
    with open(log_dir / "config.json", "w") as f:
        json.dump({"env": {"dof": {"joint_names": ["left_knee_joint", "right_knee_joint"]}}}, f)

    joint_names = load_joint_names(log_dir)

    assert joint_names == ["left_knee_joint", "right_knee_joint"]
    assert resolve_joint_index("right_knee_joint", joint_names) == 1
