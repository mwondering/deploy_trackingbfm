from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np


def _load_callbacks_module():
    path = Path("third_party/mujoco_viewer/mujoco_viewer/callbacks.py")
    spec = importlib.util.spec_from_file_location("mujoco_viewer_callbacks", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_id_converts_mjv_select_numpy_output_to_python_int() -> None:
    callbacks = _load_callbacks_module()
    pert = mujoco.MjvPerturb()

    pert.skinselect = callbacks._selection_id(np.array([[-1]], dtype=np.int32))

    assert pert.skinselect == -1
    assert isinstance(callbacks._selection_id(np.array([[3]], dtype=np.int32)), int)
