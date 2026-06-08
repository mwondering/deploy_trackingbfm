from __future__ import annotations

import mujoco
import numpy as np

from robojudo.config import ASSETS_DIR
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.tools.tracking_bfm_wbteleop_command import DEFAULT_WBTELEOP_MOTION_BODY_NAMES


class _FakeViewer:
    def __init__(self):
        self.markers = []

    def add_marker(self, **marker_params):
        self.markers.append(marker_params)


def test_update_ref_motion_ghost_draws_spheres_and_skeleton_capsules() -> None:
    viewer = _FakeViewer()
    visualizer = MujocoVisualizer(viewer)
    body_pos = np.zeros((len(DEFAULT_WBTELEOP_MOTION_BODY_NAMES), 3), dtype=np.float32)
    for i in range(body_pos.shape[0]):
        body_pos[i] = [0.1 * i, 0.02 * i, 0.8 + 0.01 * i]

    visualizer.update_ref_motion_ghost(body_pos, body_names=DEFAULT_WBTELEOP_MOTION_BODY_NAMES)

    assert len(viewer.markers) > len(DEFAULT_WBTELEOP_MOTION_BODY_NAMES)
    assert {marker["id"] for marker in viewer.markers[: body_pos.shape[0]]} == set(
        range(5000, 5000 + body_pos.shape[0])
    )
    assert any(marker["id"] >= 5100 for marker in viewer.markers)


class _FakeMujocoViewer:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path((ASSETS_DIR / "robots/g1/g1_29dof_rev_1_0.xml").as_posix())
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.markers = []
        self.normal_markers = []

    def add_marker(self, **marker_params):
        self.markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        self.normal_markers.append(marker)


def test_update_ref_motion_ghost_mesh_adds_reference_robot_geoms_to_scene() -> None:
    viewer = _FakeMujocoViewer()
    visualizer = MujocoVisualizer(viewer)
    qpos = np.asarray(viewer.model.qpos0, dtype=np.float32).copy()

    visualizer.update_ref_motion_ghost_mesh(qpos)

    assert len(viewer.markers) == 1
    assert viewer.markers[0]["_robojudo_ghost_mesh"] is True

    before = viewer.scn.ngeom
    viewer._add_marker_to_scene(viewer.markers[0])

    assert viewer.scn.ngeom > before
    assert len(viewer.normal_markers) == 0
    assert any(viewer.scn.geoms[i].type != mujoco.mjtGeom.mjGEOM_SPHERE for i in range(before, viewer.scn.ngeom))
