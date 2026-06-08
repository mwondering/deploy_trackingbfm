import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot


class MujocoVisualizer:
    def __init__(self, viewer):
        self.viewer = viewer
        self._ghost_model = None
        self._ghost_data = None
        self._ghost_vopt = None
        self._ghost_pert = None
        self._ghost_marker_id = -991337
        self._patch_viewer_for_ghost_mesh()

    # TODO: reset: clear all markers

    def _patch_viewer_for_ghost_mesh(self):
        if not hasattr(self.viewer, "_add_marker_to_scene"):
            return
        if getattr(self.viewer, "_robojudo_ghost_mesh_patch", False):
            return

        original_add_marker_to_scene = self.viewer._add_marker_to_scene

        def add_marker_or_ghost(marker):
            if marker.get("_robojudo_ghost_mesh", False):
                self._add_ghost_mesh_to_scene(marker["qpos"])
                return
            original_add_marker_to_scene(marker)

        self.viewer._add_marker_to_scene = add_marker_or_ghost
        self.viewer._robojudo_ghost_mesh_patch = True

    def _ensure_ghost_model(self):
        if self._ghost_model is not None:
            return

        self._ghost_model = copy.deepcopy(self.viewer.model)
        for geom_id in range(self._ghost_model.ngeom):
            if (
                self._ghost_model.geom_contype[geom_id] != 0
                or self._ghost_model.geom_conaffinity[geom_id] != 0
            ):
                self._ghost_model.geom_rgba[geom_id, 3] = 0.0
            else:
                self._ghost_model.geom_rgba[geom_id] = np.array([0.25, 0.85, 0.35, 0.35], dtype=np.float32)

        self._ghost_data = mujoco.MjData(self._ghost_model)  # pyright: ignore[reportAttributeAccessIssue]
        self._ghost_vopt = mujoco.MjvOption()  # pyright: ignore[reportAttributeAccessIssue]
        self._ghost_vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # pyright: ignore[reportAttributeAccessIssue]
        self._ghost_pert = mujoco.MjvPerturb()  # pyright: ignore[reportAttributeAccessIssue]

    def _fit_qpos_to_ghost_model(self, qpos):
        self._ensure_ghost_model()
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] == self._ghost_model.nq:
            return qpos.copy()
        if qpos.shape[0] < self._ghost_model.nq:
            fitted = np.asarray(self._ghost_model.qpos0, dtype=np.float64).copy()
            fitted[-qpos.shape[0] :] = qpos
            return fitted
        raise ValueError(f"reference qpos has {qpos.shape[0]} values, ghost model expects {self._ghost_model.nq}")

    def _add_ghost_mesh_to_scene(self, qpos):
        self._ensure_ghost_model()
        self._ghost_data.qpos[:] = self._fit_qpos_to_ghost_model(qpos)
        mujoco.mj_forward(self._ghost_model, self._ghost_data)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mjv_addGeoms(  # pyright: ignore[reportAttributeAccessIssue]
            self._ghost_model,
            self._ghost_data,
            self._ghost_vopt,
            self._ghost_pert,
            mujoco.mjtCatBit.mjCAT_DYNAMIC.value,  # pyright: ignore[reportAttributeAccessIssue]
            self.viewer.scn,
        )

    def update_ref_motion_ghost_mesh(self, qpos):
        self.viewer.add_marker(
            id=self._ghost_marker_id,
            _robojudo_ghost_mesh=True,
            qpos=self._fit_qpos_to_ghost_model(qpos),
        )

    def update_rg_view(self, body_pos, body_rot, humanoid_id):
        if humanoid_id == 0:
            rgba = (1, 0, 0, 1)
        elif humanoid_id == 1:
            rgba = (0, 1, 0, 1)
        else:
            return

        for j in range(body_pos.shape[0]):
            # need to modify mujoco_viewer to support this
            self.viewer.add_marker(
                pos=body_pos[j],
                size=0.05,
                rgba=rgba,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
                label="",
                id=humanoid_id * 1000 + j,
            )

    def update_ref_motion_ghost(self, body_pos, body_names=None, *, id_offset=5000):
        body_pos = np.asarray(body_pos, dtype=float)
        body_names = tuple(body_names or ())
        body_index = {name: idx for idx, name in enumerate(body_names)}

        for j in range(body_pos.shape[0]):
            self.viewer.add_marker(
                pos=body_pos[j],
                size=0.055,
                rgba=(0.45, 0.75, 0.45, 0.65),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
                label="",
                id=id_offset + j,
            )

        ghost_edges = (
            ("pelvis", "left_hip_roll_link"),
            ("left_hip_roll_link", "left_knee_link"),
            ("left_knee_link", "left_ankle_roll_link"),
            ("pelvis", "right_hip_roll_link"),
            ("right_hip_roll_link", "right_knee_link"),
            ("right_knee_link", "right_ankle_roll_link"),
            ("pelvis", "torso_link"),
            ("torso_link", "left_shoulder_roll_link"),
            ("left_shoulder_roll_link", "left_elbow_link"),
            ("left_elbow_link", "left_wrist_yaw_link"),
            ("torso_link", "right_shoulder_roll_link"),
            ("right_shoulder_roll_link", "right_elbow_link"),
            ("right_elbow_link", "right_wrist_yaw_link"),
        )
        for edge_id, (parent, child) in enumerate(ghost_edges):
            if parent not in body_index or child not in body_index:
                continue
            start = body_pos[body_index[parent]]
            end = body_pos[body_index[child]]
            delta = end - start
            length = np.linalg.norm(delta)
            if length <= 1e-6:
                continue
            direction = delta / length
            rot, _ = sRot.align_vectors([direction], [[0, 0, 1]])
            self.viewer.add_marker(
                pos=(start + end) * 0.5,
                mat=rot.as_matrix(),
                size=np.array([0.025, length * 0.5, 0.0]),
                rgba=(0.45, 0.75, 0.45, 0.38),
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # pyright: ignore[reportAttributeAccessIssue]
                label="",
                id=id_offset + 100 + edge_id,
            )

    def draw_arrow(self, origin, root_quat, vec_local, color, scale=1.0, horizontal_only=False, id=0):
        vec_local = np.array(vec_local, dtype=float)
        r = sRot.from_quat(root_quat)

        if horizontal_only:
            yaw = r.as_euler("xyz", degrees=False)[2]
            yaw_rot = sRot.from_euler("z", yaw).as_matrix()
            vec_world = yaw_rot @ vec_local
        else:
            vec_world = r.as_matrix() @ vec_local

        length = np.linalg.norm(vec_world)
        if length > 1e-6:
            dir_world = vec_world / length
            scaled_length = length * scale

            rot, _ = sRot.align_vectors([dir_world], [[0, 0, 1]])
            mat = rot.as_matrix()

            center_pos = origin  # dir_world * scaled_length * 0.5
        else:
            # zero length
            scaled_length = 0
            mat = np.eye(3)
            center_pos = origin

        self.viewer.add_marker(
            pos=center_pos,
            mat=mat,
            size=np.array([0.02, 0.02, scaled_length]),
            rgba=np.array(color),
            type=mujoco.mjtGeom.mjGEOM_ARROW,  # pyright: ignore[reportAttributeAccessIssue]
            id=3000 + id,
        )
