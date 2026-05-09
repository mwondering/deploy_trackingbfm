from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from robojudo.tools.tracking_bfm_sparse_command import (
    RetargetMotionSnapshot,
    extract_tracking_bfm_sparse_command,
)


def _quat_wxyz_from_euler_xyz(rpy: tuple[float, float, float]) -> np.ndarray:
    q_xyzw = R.from_euler("xyz", rpy).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


class TestTrackingBfmSparseCommandExtractor(unittest.TestCase):
    def test_extract_sparse_command_matches_training_term_shapes_and_anchor_height(self):
        snapshot = RetargetMotionSnapshot(
            body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
            body_pos_w=np.array(
                [
                    [0.0, 0.0, 0.8],
                    [0.2, 0.1, 1.0],
                    [-0.2, 0.1, 0.9],
                ],
                dtype=np.float32,
            ),
            body_quat_w=np.array(
                [
                    _quat_wxyz_from_euler_xyz((0.0, 0.0, 0.0)),
                    _quat_wxyz_from_euler_xyz((0.0, 0.0, 0.0)),
                    _quat_wxyz_from_euler_xyz((0.0, 0.0, 0.0)),
                ],
                dtype=np.float32,
            ),
            body_lin_vel_w=np.array(
                [
                    [0.5, -0.3, 0.1],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            body_ang_vel_w=np.array(
                [
                    [0.4, 0.2, -0.1],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            timestamp_ns=123,
        )

        command = extract_tracking_bfm_sparse_command(snapshot)

        self.assertEqual(command["ee_pose"].shape, (18,))
        self.assertEqual(command["base_lin_vel_b"].shape, (3,))
        self.assertEqual(command["base_ang_vel_b"].shape, (3,))
        self.assertEqual(command["anchor_height_w"].shape, (1,))
        np.testing.assert_allclose(command["ee_pose"][:3], [0.2, 0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(command["ee_pose"][9:12], [-0.2, 0.1, 0.1], atol=1e-6)
        np.testing.assert_allclose(command["base_lin_vel_b"], [0.5, -0.3, 0.1], atol=1e-6)
        np.testing.assert_allclose(command["base_ang_vel_b"], [0.4, 0.2, -0.1], atol=1e-6)
        np.testing.assert_allclose(command["anchor_height_w"], [0.8], atol=1e-6)

    def test_extract_sparse_command_rotates_world_velocities_into_pelvis_frame(self):
        pelvis_quat = _quat_wxyz_from_euler_xyz((0.0, 0.0, np.pi / 2.0))
        snapshot = RetargetMotionSnapshot(
            body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
            body_pos_w=np.array(
                [
                    [1.0, 2.0, 0.8],
                    [1.0, 3.0, 0.8],
                    [0.0, 2.0, 0.8],
                ],
                dtype=np.float32,
            ),
            body_quat_w=np.array([pelvis_quat, pelvis_quat, pelvis_quat], dtype=np.float32),
            body_lin_vel_w=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            body_ang_vel_w=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            timestamp_ns=123,
        )

        command = extract_tracking_bfm_sparse_command(snapshot)

        np.testing.assert_allclose(command["base_lin_vel_b"], [0.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(command["base_ang_vel_b"], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(command["ee_pose"][:3], [1.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
