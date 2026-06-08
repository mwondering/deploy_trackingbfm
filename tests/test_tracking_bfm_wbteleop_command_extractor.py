from __future__ import annotations

import unittest

import numpy as np

from robojudo.tools.tracking_bfm_sparse_command import RetargetMotionSnapshot
from robojudo.tools.tracking_bfm_wbteleop_command import (
    DEFAULT_WBTELEOP_LIMB_BODY_NAMES,
    DEFAULT_WBTELEOP_MOTION_BODY_NAMES,
    WbTeleopRetargetCommandExtractor,
)


def _snapshot(qpos: np.ndarray, timestamp_ns: int) -> RetargetMotionSnapshot:
    body_names = DEFAULT_WBTELEOP_MOTION_BODY_NAMES
    body_pos_w = np.zeros((len(body_names), 3), dtype=np.float32)
    body_quat_w = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(body_names), 1))
    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_ang_vel_w = np.zeros_like(body_pos_w)

    body_pos_w[body_names.index("pelvis")] = [0.0, 0.0, 0.8]
    body_ang_vel_w[body_names.index("torso_link")] = [0.1, -0.2, 0.3]
    for i, body_name in enumerate(DEFAULT_WBTELEOP_LIMB_BODY_NAMES):
        body_pos_w[body_names.index(body_name)] = [0.1 * (i + 1), -0.05 * i, 0.9 + 0.02 * i]

    return RetargetMotionSnapshot(
        body_names=body_names,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        timestamp_ns=timestamp_ns,
        qpos=qpos,
    )


class TestTrackingBfmWbTeleopCommandExtractor(unittest.TestCase):
    def test_extracts_wbteleop_terms_from_retargeted_motion(self):
        extractor = WbTeleopRetargetCommandExtractor(joint_dof=29)
        first_qpos = np.concatenate([np.zeros(7, dtype=np.float32), np.arange(29, dtype=np.float32)])
        second_qpos = first_qpos.copy()
        second_qpos[7:] += 0.02

        first = extractor.extract(_snapshot(first_qpos, 1_000_000_000), state="active")
        second = extractor.extract(_snapshot(second_qpos, 1_020_000_000), state="active")

        self.assertEqual(first["command"].shape, (58,))
        self.assertEqual(second["ref_limb_ee_pose_b"].shape, (36,))
        self.assertEqual(second["motion_ref_ang_vel"].shape, (3,))
        np.testing.assert_allclose(first["command"][:29], np.arange(29, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(first["command"][29:], np.zeros(29, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(second["command"][29:], np.ones(29, dtype=np.float32), atol=5e-5)
        np.testing.assert_allclose(second["motion_ref_ang_vel"], [0.1, -0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(second["ref_limb_ee_pose_b"][:3], [0.1, 0.0, 0.1], atol=1e-6)
        self.assertEqual(tuple(second["_ref_body_names"]), DEFAULT_WBTELEOP_MOTION_BODY_NAMES)
        self.assertEqual(second["_ref_body_pos_w"].shape, (len(DEFAULT_WBTELEOP_MOTION_BODY_NAMES), 3))


if __name__ == "__main__":
    unittest.main()
