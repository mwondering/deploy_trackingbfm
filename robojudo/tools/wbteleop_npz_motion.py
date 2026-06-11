from __future__ import annotations

import os
from typing import Literal

import numpy as np

from robojudo.tools.tracking_bfm_sparse_command import RetargetMotionSnapshot
from robojudo.tools.tracking_bfm_wbteleop_command import DEFAULT_WBTELEOP_MOTION_BODY_NAMES

ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

MUJOCO_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

_ISAACLAB_TO_MUJOCO_JOINT_REINDEX = [ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES]
_ISAACLAB_TO_MUJOCO_BODY_REINDEX = [ISAACLAB_BODY_NAMES.index(name) for name in MUJOCO_BODY_NAMES]
_REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class WbTeleopNpzMotionLoader:
    """Load tracking_bfm npz reference motions for RoboJuDo wbteleop playback."""

    def __init__(
        self,
        motion_file: str | os.PathLike[str],
        *,
        motion_type: Literal["isaaclab", "mujoco"] = "isaaclab",
        body_names: tuple[str, ...] = DEFAULT_WBTELEOP_MOTION_BODY_NAMES,
    ):
        self.motion_file = os.fspath(motion_file)
        if not os.path.isfile(self.motion_file):
            raise FileNotFoundError(f"Invalid motion file: {self.motion_file}")
        if motion_type not in {"isaaclab", "mujoco"}:
            raise ValueError(f"Unsupported motion_type: {motion_type}")
        self.motion_type = motion_type
        self.body_names = tuple(body_names)

        data = np.load(self.motion_file)
        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"tracking_bfm npz missing required keys: {missing}")

        self.fps = float(np.asarray(data["fps"]).reshape(()))
        if self.fps <= 0:
            raise ValueError(f"motion fps must be positive, got {self.fps}")

        self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        self._body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        self._body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        self._body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        self._body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

        self._validate_shapes()
        if self.motion_type == "isaaclab":
            self.joint_pos = self.joint_pos[:, _ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
            self.joint_vel = self.joint_vel[:, _ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
            self._body_pos_w = self._body_pos_w[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
            self._body_quat_w = self._body_quat_w[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
            self._body_lin_vel_w = self._body_lin_vel_w[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
            self._body_ang_vel_w = self._body_ang_vel_w[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]

        self._body_indexes = [MUJOCO_BODY_NAMES.index(body_name) for body_name in self.body_names]
        self.total_frames = int(self.joint_pos.shape[0])
        self.dt_ns = int(round(1e9 / self.fps))

    def _validate_shapes(self) -> None:
        if self.joint_pos.ndim != 2 or self.joint_pos.shape[1] != len(MUJOCO_JOINT_NAMES):
            raise ValueError(f"joint_pos must have shape (T, 29), got {self.joint_pos.shape}")
        if self.joint_vel.shape != self.joint_pos.shape:
            raise ValueError(f"joint_vel shape {self.joint_vel.shape} must match joint_pos {self.joint_pos.shape}")
        expected_body_shape = (self.joint_pos.shape[0], len(MUJOCO_BODY_NAMES), 3)
        expected_quat_shape = (self.joint_pos.shape[0], len(MUJOCO_BODY_NAMES), 4)
        for key, value, expected in (
            ("body_pos_w", self._body_pos_w, expected_body_shape),
            ("body_quat_w", self._body_quat_w, expected_quat_shape),
            ("body_lin_vel_w", self._body_lin_vel_w, expected_body_shape),
            ("body_ang_vel_w", self._body_ang_vel_w, expected_body_shape),
        ):
            if value.shape != expected:
                raise ValueError(f"{key} must have shape {expected}, got {value.shape}")

    def snapshot_at(self, frame_index: int, *, timestamp_ns: int | None = None) -> RetargetMotionSnapshot:
        if self.total_frames <= 0:
            raise ValueError("motion has no frames")
        frame_index = int(np.clip(frame_index, 0, self.total_frames - 1))
        if timestamp_ns is None:
            timestamp_ns = frame_index * self.dt_ns

        body_pos_w = self._body_pos_w[frame_index, self._body_indexes, :].copy()
        body_quat_w = self._body_quat_w[frame_index, self._body_indexes, :].copy()
        body_lin_vel_w = self._body_lin_vel_w[frame_index, self._body_indexes, :].copy()
        body_ang_vel_w = self._body_ang_vel_w[frame_index, self._body_indexes, :].copy()
        qpos = np.concatenate(
            [
                self._body_pos_w[frame_index, 0, :],
                self._body_quat_w[frame_index, 0, :],
                self.joint_pos[frame_index],
            ],
            dtype=np.float32,
        )

        return RetargetMotionSnapshot(
            body_names=self.body_names,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            timestamp_ns=int(timestamp_ns),
            qpos=qpos,
            joint_vel=self.joint_vel[frame_index].copy(),
        )
