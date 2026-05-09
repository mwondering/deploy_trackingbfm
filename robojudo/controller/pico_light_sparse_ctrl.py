from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import PicoLightSparseCtrlCfg


def _deadzone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _unity_pos_to_robot(pos_unity: np.ndarray) -> np.ndarray:
    x_u, y_u, z_u = pos_unity
    return np.array([-z_u, -x_u, y_u], dtype=np.float32)


def _unity_quat_to_robot_xyzw(quat_unity_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat_unity_xyzw
    return np.array([-qz, -qx, qy, qw], dtype=np.float32)


def _relative_quat_xyzw(anchor_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    relative = sRot.from_quat(anchor_quat).inv() * sRot.from_quat(current_quat)
    return relative.as_quat().astype(np.float32)


def _rot6d_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    rotmat = sRot.from_quat(quat_xyzw).as_matrix()
    return rotmat[:, :2].reshape(-1).astype(np.float32)


class _PicoSdkReader:
    def __init__(self):
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise ImportError(
                "xrobotoolkit_sdk not found. Activate the Pico runtime environment "
                "before using PicoLightSparseCtrl."
            ) from exc
        self._xrt = xrt
        self._xrt.init()

    def read(self) -> dict[str, Any]:
        xrt = self._xrt
        left_pose = xrt.get_left_controller_pose()
        right_pose = xrt.get_right_controller_pose()
        headset_pose = xrt.get_headset_pose()
        return {
            "timestamp_ns": int(xrt.get_time_stamp_ns()),
            "left_axis": tuple(xrt.get_left_axis()),
            "right_axis": tuple(xrt.get_right_axis()),
            "left_trigger": float(xrt.get_left_trigger()),
            "right_trigger": float(xrt.get_right_trigger()),
            "left_grip": float(xrt.get_left_grip()),
            "right_grip": float(xrt.get_right_grip()),
            "left_pos": np.asarray(left_pose[:3], dtype=np.float32),
            "right_pos": np.asarray(right_pose[:3], dtype=np.float32),
            "left_quat": np.asarray(left_pose[3:7], dtype=np.float32),
            "right_quat": np.asarray(right_pose[3:7], dtype=np.float32),
            "headset_quat": np.asarray(headset_pose[3:7], dtype=np.float32),
            "left_x": bool(xrt.get_X_button()),
            "right_a": bool(xrt.get_A_button()),
            "left_axis_click": bool(xrt.get_left_axis_click()),
            "right_axis_click": bool(xrt.get_right_axis_click()),
        }


@ctrl_registry.register
class PicoLightSparseCtrl(Controller):
    cfg_ctrl: PicoLightSparseCtrlCfg

    def __init__(self, cfg_ctrl: PicoLightSparseCtrlCfg, env=None, device="cpu", sdk_reader=None):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._sdk_reader = sdk_reader or _PicoSdkReader()
        self.reset()

    def reset(self):
        self.state = "idle"
        self._emergency_stop = False
        self._last_timestamp_ns: int | None = None
        self._right_a_prev = False
        self._left_x_prev = False
        self._left_axis_click_prev = False
        self._right_axis_click_prev = False

        self._base_height = float(self.cfg_ctrl.base_height_init)
        self._anchor_left_robot = np.zeros(3, dtype=np.float32)
        self._anchor_right_robot = np.zeros(3, dtype=np.float32)
        self._anchor_left_quat_robot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._anchor_right_quat_robot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self._last_output = self._neutral_output(commands=[])

    def _neutral_output(self, *, commands: list[str]) -> dict[str, Any]:
        return {
            "ee_pose": np.array(
                [
                    *self.cfg_ctrl.ee_neutral_left,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    *self.cfg_ctrl.ee_neutral_right,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            "base_lin_vel_b": np.zeros(3, dtype=np.float32),
            "base_ang_vel_b": np.zeros(3, dtype=np.float32),
            "anchor_height_w": np.array([self._base_height], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": self._last_timestamp_ns or int(time.time() * 1e9),
            "_commands": commands,
        }

    def _dt(self, timestamp_ns: int) -> float:
        if self._last_timestamp_ns is None or timestamp_ns <= self._last_timestamp_ns:
            self._last_timestamp_ns = timestamp_ns
            return 0.01
        dt = (timestamp_ns - self._last_timestamp_ns) * 1e-9
        self._last_timestamp_ns = timestamp_ns
        return float(np.clip(dt, 1e-3, 0.1))

    def _anchor_from_frame(self, frame: dict[str, Any]) -> None:
        self._anchor_left_robot = _unity_pos_to_robot(frame["left_pos"])
        self._anchor_right_robot = _unity_pos_to_robot(frame["right_pos"])
        self._anchor_left_quat_robot = _unity_quat_to_robot_xyzw(frame["left_quat"])
        self._anchor_right_quat_robot = _unity_quat_to_robot_xyzw(frame["right_quat"])

    def _step_state_machine(self, frame: dict[str, Any]) -> list[str]:
        commands: list[str] = []
        right_a_pressed = frame["right_a"] and not self._right_a_prev
        left_x_pressed = frame["left_x"] and not self._left_x_prev
        left_axis_click_pressed = frame["left_axis_click"] and not self._left_axis_click_prev
        right_axis_click_pressed = frame["right_axis_click"] and not self._right_axis_click_prev

        self._right_a_prev = frame["right_a"]
        self._left_x_prev = frame["left_x"]
        self._left_axis_click_prev = frame["left_axis_click"]
        self._right_axis_click_prev = frame["right_axis_click"]

        if left_axis_click_pressed:
            self._emergency_stop = True
            self.state = "exit"
            commands.append("[SHUTDOWN]")
            return commands

        if left_x_pressed:
            self.state = "exit"
            commands.append("[SHUTDOWN]")
            return commands

        if right_a_pressed:
            if self.state == "idle":
                self.state = "active"
                self._anchor_from_frame(frame)
            elif self.state == "active":
                self.state = "pause"
            elif self.state == "pause":
                self.state = "active"
                self._anchor_from_frame(frame)

        if right_axis_click_pressed and self.state in {"idle", "active"}:
            self.state = "active"
            self._anchor_from_frame(frame)

        return commands

    def _active_output(self, frame: dict[str, Any], dt: float, commands: list[str]) -> dict[str, Any]:
        left_axis = frame["left_axis"]
        right_axis = frame["right_axis"]
        lx = _deadzone(float(left_axis[0]), self.cfg_ctrl.stick_deadzone)
        ly = _deadzone(float(left_axis[1]), self.cfg_ctrl.stick_deadzone)
        rx = _deadzone(float(right_axis[0]), self.cfg_ctrl.stick_deadzone)

        base_lin_vel_b = np.array(
            [
                ly * self.cfg_ctrl.vx_scale,
                -lx * self.cfg_ctrl.vy_scale,
                0.0,
            ],
            dtype=np.float32,
        )
        base_ang_vel_b = np.array([0.0, 0.0, -rx * self.cfg_ctrl.wz_scale], dtype=np.float32)

        up = _deadzone(float(frame["left_grip"]), self.cfg_ctrl.trigger_deadzone)
        down = _deadzone(float(frame["left_trigger"]), self.cfg_ctrl.trigger_deadzone)
        self._base_height = float(
            np.clip(
                self._base_height + (up - down) * self.cfg_ctrl.base_height_rate * dt,
                self.cfg_ctrl.base_height_min,
                self.cfg_ctrl.base_height_max,
            )
        )

        left_robot = _unity_pos_to_robot(frame["left_pos"])
        right_robot = _unity_pos_to_robot(frame["right_pos"])
        left_quat_robot = _unity_quat_to_robot_xyzw(frame["left_quat"])
        right_quat_robot = _unity_quat_to_robot_xyzw(frame["right_quat"])
        left_delta = (left_robot - self._anchor_left_robot) * self.cfg_ctrl.ee_scale
        right_delta = (right_robot - self._anchor_right_robot) * self.cfg_ctrl.ee_scale
        left_rot6d = _rot6d_from_quat_xyzw(_relative_quat_xyzw(self._anchor_left_quat_robot, left_quat_robot))
        right_rot6d = _rot6d_from_quat_xyzw(_relative_quat_xyzw(self._anchor_right_quat_robot, right_quat_robot))
        ee_pose = np.concatenate(
            [
                np.asarray(self.cfg_ctrl.ee_neutral_left, dtype=np.float32) + left_delta,
                left_rot6d,
                np.asarray(self.cfg_ctrl.ee_neutral_right, dtype=np.float32) + right_delta,
                right_rot6d,
            ]
        ).astype(np.float32)

        return {
            "ee_pose": ee_pose,
            "base_lin_vel_b": base_lin_vel_b,
            "base_ang_vel_b": base_ang_vel_b,
            "anchor_height_w": np.array([self._base_height], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": frame["timestamp_ns"],
            "_commands": commands,
        }

    def get_data(self):
        frame = self._sdk_reader.read()
        dt = self._dt(int(frame["timestamp_ns"]))
        commands = self._step_state_machine(frame)

        if self.state == "active":
            self._last_output = self._active_output(frame, dt, commands)
        elif self.state == "idle":
            self._last_output = self._neutral_output(commands=commands)
        else:
            self._last_output = dict(self._last_output)
            self._last_output["state"] = self.state
            self._last_output["timestamp_ns"] = frame["timestamp_ns"]
            self._last_output["_commands"] = commands

        return dict(self._last_output)

    def process_triggers(self, ctrl_data):
        commands = list(ctrl_data.pop("_commands", []))
        return ctrl_data, commands
