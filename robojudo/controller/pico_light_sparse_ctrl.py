from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import PicoLightSparseCtrlCfg
from robojudo.controller.pico_retarget_tracking_bfm_ctrl import (
    _make_real_retarget,
    _make_real_snapshot_builder,
    _make_real_streamer,
)
from robojudo.controller.utils.latest_output_worker import LatestOutputWorker
from robojudo.tools.tracking_bfm_sparse_command import extract_tracking_bfm_sparse_command


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


def _apply_relative_quat(default_quat_xyzw: np.ndarray, relative_quat_xyzw: np.ndarray) -> np.ndarray:
    quat = sRot.from_quat(default_quat_xyzw) * sRot.from_quat(relative_quat_xyzw)
    return quat.as_quat().astype(np.float32)


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


def _controller(controller_data, name: str) -> dict[str, Any]:
    if not isinstance(controller_data, dict):
        return {}
    controller = controller_data.get(name, {})
    return controller if isinstance(controller, dict) else {}


def _button(controller_data, controller_name: str, *button_names: str) -> bool:
    controller = _controller(controller_data, controller_name)
    return any(bool(controller.get(name, False)) for name in button_names)


def _axis_pair(controller_data, controller_name: str) -> tuple[float, float]:
    controller = _controller(controller_data, controller_name)
    for key in ("axis", "joystick", "stick", "thumbstick", "primary2DAxis"):
        value = controller.get(key)
        if isinstance(value, list | tuple | np.ndarray) and len(value) >= 2:
            return float(value[0]), float(value[1])
    x = controller.get("axis_x", controller.get("x", 0.0))
    y = controller.get("axis_y", controller.get("y", 0.0))
    return float(x or 0.0), float(y or 0.0)


def _analog(controller_data, controller_name: str, *names: str) -> float:
    controller = _controller(controller_data, controller_name)
    for name in names:
        value = controller.get(name)
        if value is not None:
            return float(value)
    return 0.0


def _timestamp_ns(controller_data) -> int:
    if isinstance(controller_data, dict):
        timestamp = controller_data.get("timestamp")
        if timestamp is not None:
            return int(timestamp)
    return int(time.time() * 1e9)


def _retarget_ee_pose(
    cfg: PicoLightSparseCtrlCfg,
    retarget,
    snapshot_builder,
    smplx_data,
    timestamp_ns: int,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    t_last = time.perf_counter()
    qpos = np.asarray(
        retarget.retarget(smplx_data, offset_to_ground=cfg.offset_to_ground),
        dtype=np.float32,
    ).copy()
    t_now = time.perf_counter()
    if timings is not None:
        timings["retarget"] = (t_now - t_last) * 1000.0
    t_last = t_now
    if qpos.shape[0] >= 3:
        qpos[2] += float(cfg.root_z_offset)

    snapshot = snapshot_builder.build(qpos, timestamp_ns=timestamp_ns)
    t_now = time.perf_counter()
    if timings is not None:
        timings["snapshot"] = (t_now - t_last) * 1000.0
    t_last = t_now
    command = extract_tracking_bfm_sparse_command(
        snapshot,
        anchor_body_name=cfg.anchor_body_name,
        ee_body_names=(cfg.left_ee_body_name, cfg.right_ee_body_name),
    )
    t_now = time.perf_counter()
    if timings is not None:
        timings["sparse_extract"] = (t_now - t_last) * 1000.0
    return np.asarray(command["ee_pose"], dtype=np.float32).copy()


class _RetargetPicoReader:
    def __init__(self, cfg: PicoLightSparseCtrlCfg, streamer=None, retarget=None, snapshot_builder=None):
        self.cfg = cfg
        self.streamer = streamer or _make_real_streamer()
        self.retarget = retarget or _make_real_retarget(cfg)
        self.snapshot_builder = snapshot_builder or _make_real_snapshot_builder(cfg)

    def read(self) -> dict[str, Any]:
        timings: dict[str, float] = {}
        t_last = time.perf_counter()
        (
            smplx_data,
            _left_hand_data,
            _right_hand_data,
            controller_data,
            _headset_data,
        ) = self.streamer.get_current_frame()
        t_now = time.perf_counter()
        timings["pico_read"] = (t_now - t_last) * 1000.0
        timestamp_ns = _timestamp_ns(controller_data)
        retarget_ee_pose = None
        if smplx_data is not None:
            retarget_ee_pose = _retarget_ee_pose(
                self.cfg,
                self.retarget,
                self.snapshot_builder,
                smplx_data,
                timestamp_ns,
                timings,
            )

        return {
            "timestamp_ns": timestamp_ns,
            "left_axis": _axis_pair(controller_data, "LeftController"),
            "right_axis": _axis_pair(controller_data, "RightController"),
            "left_trigger": _analog(controller_data, "LeftController", "index_trig", "trigger", "trigger_value"),
            "right_trigger": _analog(controller_data, "RightController", "index_trig", "trigger", "trigger_value"),
            "left_grip": _analog(controller_data, "LeftController", "grip", "grip_value"),
            "right_grip": _analog(controller_data, "RightController", "grip", "grip_value"),
            "left_pos": np.zeros(3, dtype=np.float32),
            "right_pos": np.zeros(3, dtype=np.float32),
            "left_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "right_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "headset_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "left_x": _button(controller_data, "LeftController", "key_one", "X", "x"),
            "right_a": _button(controller_data, "RightController", "key_one", "A", "a"),
            "left_axis_click": _button(controller_data, "LeftController", "axis_click", "axisClick"),
            "right_axis_click": _button(controller_data, "RightController", "axis_click", "axisClick"),
            "_retarget_ee_pose": retarget_ee_pose,
            "_profile_timings": timings,
        }

@ctrl_registry.register
class PicoLightSparseCtrl(Controller):
    cfg_ctrl: PicoLightSparseCtrlCfg

    def __init__(
        self,
        cfg_ctrl: PicoLightSparseCtrlCfg,
        env=None,
        device="cpu",
        sdk_reader=None,
        ee_pose_source=None,
    ):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        if sdk_reader is None and self.cfg_ctrl.retarget_ee_pose and ee_pose_source is None:
            self._sdk_reader = _RetargetPicoReader(self.cfg_ctrl)
        else:
            self._sdk_reader = sdk_reader or _PicoSdkReader()
        self._ee_pose_source = ee_pose_source
        self._async_worker = None
        self.reset()
        if self.cfg_ctrl.async_read:
            self._async_worker = LatestOutputWorker(
                name="PicoLightSparseCtrlWorker",
                producer=self._get_data_sync,
                initial_output=self._last_output,
                sleep_s=self.cfg_ctrl.async_worker_sleep_s,
                profile_enabled=self.cfg_ctrl.async_profile,
                profile_interval=self.cfg_ctrl.async_profile_interval,
            )
            self._async_worker.start()

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
        if self._async_worker is not None:
            self._async_worker.reset(self._last_output)

    def _ee_pose_from_delta(
        self,
        *,
        left_delta: np.ndarray | None = None,
        right_delta: np.ndarray | None = None,
        left_relative_quat: np.ndarray | None = None,
        right_relative_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        left_delta = np.zeros(3, dtype=np.float32) if left_delta is None else np.asarray(left_delta, dtype=np.float32)
        right_delta = (
            np.zeros(3, dtype=np.float32) if right_delta is None else np.asarray(right_delta, dtype=np.float32)
        )
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        left_relative_quat = (
            identity_quat if left_relative_quat is None else np.asarray(left_relative_quat, dtype=np.float32)
        )
        right_relative_quat = (
            identity_quat if right_relative_quat is None else np.asarray(right_relative_quat, dtype=np.float32)
        )

        left_default_pos = np.asarray(self.cfg_ctrl.ee_default_left_pos_b, dtype=np.float32)
        right_default_pos = np.asarray(self.cfg_ctrl.ee_default_right_pos_b, dtype=np.float32)
        left_default_quat = np.asarray(self.cfg_ctrl.ee_default_left_quat_b_xyzw, dtype=np.float32)
        right_default_quat = np.asarray(self.cfg_ctrl.ee_default_right_quat_b_xyzw, dtype=np.float32)
        left_quat = _apply_relative_quat(left_default_quat, left_relative_quat)
        right_quat = _apply_relative_quat(right_default_quat, right_relative_quat)

        return np.concatenate(
            [
                left_default_pos + left_delta,
                _rot6d_from_quat_xyzw(left_quat),
                right_default_pos + right_delta,
                _rot6d_from_quat_xyzw(right_quat),
            ]
        ).astype(np.float32)

    def _neutral_output(self, *, commands: list[str]) -> dict[str, Any]:
        return {
            "ee_pose": self._ee_pose_from_delta(),
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

    def _active_output(
        self,
        frame: dict[str, Any],
        dt: float,
        commands: list[str],
        timings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
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

        retarget_ee_pose = frame.get("_retarget_ee_pose")
        if retarget_ee_pose is not None:
            ee_pose = np.asarray(retarget_ee_pose, dtype=np.float32).copy()
            if timings is not None:
                timings["ee_pose_select"] = 0.0
        elif self._ee_pose_source is not None:
            t_last = time.perf_counter()
            ee_pose = self._ee_pose_source.get_ee_pose(int(frame["timestamp_ns"]))
            t_now = time.perf_counter()
            if timings is not None:
                timings["ee_pose_source"] = (t_now - t_last) * 1000.0
            if ee_pose is None:
                ee_pose = np.asarray(self._last_output["ee_pose"], dtype=np.float32).copy()
        elif self.cfg_ctrl.retarget_ee_pose:
            ee_pose = np.asarray(self._last_output["ee_pose"], dtype=np.float32).copy()
            if timings is not None:
                timings["ee_pose_fallback"] = 0.0
        else:
            t_last = time.perf_counter()
            left_robot = _unity_pos_to_robot(frame["left_pos"])
            right_robot = _unity_pos_to_robot(frame["right_pos"])
            left_quat_robot = _unity_quat_to_robot_xyzw(frame["left_quat"])
            right_quat_robot = _unity_quat_to_robot_xyzw(frame["right_quat"])
            left_delta = (left_robot - self._anchor_left_robot) * self.cfg_ctrl.ee_scale
            right_delta = (right_robot - self._anchor_right_robot) * self.cfg_ctrl.ee_scale
            left_relative_quat = _relative_quat_xyzw(self._anchor_left_quat_robot, left_quat_robot)
            right_relative_quat = _relative_quat_xyzw(self._anchor_right_quat_robot, right_quat_robot)
            ee_pose = self._ee_pose_from_delta(
                left_delta=left_delta,
                right_delta=right_delta,
                left_relative_quat=left_relative_quat,
                right_relative_quat=right_relative_quat,
            )
            t_now = time.perf_counter()
            if timings is not None:
                timings["delta_mapping"] = (t_now - t_last) * 1000.0

        if timings is not None:
            timings["active_output"] = (time.perf_counter() - t_start) * 1000.0

        return {
            "ee_pose": ee_pose,
            "base_lin_vel_b": base_lin_vel_b,
            "base_ang_vel_b": base_ang_vel_b,
            "anchor_height_w": np.array([self._base_height], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": frame["timestamp_ns"],
            "_commands": commands,
        }

    def _get_data_sync(self):
        timings: dict[str, float] = {}
        t_last = time.perf_counter()
        frame = self._sdk_reader.read()
        t_now = time.perf_counter()
        timings.update(frame.pop("_profile_timings", {}))
        timings.setdefault("pico_read", (t_now - t_last) * 1000.0)
        t_last = t_now
        dt = self._dt(int(frame["timestamp_ns"]))
        commands = self._step_state_machine(frame)
        t_now = time.perf_counter()
        timings["state_machine"] = (t_now - t_last) * 1000.0

        if self.state == "active":
            self._last_output = self._active_output(frame, dt, commands, timings)
        elif self.state == "idle":
            self._last_output = self._neutral_output(commands=commands)
        else:
            self._last_output = dict(self._last_output)
            self._last_output["state"] = self.state
            self._last_output["timestamp_ns"] = frame["timestamp_ns"]
            self._last_output["_commands"] = commands
        self._last_output["_profile_timings"] = dict(timings)

        return dict(self._last_output)

    def get_data(self):
        if self._async_worker is not None:
            return self._async_worker.get_data()
        return self._get_data_sync()

    def process_triggers(self, ctrl_data):
        commands = list(ctrl_data.pop("_commands", []))
        return ctrl_data, commands
