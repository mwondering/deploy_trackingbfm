from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
SHOULDER_ROLL_JOINT_NAME = "left_shoulder_roll_joint"
SHOULDER_ROLL_JOINT_INDEX = LEFT_ARM_JOINT_NAMES.index(SHOULDER_ROLL_JOINT_NAME)
G1_LEFT_ARM_JOINT_INDICES = np.asarray([15, 16, 17, 18, 19, 20, 21], dtype=np.int64)

_NON_GUI_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo"}


def nan_left_arm_joints() -> np.ndarray:
    return np.full(len(LEFT_ARM_JOINT_NAMES), np.nan, dtype=np.float32)


def _as_1d_float_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim > 1:
        return None
    return arr.reshape(-1)


def extract_left_arm_from_robot_vector(
    value: Any,
    joint_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray | None:
    arr = _as_1d_float_array(value)
    if arr is None:
        return None

    if arr.shape[0] == len(LEFT_ARM_JOINT_NAMES):
        return arr.astype(np.float32, copy=True)

    if joint_names is not None:
        names = tuple(str(name) for name in joint_names)
        if len(names) > 0 and arr.shape[0] >= len(names):
            joint_values = arr[-len(names) :]
            try:
                indices = np.asarray([names.index(name) for name in LEFT_ARM_JOINT_NAMES], dtype=np.int64)
            except ValueError:
                indices = None
            if indices is not None and int(indices.max()) < joint_values.shape[0]:
                return joint_values[indices].astype(np.float32, copy=True)

    if arr.shape[0] >= 29:
        return arr[-29:][G1_LEFT_ARM_JOINT_INDICES].astype(np.float32, copy=True)
    return None


def left_arm_joints_or_nan(
    value: Any,
    joint_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    joints = extract_left_arm_from_robot_vector(value, joint_names=joint_names)
    if joints is None:
        return nan_left_arm_joints()
    return joints


def _is_non_gui_backend(backend: str) -> bool:
    backend_lower = str(backend).lower()
    return backend_lower in _NON_GUI_BACKENDS or "inline" in backend_lower


class LeftArmJointDebugPlot:
    def __init__(
        self,
        *,
        window_s: float = 10.0,
        update_hz: float = 10.0,
        joint_names: tuple[str, ...] = LEFT_ARM_JOINT_NAMES,
    ):
        self.window_s = max(float(window_s), 0.1)
        self.update_interval_s = 1.0 / max(float(update_hz), 0.1)
        self.joint_names = tuple(joint_names)
        self._last_update_s = 0.0
        self._times: deque[float] = deque()
        self._retarget: deque[float] = deque()
        self._interp: deque[float] = deque()
        self._actual: deque[float] = deque()
        self._enabled = True

        try:
            import matplotlib

            # TkAgg can fail while resizing zero-sized toolbar icons on some displays.
            matplotlib.rcParams["toolbar"] = "None"
            import matplotlib.pyplot as plt

            self._plt = plt
            backend = str(plt.get_backend())
            if _is_non_gui_backend(backend):
                self._enabled = False
                logger.warning(
                    "Matplotlib left arm joint debug plot disabled: backend=%s is non-GUI. "
                    "Use a GUI backend such as QtAgg or TkAgg and make sure DISPLAY is set.",
                    backend,
                )
                return
            plt.ion()
            self._fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 5.2))
            self._axis = np.asarray(axes, dtype=object).reshape(-1)[0]
            column_specs = (
                ("retarget (raw)", "#0072b2"),
                ("interp", "#e69f00"),
                ("sim actual", "#009e73"),
            )
            self._lines = tuple(
                self._axis.plot([], [], color=color, linewidth=2.0, marker=".", markersize=3.0)[0]
                for _name, color in column_specs
            )
            self._axis.set_facecolor("#111820")
            self._axis.set_xlim(-self.window_s, 0.0)
            self._axis.set_ylim(-1.0, 1.0)
            self._axis.set_xticks([])
            self._axis.set_yticks([])
            self._axis.tick_params(length=0)
            for spine in self._axis.spines.values():
                spine.set_color("#2d3840")
                spine.set_linewidth(0.8)
            manager = getattr(getattr(self._fig, "canvas", None), "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title(
                    "Left shoulder roll: retarget/raw (blue) | interp (orange) | sim actual (green)"
                )
            window = getattr(manager, "window", None)
            if window is not None:
                if hasattr(window, "geometry"):
                    window.geometry("1000x520+60+60")
                if hasattr(window, "minsize"):
                    window.minsize(700, 360)
            if manager is not None and hasattr(manager, "resize"):
                manager.resize(1000, 520)
            patch = getattr(self._fig, "patch", None)
            if patch is not None and hasattr(patch, "set_facecolor"):
                patch.set_facecolor("#0b0f13")
            if hasattr(self._fig, "set_size_inches"):
                self._fig.set_size_inches(10, 5.2, forward=True)
            if hasattr(self._fig, "subplots_adjust"):
                self._fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.08)
            try:
                self._fig.show()
            except Exception as exc:
                logger.warning("Left arm joint debug plot show failed; updates will still be attempted: %s", exc)
            try:
                if hasattr(plt, "show"):
                    plt.show(block=False)
                canvas = getattr(self._fig, "canvas", None)
                if canvas is not None and hasattr(canvas, "draw_idle"):
                    canvas.draw_idle()
                if canvas is not None and hasattr(canvas, "flush_events"):
                    canvas.flush_events()
                plt.pause(0.001)
            except Exception as exc:
                logger.warning("Left arm joint debug plot event pump failed; updates will still be attempted: %s", exc)
            logger.info("Left arm joint debug plot opened with matplotlib backend=%s", backend)
        except Exception as exc:
            self._enabled = False
            logger.warning("Matplotlib left arm joint debug plot disabled: %s", exc)

    def push(self, timestamp_s: float, retarget_joints: Any, interp_joints: Any, actual_joints: Any) -> None:
        if not self._enabled:
            return
        self._times.append(float(timestamp_s))
        self._retarget.append(self._shoulder_roll_or_nan(retarget_joints))
        self._interp.append(self._shoulder_roll_or_nan(interp_joints))
        self._actual.append(self._shoulder_roll_or_nan(actual_joints))
        self._trim()

    def _shoulder_roll_or_nan(self, joints: Any) -> float:
        left_arm = left_arm_joints_or_nan(joints)
        return float(left_arm[SHOULDER_ROLL_JOINT_INDEX])

    def _trim(self) -> None:
        if not self._times:
            return
        min_time_s = self._times[-1] - self.window_s
        while self._times and self._times[0] < min_time_s:
            self._times.popleft()
            self._retarget.popleft()
            self._interp.popleft()
            self._actual.popleft()

    def maybe_update(self) -> None:
        if not self._enabled or not self._times:
            return
        now_s = time.perf_counter()
        if now_s - self._last_update_s < self.update_interval_s:
            return
        self._last_update_s = now_s

        try:
            # Timestamps are absolute epoch seconds (~1.7e9); subtracting in
            # float32 collapses them to ~128 s resolution, stacking every point
            # at x=0. Do the windowing math in float64, then cast for plotting.
            times = np.asarray(self._times, dtype=np.float64)
            times = (times - times[-1]).astype(np.float32)
            retarget = np.asarray(self._retarget, dtype=np.float32)
            interp = np.asarray(self._interp, dtype=np.float32)
            actual = np.asarray(self._actual, dtype=np.float32)
            retarget_line, interp_line, actual_line = self._lines
            retarget_line.set_data(times, retarget)
            interp_line.set_data(times, interp)
            actual_line.set_data(times, actual)
            values = np.concatenate([retarget, interp, actual])
            values = values[np.isfinite(values)]
            self._axis.set_xlim(-self.window_s, 0.0)
            if values.size > 0:
                pad = max(float(np.ptp(values)) * 0.1, 0.05)
                self._axis.set_ylim(float(values.min()) - pad, float(values.max()) + pad)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            self._plt.pause(0.001)
        except Exception as exc:
            self._enabled = False
            logger.warning("Left arm joint debug plot disabled after update failure: %s", exc)

