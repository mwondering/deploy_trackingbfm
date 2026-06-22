"""Sim-rate smoothing for low-frequency teleop reference commands.

The Pico retarget controller produces reference commands at the (lower, jittery)
source rate while the control loop runs at a fixed higher rate. The consumer
therefore sees a "hold-then-jump" staircase, and because the reference joint
velocity is a finite difference of joint positions, every jump becomes a large
velocity spike. Both feed the tracking network and cause visible jumps.

This module applies a second-order critically damped filter to the reference
fields at control rate, in the main process (downstream of ``get_data``). The
filter smooths the position staircase and exposes its own velocity state, which
replaces the spiky finite-difference reference velocity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class CriticallyDampedFilter:
    """Vectorized second-order critically damped smoothing filter.

    Tracks a target signal with position state ``x`` and velocity state ``v``.
    The velocity state is the time derivative of the smoothed signal, so it can
    be used directly as a smoothed reference velocity.
    """

    def __init__(self, cutoff_hz: float, *, max_jump: float | None = None):
        self.omega = 2.0 * math.pi * max(float(cutoff_hz), 1e-3)
        self.max_jump = None if max_jump is None else float(max_jump)
        self._x: np.ndarray | None = None
        self._v: np.ndarray | None = None

    def reset(self, x: Any = None) -> None:
        if x is None:
            self._x = None
            self._v = None
        else:
            self._x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
            self._v = np.zeros_like(self._x)

    def update(self, target: Any, dt: float) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        dt = max(float(dt), 1e-6)

        # First sample, shape change, or teleport -> snap to the target.
        if (
            self._x is None
            or self._x.shape != target.shape
            or (self.max_jump is not None and np.max(np.abs(target - self._x)) > self.max_jump)
        ):
            self.reset(target)
            return self._x.copy(), self._v.copy()

        # Exact critically damped spring step (constant target over dt).
        w = self.omega
        e = math.exp(-w * dt)
        change = self._x - target
        temp = (self._v + w * change) * dt
        self._x = target + (change + temp) * e
        self._v = (self._v - w * temp) * e
        return self._x.copy(), self._v.copy()


class WbTeleopCommandSmoother:
    """Smooth the wbteleop reference fields consumed by the tracking policy.

    Smooths the three fields fed to the network: ``command`` (joint_pos+joint_vel),
    ``ref_limb_ee_pose_b`` and ``motion_ref_ang_vel``. The ``command`` joint
    velocity is replaced by the filter's velocity state to remove finite-difference
    spikes. Filters are reset whenever the controller is not ``active`` so legitimate
    discontinuities (activation, motion reset) are not blended across.
    """

    def __init__(
        self,
        *,
        cutoff_hz: float = 10.0,
        joint_dof: int = 29,
        joint_snap_threshold: float | None = 1.0,
    ):
        self.joint_dof = int(joint_dof)
        self._joint_filter = CriticallyDampedFilter(cutoff_hz, max_jump=joint_snap_threshold)
        self._limb_filter = CriticallyDampedFilter(cutoff_hz)
        self._angvel_filter = CriticallyDampedFilter(cutoff_hz)

    def reset(self) -> None:
        self._joint_filter.reset()
        self._limb_filter.reset()
        self._angvel_filter.reset()

    def smooth(self, ctrl: dict[str, Any], dt: float) -> None:
        """Smooth the reference fields of ``ctrl`` in place."""
        if ctrl.get("state") != "active":
            self.reset()
            return

        command = ctrl.get("command")
        if command is not None:
            command = np.asarray(command, dtype=np.float32).reshape(-1)
            if command.shape[0] == 2 * self.joint_dof:
                joint_pos = command[: self.joint_dof]
                smooth_pos, smooth_vel = self._joint_filter.update(joint_pos, dt)
                ctrl["command"] = np.concatenate([smooth_pos, smooth_vel], dtype=np.float32)

        limb = ctrl.get("ref_limb_ee_pose_b")
        if limb is not None:
            smooth_limb, _ = self._limb_filter.update(limb, dt)
            ctrl["ref_limb_ee_pose_b"] = smooth_limb

        ang_vel = ctrl.get("motion_ref_ang_vel")
        if ang_vel is not None:
            smooth_ang_vel, _ = self._angvel_filter.update(ang_vel, dt)
            ctrl["motion_ref_ang_vel"] = smooth_ang_vel
