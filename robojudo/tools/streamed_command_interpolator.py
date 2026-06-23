"""Time-based interpolation buffer for low-rate teleop reference commands.

This is a Python adaptation of the streaming-motion synchronisation used in
GR00T ``gear_sonic_deploy`` (``streamed_motion_merger.hpp`` +
``localmotion_kplanner.hpp``). The reference solves the "reader/writer rate
mismatch" by treating the reference stream as a proper real-time media-sync
problem rather than a "latest value register":

  * frames live on a global timeline (monotonic timestamps / indices),
  * a sliding window keeps a few history frames,
  * the playback cursor advances at the control rate and **interpolates**
    between buffered frames (linear for vectors, SLERP for orientations),
  * a **catch-up reset** bounds accumulated latency,
  * out-of-order / stale frames are rejected,
  * on a dropout the cursor **holds the last frame**.

Adaptation to RoboJuDo: the Pico source delivers a single *latest* retargeted
command per source tick (no look-ahead chunk, no sender-side frame indices), so
we synthesise the global timeline from each command's ``timestamp_ns`` and can
only interpolate between *past* frames. Playback therefore trails the newest
frame by ``target_lag_s`` (≈ one source period) so the cursor normally sits
between two real samples. This removes the hold-then-jump staircase (and the
finite-difference velocity spikes it caused) while bounding latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from robojudo.tools.tracking_bfm_sparse_command import quat_xyzw_to_wxyz, rot6d_from_quat_wxyz
from scipy.spatial.transform import Rotation as R

_ROT6D_COL0 = (0, 2, 4)
_ROT6D_COL1 = (1, 3, 5)


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path SLERP between two wxyz quaternions (port of quat_slerp_d)."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return (result / np.linalg.norm(result)).astype(np.float32)
    theta = np.arccos(min(dot, 1.0))
    sin_theta = np.sin(theta)
    f0 = np.sin((1.0 - t) * theta) / sin_theta
    f1 = np.sin(t * theta) / sin_theta
    return (f0 * q0 + f1 * q1).astype(np.float32)


def quat_wxyz_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Reconstruct a wxyz quaternion from a 6D rotation via Gram-Schmidt.

    Inverse of ``rot6d_from_quat_wxyz`` (which stores the first two columns of the
    rotation matrix, row-interleaved).
    """
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = rot6d[list(_ROT6D_COL0)]
    a2 = rot6d[list(_ROT6D_COL1)]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b1 = a1 / n1
    a2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2)
    if n2 < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b2 = a2 / n2
    b3 = np.cross(b1, b2)
    rotmat = np.stack([b1, b2, b3], axis=1)  # columns
    return quat_xyzw_to_wxyz(R.from_matrix(rotmat).as_quat()).astype(np.float32)


def slerp_rot6d(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Spherically interpolate two 6D rotations."""
    q = quat_slerp_wxyz(quat_wxyz_from_rot6d(r0), quat_wxyz_from_rot6d(r1), t)
    return rot6d_from_quat_wxyz(q)


@dataclass
class _Frame:
    t: float
    joint_pos: np.ndarray | None  # (joint_dof,)
    limb: np.ndarray | None  # (num_limbs * 9,) = [pos(3), rot6d(6)] per limb
    ang_vel: np.ndarray | None  # (3,)


class StreamedCommandInterpolator:
    """Resample the wbteleop reference command to the control rate.

    Buffers timestamped reference frames and, each control step, emits the
    reference interpolated at a fixed-lag playback cursor. Joint positions and
    the angular-velocity term are linearly interpolated; limb orientations use
    SLERP. The reference joint velocity is recomputed as the finite difference of
    the interpolated joint position at the control step, which removes the source
    velocity spikes. Filters reset whenever the controller is not ``active``.
    """

    def __init__(
        self,
        *,
        control_dt: float,
        joint_dof: int = 29,
        target_lag_s: float = 0.033,
        max_lag_s: float = 0.2,
        history_s: float = 0.5,
        joint_snap_threshold: float | None = 1.0,
    ):
        self.control_dt = max(float(control_dt), 1e-4)
        self.joint_dof = int(joint_dof)
        self.target_lag_s = max(float(target_lag_s), 0.0)
        self.max_lag_s = max(float(max_lag_s), self.target_lag_s + self.control_dt)
        self.history_s = max(float(history_s), self.max_lag_s)
        self.joint_snap_threshold = None if joint_snap_threshold is None else float(joint_snap_threshold)

        self._buffer: deque[_Frame] = deque()
        self._playback_t: float | None = None
        self._last_ingest_t: float | None = None
        self._last_out_joint_pos: np.ndarray | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._playback_t = None
        self._last_ingest_t = None
        self._last_out_joint_pos = None

    # -- Ingestion -----------------------------------------------------------
    def _ingest(self, ctrl: dict[str, Any]) -> None:
        command = ctrl.get("command")
        if command is None:
            return
        t = self._frame_time(ctrl)
        # Reject stale / duplicate / out-of-order frames (monotonic timeline).
        if self._last_ingest_t is not None and t <= self._last_ingest_t:
            return

        command = np.asarray(command, dtype=np.float32).reshape(-1)
        joint_pos = command[: self.joint_dof] if command.shape[0] >= self.joint_dof else None

        limb = ctrl.get("ref_limb_ee_pose_b")
        limb = None if limb is None else np.asarray(limb, dtype=np.float32).reshape(-1)
        ang_vel = ctrl.get("motion_ref_ang_vel")
        ang_vel = None if ang_vel is None else np.asarray(ang_vel, dtype=np.float32).reshape(-1)

        self._buffer.append(_Frame(t=t, joint_pos=joint_pos, limb=limb, ang_vel=ang_vel))
        self._last_ingest_t = t
        self._trim()

    def _frame_time(self, ctrl: dict[str, Any]) -> float:
        ts = ctrl.get("timestamp_ns")
        if ts is not None:
            return float(ts) * 1e-9
        # Fall back to a synthetic monotonic clock advanced by one source period.
        return (self._last_ingest_t or 0.0) + max(self.target_lag_s, self.control_dt)

    def _trim(self) -> None:
        if not self._buffer:
            return
        min_t = self._buffer[-1].t - self.history_s
        while len(self._buffer) > 2 and self._buffer[0].t < min_t:
            self._buffer.popleft()

    # -- Playback ------------------------------------------------------------
    def update(self, ctrl: dict[str, Any]) -> None:
        """Replace the reference fields of ``ctrl`` in place with the resample."""
        if ctrl.get("state") != "active":
            self.reset()
            return

        self._ingest(ctrl)
        if not self._buffer:
            return

        newest = self._buffer[-1].t
        oldest = self._buffer[0].t
        discontinuity = False

        if self._playback_t is None:
            self._playback_t = oldest
            discontinuity = True
        else:
            self._playback_t += self.control_dt

        # Catch-up reset: if we fell too far behind (e.g. after a stall), discard
        # the stale pre-gap history and restart playback at the newest frame so we
        # never blend across the gap and latency stays bounded.
        if newest - self._playback_t > self.max_lag_s:
            while len(self._buffer) > 1:
                self._buffer.popleft()
            oldest = self._buffer[0].t
            self._playback_t = newest
            discontinuity = True

        # Dropout hold: never play past the newest buffered frame.
        if self._playback_t > newest:
            self._playback_t = newest
        if self._playback_t < oldest:
            self._playback_t = oldest
            discontinuity = True

        joint_pos, limb, ang_vel, sample_jump = self._sample(self._playback_t)
        discontinuity = discontinuity or sample_jump

        if joint_pos is not None:
            if discontinuity or self._last_out_joint_pos is None or self._last_out_joint_pos.shape != joint_pos.shape:
                joint_vel = np.zeros_like(joint_pos)
            else:
                joint_vel = (joint_pos - self._last_out_joint_pos) / self.control_dt
            self._last_out_joint_pos = joint_pos.copy()
            ctrl["command"] = np.concatenate([joint_pos, joint_vel], dtype=np.float32)

        if limb is not None:
            ctrl["ref_limb_ee_pose_b"] = limb
        if ang_vel is not None:
            ctrl["motion_ref_ang_vel"] = ang_vel

    def _bracket(self, t: float) -> tuple[_Frame, _Frame, float]:
        buf = self._buffer
        if t <= buf[0].t:
            return buf[0], buf[0], 0.0
        if t >= buf[-1].t:
            return buf[-1], buf[-1], 0.0
        for i in range(len(buf) - 1):
            f0, f1 = buf[i], buf[i + 1]
            if f0.t <= t <= f1.t:
                span = f1.t - f0.t
                alpha = 0.0 if span <= 1e-9 else (t - f0.t) / span
                return f0, f1, float(alpha)
        return buf[-1], buf[-1], 0.0

    def _sample(self, t: float):
        f0, f1, alpha = self._bracket(t)

        # Teleport guard: do not blend across a large joint jump, snap to newer.
        jump = False
        if (
            self.joint_snap_threshold is not None
            and f0.joint_pos is not None
            and f1.joint_pos is not None
            and f0.joint_pos.shape == f1.joint_pos.shape
            and np.max(np.abs(f1.joint_pos - f0.joint_pos)) > self.joint_snap_threshold
        ):
            alpha = 1.0
            jump = True

        joint_pos = _lerp(f0.joint_pos, f1.joint_pos, alpha)
        ang_vel = _lerp(f0.ang_vel, f1.ang_vel, alpha)
        limb = self._sample_limb(f0.limb, f1.limb, alpha)
        return joint_pos, limb, ang_vel, jump

    def _sample_limb(self, l0: np.ndarray | None, l1: np.ndarray | None, alpha: float) -> np.ndarray | None:
        if l0 is None:
            return None
        if l1 is None or l1.shape != l0.shape or alpha <= 0.0:
            return l0.copy()
        if alpha >= 1.0:
            return l1.copy()
        if l0.shape[0] % 9 != 0:
            return _lerp(l0, l1, alpha)  # unknown layout -> plain lerp
        out = np.empty_like(l0)
        for limb in range(l0.shape[0] // 9):
            base = limb * 9
            out[base : base + 3] = (1.0 - alpha) * l0[base : base + 3] + alpha * l1[base : base + 3]
            out[base + 3 : base + 9] = slerp_rot6d(l0[base + 3 : base + 9], l1[base + 3 : base + 9], alpha)
        return out


def _lerp(a: np.ndarray | None, b: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if a is None:
        return None
    if b is None or b.shape != a.shape:
        return a.copy()
    return ((1.0 - alpha) * a + alpha * b).astype(np.float32)
