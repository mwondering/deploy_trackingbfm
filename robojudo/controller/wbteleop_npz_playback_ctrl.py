from __future__ import annotations

from typing import Any

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import WbTeleopNpzPlaybackCtrlCfg
from robojudo.tools.tracking_bfm_sparse_command import (
    DEFAULT_EE_BODY_NAMES,
    DEFAULT_SPARSE_ANCHOR_HEIGHT_W,
    DEFAULT_SPARSE_EE_POSE,
    extract_tracking_bfm_sparse_command,
)
from robojudo.tools.tracking_bfm_wbteleop_command import WbTeleopRetargetCommandExtractor
from robojudo.tools.wbteleop_npz_motion import WbTeleopNpzMotionLoader


@ctrl_registry.register
class WbTeleopNpzPlaybackCtrl(Controller):
    cfg_ctrl: WbTeleopNpzPlaybackCtrlCfg

    def __init__(self, cfg_ctrl: WbTeleopNpzPlaybackCtrlCfg, env=None, device="cpu", motion_loader=None):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.motion = motion_loader or WbTeleopNpzMotionLoader(
            cfg_ctrl.motion_file,
            motion_type=cfg_ctrl.motion_type,
        )
        self.wbteleop_extractor = WbTeleopRetargetCommandExtractor(joint_dof=29)
        self.reset()

    def reset(self):
        self.state = "active" if self.cfg_ctrl.auto_start else "idle"
        self.frame_index = 0
        self._started = False
        self._done_command_sent = False
        self.wbteleop_extractor.reset()
        self._last_output = self._neutral_output([])

    def _start_replay(self):
        self.state = "active"
        self.frame_index = 0
        self._started = True
        self._done_command_sent = False
        self.wbteleop_extractor.reset()

    def _stop_replay(self):
        self.state = "idle"
        self.frame_index = 0
        self._started = False
        self._done_command_sent = False
        self.wbteleop_extractor.reset()
        self._last_output = self._neutral_output([])

    def _neutral_output(self, commands: list[str]) -> dict[str, Any]:
        return {
            "ee_pose": DEFAULT_SPARSE_EE_POSE.copy(),
            "base_lin_vel_b": np.zeros(3, dtype=np.float32),
            "base_ang_vel_b": np.zeros(3, dtype=np.float32),
            "anchor_height_w": np.array([DEFAULT_SPARSE_ANCHOR_HEIGHT_W], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": 0,
            "_commands": list(commands),
        }

    def _active_output(self, commands: list[str]) -> dict[str, Any]:
        snapshot = self.motion.snapshot_at(
            self.frame_index,
            timestamp_ns=self.frame_index * self.motion.dt_ns,
        )
        output = extract_tracking_bfm_sparse_command(
            snapshot,
            anchor_body_name="pelvis",
            ee_body_names=DEFAULT_EE_BODY_NAMES,
            state=self.state,
        )
        output.update(self.wbteleop_extractor.extract(snapshot, state=self.state))
        output["_commands"] = list(commands)
        return output

    def get_data(self):
        commands: list[str] = []
        if self.state == "idle":
            self._last_output = self._neutral_output(commands)
            return dict(self._last_output)

        if self.frame_index >= self.motion.total_frames:
            if self.cfg_ctrl.loop:
                self.frame_index = 0
                self.wbteleop_extractor.reset()
                commands.append("[MOTION_RESET]")
                self._started = True
                reset_frame = True
            else:
                self.state = "done"
                if not self._done_command_sent:
                    commands.append("[MOTION_FADE_OUT]")
                    self._done_command_sent = True
                self._last_output = dict(self._last_output)
                self._last_output["state"] = self.state
                self._last_output["_commands"] = list(commands)
                return dict(self._last_output)
        else:
            reset_frame = False

        if not self._started:
            commands.append("[MOTION_RESET]")
            self._started = True
            reset_frame = True

        self.state = "active"
        self._last_output = self._active_output(commands)
        if not reset_frame:
            self.frame_index += 1
        return dict(self._last_output)

    def process_triggers(self, ctrl_data):
        commands = list(ctrl_data.pop("_commands", []))
        return ctrl_data, commands

    def post_step_callback(self, commands: list[str] | None = None):
        for command in commands or []:
            match command:
                case "[MOTION_RESET]" | "[MOTION_FADE_IN]":
                    self._start_replay()
                case "[MOTION_FADE_OUT]":
                    self._stop_replay()
