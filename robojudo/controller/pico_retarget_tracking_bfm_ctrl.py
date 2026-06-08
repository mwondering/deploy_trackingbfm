from __future__ import annotations

import time
from typing import Any

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import PicoRetargetTrackingBfmCtrlCfg
from robojudo.tools.tracking_bfm_sparse_command import (
    DEFAULT_SPARSE_ANCHOR_HEIGHT_W,
    DEFAULT_SPARSE_EE_POSE,
    MujocoRetargetSnapshotBuilder,
    extract_tracking_bfm_sparse_command,
)
from robojudo.tools.tracking_bfm_wbteleop_command import (
    DEFAULT_WBTELEOP_MOTION_BODY_NAMES,
    WbTeleopRetargetCommandExtractor,
)


def _make_real_streamer():
    try:
        from general_motion_retargeting import XRobotStreamer
    except ImportError as exc:
        raise ImportError(
            "general_motion_retargeting.XRobotStreamer not found. "
            "Activate the RoboJuDo environment with GMR and xrobotoolkit_sdk installed."
        ) from exc
    return XRobotStreamer()


def _make_real_retarget(cfg: PicoRetargetTrackingBfmCtrlCfg):
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
    except ImportError as exc:
        raise ImportError(
            "general_motion_retargeting not found. Install GMR in the RoboJuDo environment "
            "before using PicoRetargetTrackingBfmCtrl."
        ) from exc
    return GMR(
        src_human="xrobot",
        tgt_robot=cfg.robot,
        actual_human_height=cfg.actual_human_height,
    )


def _make_real_snapshot_builder(cfg: PicoRetargetTrackingBfmCtrlCfg):
    try:
        import mujoco as mj
        from general_motion_retargeting import ROBOT_XML_DICT
    except ImportError as exc:
        raise ImportError(
            "mujoco and general_motion_retargeting are required for PicoRetargetTrackingBfmCtrl."
        ) from exc

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[cfg.robot]))
    data = mj.MjData(model)
    body_names = list(DEFAULT_WBTELEOP_MOTION_BODY_NAMES)
    for body_name in (cfg.anchor_body_name, cfg.left_ee_body_name, cfg.right_ee_body_name):
        if body_name not in body_names:
            body_names.append(body_name)
    return MujocoRetargetSnapshotBuilder(model, data, tuple(body_names))


@ctrl_registry.register
class PicoRetargetTrackingBfmCtrl(Controller):
    cfg_ctrl: PicoRetargetTrackingBfmCtrlCfg

    def __init__(
        self,
        cfg_ctrl: PicoRetargetTrackingBfmCtrlCfg,
        env=None,
        device="cpu",
        streamer=None,
        retarget=None,
        snapshot_builder=None,
    ):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.streamer = streamer or _make_real_streamer()
        self.retarget = retarget or _make_real_retarget(cfg_ctrl)
        self.snapshot_builder = snapshot_builder or _make_real_snapshot_builder(cfg_ctrl)
        self.wbteleop_extractor = WbTeleopRetargetCommandExtractor(joint_dof=29)
        self.reset()

    def reset(self):
        self.state = "idle"
        self._right_key_prev = False
        self._left_key_prev = False
        self._left_axis_click_prev = False
        self.wbteleop_extractor.reset()
        self._last_output = self._neutral_output([])

    def _neutral_output(self, commands: list[str]) -> dict[str, Any]:
        return {
            "ee_pose": DEFAULT_SPARSE_EE_POSE.copy(),
            "base_lin_vel_b": np.zeros(3, dtype=np.float32),
            "base_ang_vel_b": np.zeros(3, dtype=np.float32),
            "anchor_height_w": np.array([DEFAULT_SPARSE_ANCHOR_HEIGHT_W], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": int(time.time() * 1e9),
            "_commands": list(commands),
        }

    def _button(self, controller_data, controller_name: str, button_name: str) -> bool:
        if not isinstance(controller_data, dict):
            return False
        controller = controller_data.get(controller_name, {})
        if not isinstance(controller, dict):
            return False
        return bool(controller.get(button_name, False))

    def _step_state_machine(self, controller_data) -> list[str]:
        commands: list[str] = []
        right_key = self._button(controller_data, "RightController", "key_one")
        left_key = self._button(controller_data, "LeftController", "key_one")
        left_axis_click = self._button(controller_data, "LeftController", "axis_click")

        right_key_pressed = right_key and not self._right_key_prev
        left_key_pressed = left_key and not self._left_key_prev
        left_axis_click_pressed = left_axis_click and not self._left_axis_click_prev

        self._right_key_prev = right_key
        self._left_key_prev = left_key
        self._left_axis_click_prev = left_axis_click

        if left_key_pressed or left_axis_click_pressed:
            self.state = "exit"
            commands.append("[SHUTDOWN]")
            return commands

        if right_key_pressed:
            if self.state == "idle":
                self.state = "active"
                commands.append("[MOTION_RESET]")
            elif self.state == "active":
                self.state = "pause"
            elif self.state == "pause":
                self.state = "active"

        return commands

    def _active_output(self, smplx_data, timestamp_ns: int, commands: list[str]) -> dict[str, Any]:
        qpos = np.asarray(
            self.retarget.retarget(smplx_data, offset_to_ground=self.cfg_ctrl.offset_to_ground),
            dtype=np.float32,
        ).copy()
        if qpos.shape[0] >= 3:
            qpos[2] += float(self.cfg_ctrl.root_z_offset)

        snapshot = self.snapshot_builder.build(qpos, timestamp_ns=timestamp_ns)
        output = extract_tracking_bfm_sparse_command(
            snapshot,
            anchor_body_name=self.cfg_ctrl.anchor_body_name,
            ee_body_names=(self.cfg_ctrl.left_ee_body_name, self.cfg_ctrl.right_ee_body_name),
            state=self.state,
        )
        try:
            output.update(self.wbteleop_extractor.extract(snapshot, state=self.state))
        except (KeyError, ValueError):
            pass
        output["_commands"] = list(commands)
        return output

    def get_data(self):
        (
            smplx_data,
            _left_hand_data,
            _right_hand_data,
            controller_data,
            _headset_data,
        ) = self.streamer.get_current_frame()
        commands = self._step_state_machine(controller_data)
        timestamp_ns = int(time.time() * 1e9)

        if self.state == "active" and smplx_data is not None:
            self._last_output = self._active_output(smplx_data, timestamp_ns, commands)
        elif self.state == "idle":
            self._last_output = self._neutral_output(commands)
        else:
            self._last_output = dict(self._last_output)
            self._last_output["state"] = self.state
            self._last_output["timestamp_ns"] = timestamp_ns
            self._last_output["_commands"] = list(commands)
        return dict(self._last_output)

    def process_triggers(self, ctrl_data):
        commands = list(ctrl_data.pop("_commands", []))
        return ctrl_data, commands
