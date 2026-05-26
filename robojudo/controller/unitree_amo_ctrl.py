import logging

import numpy as np

from robojudo.controller import ctrl_registry
from robojudo.controller.ctrl_cfgs import UnitreeAmoCtrlCfg
from robojudo.controller.unitree_ctrl import UnitreeCtrl

logger = logging.getLogger(__name__)


class UnitreeAmoInputState:
    """Translate Unitree remote input into AMO's native 8D command vector."""

    def __init__(self, cfg_ctrl: UnitreeAmoCtrlCfg):
        self.cfg_ctrl = cfg_ctrl
        self.reset()

    def reset(self):
        self.torso_yaw = 0.0
        self.torso_pitch = 0.0
        self.torso_roll = 0.0

    def transform(self, axes: dict[str, float], button_events: list[dict]) -> dict:
        upper_body_mode_event = None
        stand_still_event = None

        for event in button_events:
            if event.get("type") != "button" or not event.get("pressed", False):
                continue

            name = event.get("name")
            match name:
                case "Left":
                    self.torso_yaw -= self.cfg_ctrl.torso_yaw_step
                case "Right":
                    self.torso_yaw += self.cfg_ctrl.torso_yaw_step
                case "Up":
                    self.torso_pitch += self.cfg_ctrl.torso_pitch_step
                case "Down":
                    self.torso_pitch -= self.cfg_ctrl.torso_pitch_step
                case "L1":
                    self.torso_roll -= self.cfg_ctrl.torso_roll_step
                case "R1":
                    self.torso_roll += self.cfg_ctrl.torso_roll_step
                case "X":
                    upper_body_mode_event = "swing"
                    logger.info("AMO upper body mode: swing")
                case "B":
                    upper_body_mode_event = "lock"
                    logger.info("AMO upper body mode: lock")
                case "Y":
                    stand_still_event = "toggle"
                    logger.info("AMO stand-still lock: toggle")

        self.torso_yaw = self._clip(self.torso_yaw, self.cfg_ctrl.torso_yaw_limit)
        self.torso_pitch = self._clip(self.torso_pitch, self.cfg_ctrl.torso_pitch_limit)
        self.torso_roll = self._clip(self.torso_roll, self.cfg_ctrl.torso_roll_limit)

        commands = np.zeros(8, dtype=np.float32)
        commands[0] = self._axis(axes.get("LeftY", 0.0)) * self.cfg_ctrl.vx_limit
        commands[2] = -self._axis(axes.get("LeftX", 0.0)) * self.cfg_ctrl.vy_limit
        commands[1] = -self._axis(axes.get("RightX", 0.0)) * self.cfg_ctrl.yaw_rate_limit
        if self.cfg_ctrl.enable_height_axis:
            commands[3] = self._height_axis(axes.get("RightY", 0.0))
        commands[4] = self.torso_yaw
        commands[5] = self.torso_pitch
        commands[6] = self.torso_roll

        return {
            "axes": axes,
            "button_event": button_events,
            "commands": commands.tolist(),
            "yaw_command_mode": "velocity",
            "upper_body_mode_event": upper_body_mode_event,
            "stand_still_event": stand_still_event,
        }

    def _axis(self, value: float) -> float:
        value = float(value)
        if abs(value) < self.cfg_ctrl.axis_deadzone:
            return 0.0
        return self._clip(value, 1.0)

    def _height_axis(self, value: float) -> float:
        value = self._axis(value)
        if value >= 0.0:
            return value * self.cfg_ctrl.height_upper_limit
        return -value * self.cfg_ctrl.height_lower_limit

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return float(np.clip(value, -limit, limit))


@ctrl_registry.register
class UnitreeAmoCtrl(UnitreeCtrl):
    cfg_ctrl: UnitreeAmoCtrlCfg

    def __init__(self, cfg_ctrl: UnitreeAmoCtrlCfg, env=None, device="cpu"):
        self.input_state = UnitreeAmoInputState(cfg_ctrl)
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

    def reset(self):
        super().reset()
        self.input_state.reset()

    def get_data(self):
        state = self.get_state()
        events = self.get_events()
        return self.input_state.transform(state["axes"], events)
