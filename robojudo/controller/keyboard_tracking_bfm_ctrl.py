from __future__ import annotations

import time
from queue import Empty, Queue

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import KeyboardTrackingBfmCtrlCfg
from robojudo.controller.utils.keyboard import KeyboardThread


def _make_identity_rot6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


@ctrl_registry.register
class KeyboardTrackingBfmCtrl(Controller):
    cfg_ctrl: KeyboardTrackingBfmCtrlCfg

    def __init__(self, cfg_ctrl: KeyboardTrackingBfmCtrlCfg, env=None, device="cpu", keyboard_thread_cls=KeyboardThread):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.event_queue = Queue(maxsize=100)
        self.keyboard_thread = keyboard_thread_cls(self.event_queue) if keyboard_thread_cls is not None else None
        if self.keyboard_thread is not None:
            self.keyboard_thread.start()
        self.reset()

    def reset(self):
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break
        self.state = "idle"
        self._pressed_keys: set[str] = set()
        self._left_offset = np.zeros(3, dtype=np.float32)
        self._right_offset = np.zeros(3, dtype=np.float32)
        self._base_height = float(self.cfg_ctrl.base_height_init)
        self._last_output = self._neutral_output([])

    def _neutral_output(self, commands: list[str]) -> dict:
        return {
            "ee_pose": np.concatenate(
                [
                    np.asarray(self.cfg_ctrl.ee_neutral_left, dtype=np.float32),
                    _make_identity_rot6d(),
                    np.asarray(self.cfg_ctrl.ee_neutral_right, dtype=np.float32),
                    _make_identity_rot6d(),
                ]
            ),
            "base_lin_vel_b": np.zeros(3, dtype=np.float32),
            "base_ang_vel_b": np.zeros(3, dtype=np.float32),
            "anchor_height_w": np.array([self._base_height], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": int(time.time() * 1e9),
            "_commands": list(commands),
        }

    def _get_events(self) -> list[dict]:
        events = []
        while not self.event_queue.empty():
            try:
                events.append(self.event_queue.get_nowait())
            except Empty:
                break
        return events

    def _apply_events(self, events: list[dict]) -> list[str]:
        commands: list[str] = []
        for event in events:
            if event.get("type") != "keyboard":
                continue
            key_name = event["name"]
            pressed = bool(event["pressed"])

            if pressed:
                self._pressed_keys.add(key_name)
            else:
                self._pressed_keys.discard(key_name)

            if not pressed and key_name == "Key.space":
                if self.state == "idle":
                    self.state = "active"
                elif self.state == "active":
                    self.state = "pause"
                elif self.state == "pause":
                    self.state = "active"

            if not pressed and key_name == "Key.enter":
                self._left_offset[:] = 0.0
                self._right_offset[:] = 0.0
                self._base_height = float(self.cfg_ctrl.base_height_init)
                self.state = "active"

            if not pressed:
                command = self.triggers.get(key_name)
                if command is not None:
                    commands.append(command)
                    if command == "[SHUTDOWN]":
                        self.state = "exit"

        return commands

    def _held(self, pos_key: str, neg_key: str) -> float:
        return float(pos_key in self._pressed_keys) - float(neg_key in self._pressed_keys)

    def _active_output(self, commands: list[str]) -> dict:
        base_lin_vel_b = np.array(
            [
                self._held("w", "s") * self.cfg_ctrl.vx_scale,
                self._held("a", "d") * self.cfg_ctrl.vy_scale,
                0.0,
            ],
            dtype=np.float32,
        )
        base_ang_vel_b = np.array([0.0, 0.0, self._held("q", "e") * self.cfg_ctrl.wz_scale], dtype=np.float32)

        self._base_height = float(
            np.clip(
                self._base_height + self._held("r", "f") * self.cfg_ctrl.base_height_step,
                self.cfg_ctrl.base_height_min,
                self.cfg_ctrl.base_height_max,
            )
        )

        left_delta = np.array(
            [
                self._held("t", "g"),
                self._held("y", "h"),
                self._held("u", "j"),
            ],
            dtype=np.float32,
        ) * self.cfg_ctrl.ee_pos_step
        right_delta = np.array(
            [
                self._held("i", "k"),
                self._held("o", "l"),
                self._held("p", ";"),
            ],
            dtype=np.float32,
        ) * self.cfg_ctrl.ee_pos_step

        self._left_offset += left_delta
        self._right_offset += right_delta

        ee_pose = np.concatenate(
            [
                np.asarray(self.cfg_ctrl.ee_neutral_left, dtype=np.float32) + self._left_offset,
                _make_identity_rot6d(),
                np.asarray(self.cfg_ctrl.ee_neutral_right, dtype=np.float32) + self._right_offset,
                _make_identity_rot6d(),
            ]
        ).astype(np.float32)
        return {
            "ee_pose": ee_pose,
            "base_lin_vel_b": base_lin_vel_b,
            "base_ang_vel_b": base_ang_vel_b,
            "anchor_height_w": np.array([self._base_height], dtype=np.float32),
            "state": self.state,
            "timestamp_ns": int(time.time() * 1e9),
            "_commands": list(commands),
        }

    def get_data(self):
        commands = self._apply_events(self._get_events())
        if self.state == "active":
            self._last_output = self._active_output(commands)
        elif self.state == "idle":
            self._last_output = self._neutral_output(commands)
        else:
            self._last_output = dict(self._last_output)
            self._last_output["state"] = self.state
            self._last_output["timestamp_ns"] = int(time.time() * 1e9)
            self._last_output["_commands"] = list(commands)
        return dict(self._last_output)

    def process_triggers(self, ctrl_data):
        commands = list(ctrl_data.pop("_commands", []))
        return ctrl_data, commands
