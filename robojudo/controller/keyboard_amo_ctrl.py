from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import KeyboardAmoCtrlCfg

try:
    from robojudo.controller.keyboard_ctrl import KeyboardCtrl
except ImportError:  # pragma: no cover - used only in headless test imports
    class KeyboardCtrl(Controller):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("KeyboardCtrl backend is unavailable in this environment")

        def get_data(self):
            raise NotImplementedError

        def reset(self):
            raise NotImplementedError


class KeyboardAmoInputState:
    """Translate keyboard presses into AMO's native 8D command vector."""

    def __init__(
        self, command_key_map: dict[str, tuple[int, float] | str] | None = None
    ):
        self.command_key_map = command_key_map or {
            "w": (0, +0.05),
            "s": (0, -0.05),
            "a": (1, +0.1),
            "d": (1, -0.1),
            "q": (2, +0.05),
            "e": (2, -0.05),
            "z": (3, +0.05),
            "x": (3, -0.05),
            "j": (4, +0.1),
            "u": (4, -0.1),
            "k": (5, +0.05),
            "i": (5, -0.05),
            "l": (6, +0.05),
            "o": (6, -0.1),
        }
        self.reset()

    def reset(self):
        self.commands = [0.0] * 8

    def transform_events(self, keyboard_events: list[dict]) -> dict:
        commands_changed = False
        upper_body_mode_event = None
        for event in keyboard_events:
            if event.get("type") != "keyboard" or not event.get("pressed", False):
                continue

            key_name = event["name"]
            if key_name == "t":
                upper_body_mode_event = "swing"
                print("AMO upper body mode: swing")
                continue
            if key_name == "g":
                upper_body_mode_event = "lock"
                print("AMO upper body mode: lock")
                continue
            mapping = self.command_key_map.get(key_name)
            if mapping is None:
                continue
            command_idx, delta = mapping
            self.commands[command_idx] += delta
            commands_changed = True

        if commands_changed:
            print(self._format_commands())

        return {
            "axes": {},
            "button_event": [],
            "commands": self.commands.copy(),
            "upper_body_mode_event": upper_body_mode_event,
            "keyboard_event": keyboard_events,
        }

    def _format_commands(self) -> str:
        return (
            f"vx: {self.commands[0]:<8.2f}"
            f"vy: {self.commands[2]:<8.2f}"
            f"yaw: {self.commands[1]:<8.2f}"
            f"height: {(0.75 + self.commands[3]):<8.2f}"
            f"torso yaw: {self.commands[4]:<8.2f}"
            f"torso pitch: {self.commands[5]:<8.2f}"
            f"torso roll: {self.commands[6]:<8.2f}"
        )


@ctrl_registry.register
class KeyboardAmoCtrl(KeyboardCtrl):
    cfg_ctrl: KeyboardAmoCtrlCfg

    def __init__(self, cfg_ctrl: KeyboardAmoCtrlCfg, env=None, **kwargs):
        self.input_state = KeyboardAmoInputState()
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)

    def reset(self):
        super().reset()
        self.input_state.reset()

    def get_data(self):
        keyboard_events = self.get_events()
        return self.input_state.transform_events(keyboard_events)
