from robojudo.config import ASSETS_DIR, Config


class CtrlCfg(Config):
    ctrl_type: str  # name of the controller class

    triggers: dict[str, str] = {}  # trigger conditions
    triggers_extra: dict[str, str] = {}  # extra trigger conditions


class KeyboardCtrlCfg(CtrlCfg):
    ctrl_type: str = "KeyboardCtrl"

    combination_init_buttons: list[str] = ["Key.ctrl_l"]
    """first button in combination, need to be held down to trigger other commands;"""

    triggers: dict[str, str] = {
        "Key.esc": "[SHUTDOWN]",
        # "Key.tab": "[POLICY_TOGGLE]",
        "`": "[SIM_REBORN]",
        "<": "[MOTION_FADE_IN]",  # note: with shift
        ">": "[MOTION_FADE_OUT]",  # note: with shift
        "|": "[MOTION_RESET]",  # note: with shift
        "{": "[MOTION_LOAD_PREV]",  # note: with shift
        "}": "[MOTION_LOAD_NEXT]",  # note: with shift
    }


class KeyboardAmoCtrlCfg(KeyboardCtrlCfg):
    ctrl_type: str = "KeyboardAmoCtrl"

    # Keyboard -> virtual joystick axes for AMO locomotion control.
    axis_key_map: dict[str, tuple[str, str]] = {
        "LeftX": ("a", "d"),
        "LeftY": ("w", "s"),
        "RightX": ("e", "q"),
        "RightY": ("r", "f"),
    }

    # Map keyboard buttons to virtual gamepad button events consumed by AMO.
    button_key_map: dict[str, str] = {
        "Key.space": "Y",
    }


class KeyboardTrackingBfmCtrlCfg(CtrlCfg):
    ctrl_type: str = "KeyboardTrackingBfmCtrl"

    vx_scale: float = 1.0
    vy_scale: float = 0.5
    wz_scale: float = 1.0

    base_height_init: float = 0.793
    base_height_min: float = 0.35
    base_height_max: float = 0.85
    base_height_step: float = 0.02

    ee_pos_step: float = 0.03
    ee_neutral_left: tuple[float, float, float] = (0.20, 0.20, 0.20)
    ee_neutral_right: tuple[float, float, float] = (0.20, -0.20, 0.20)

    triggers: dict[str, str] = {
        "Key.esc": "[SHUTDOWN]",
    }


class JoystickCtrlCfg(CtrlCfg):
    ctrl_type: str = "JoystickCtrl"

    combination_init_buttons: list[str] = ["LB", "RB"]
    """first button in combination, need to be held down to trigger other commands;"""

    # reference for button names in JoystickThread config
    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
        "X": "[MOTION_FADE_IN]",
        "B": "[MOTION_FADE_OUT]",
        "Y": "[MOTION_RESET]",
        # "LB": "[MOTION_LOAD_PREV]",
        # "RB": "[MOTION_LOAD_NEXT]",
        # Note: combo keys supported: "LB+RB+A": "[TEST]",
    }


class UnitreeCtrlCfg(JoystickCtrlCfg):
    ctrl_type: str = "UnitreeCtrl"

    combination_init_buttons: list[str] = ["L1", "R1"]
    """first button in combination, need to be held down to trigger other commands;"""

    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
        "X": "[MOTION_FADE_IN]",
        "B": "[MOTION_FADE_OUT]",
        "Y": "[MOTION_RESET]",
        # Note: combo keys supported: "L1+R1+A": "[TEST]",
    }


class UnitreeAmoCtrlCfg(UnitreeCtrlCfg):
    ctrl_type: str = "UnitreeAmoCtrl"

    # AMO direct command ranges.
    vx_limit: float = 1.0
    vy_limit: float = 0.4
    yaw_rate_limit: float = 0.8
    height_upper_limit: float = 0.03
    height_lower_limit: float = -0.3

    torso_yaw_limit: float = 0.3
    torso_pitch_limit: float = 0.2
    torso_roll_limit: float = 0.2

    torso_yaw_step: float = 0.05
    torso_pitch_step: float = 0.05
    torso_roll_step: float = 0.05

    axis_deadzone: float = 0.05
    enable_height_axis: bool = True

    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
    }


class MotionCtrlCfg(CtrlCfg):
    class PhcCfg(Config):
        robot_config_file: str
        robot_config: dict = {}  # PLACEHOLDER for phc robot config, to be parsed by config manager

        def model_post_init(self, context) -> None:
            import yaml

            from robojudo.config import THIRD_PARTY_DIR

            # parse phc configs
            phc_dir_path = THIRD_PARTY_DIR / "phc"
            phc_robot_config_file = self.robot_config_file
            phc_robot_config_file_path = phc_dir_path / "phc/data/cfg" / phc_robot_config_file
            if phc_robot_config_file_path.exists():
                phc_robot_config_dict = yaml.safe_load(phc_robot_config_file_path.open("r"))
                phc_robot_config_dict["asset"]["assetRoot"] = phc_dir_path.as_posix()
                phc_robot_config_dict["asset"]["assetFileName"] = (
                    phc_dir_path / phc_robot_config_dict["asset"]["assetFileName"]
                ).as_posix()
                # phc_robot_config_dict["asset"]["urdfFileName"] = (
                #     phc_dir_path / phc_robot_config_dict["asset"]["urdfFileName"]
                # ).as_posix()

                self.robot_config = phc_robot_config_dict

    ctrl_type: str = "MotionCtrl"

    motion_ctrl_gui: bool = True

    # ==== policy specific configs ====
    track_keypoints_names: list[str] = []
    phc: PhcCfg

    # ==== motion config ====
    robot: str
    motion_name: str = ""

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/phc/{self.motion_name}.pkl"
        return motion_path.as_posix()


class MotionH2HCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionH2HCtrl"

    extra_motion_data: bool = False  # extra data for motion recognition


class MotionKungfuBotCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionKungfuBotCtrl"

    future_max_steps: int = 95
    future_num_steps: int = 20

    anchor_index: int = 0  # root
    key_body_id: list[int]


class MotionTwistCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionTwistCtrl"

    # ==== motion config ====
    robot: str


class BeyondMimicCtrlCfg(CtrlCfg):
    ctrl_type: str = "BeyondMimicCtrl"

    override_robot_anchor_pos: bool = False  # if True, drop pos fdb

    # ==== motion config ====
    robot: str
    motion_name: str

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/beyondmimic/{self.motion_name}.npz"
        return motion_path.as_posix()

    # ==== from beyondmimic ====
    class MotionCommandCfg(Config):
        """Configuration for the motion command."""

        anchor_body_name: str
        body_names: list[str]
        body_names_all: list[str]
        """from beyondmimic asset, used for indexing"""

    motion_cfg: MotionCommandCfg


class TwistRedisCtrlCfg(CtrlCfg):
    ctrl_type: str = "TwistRedisCtrl"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_key: str = "action_mimic_g1"  # key to get command data from redis

    buffer_size: int = 5  # size of the data buffer to store recent commands


class PicoProcessWorkerCfg(Config):
    queue_size: int = 0
    sleep_s: float = 0.0
    error_sleep_s: float = 0.05
    profile: bool = False
    profile_interval: int = 50
    start_method: str = "spawn"
    stop_timeout_s: float = 1.0


class PicoSourceMonitorCfg(Config):
    enabled: bool = True
    min_update_hz: float = 50.0
    ok_interval: int = 50
    stale_repeat_s: float = 0.5


class CommandInterpolationCfg(Config):
    """Control-rate interpolation of the retargeted reference command.

    Time-based interpolation buffer (see streamed_command_interpolator): buffers
    timestamped reference frames and resamples them to the control rate with
    linear + SLERP interpolation, catch-up reset and dropout hold.
    """

    enabled: bool = True
    target_lag_s: float = 0.033
    """Playback trails the newest frame by this much (~one source period) so the
    cursor sits between two real samples for true interpolation."""
    max_lag_s: float = 0.2
    """If playback falls further behind the newest frame than this, snap the
    cursor forward (catch-up reset) to bound accumulated latency."""
    history_s: float = 0.5
    """How much past data to keep buffered for interpolation/look-back."""
    joint_snap_threshold: float = 1.0
    """Per-joint step (rad) above which interpolation snaps instead of blending (teleport guard)."""


class PicoLightSparseCtrlCfg(CtrlCfg):
    ctrl_type: str = "PicoLightSparseCtrl"

    vx_scale: float = 1.0
    vy_scale: float = 1.0
    wz_scale: float = 1.0

    base_height_init: float = 0.793
    base_height_min: float = 0.15
    base_height_max: float = 0.85
    base_height_rate: float = 0.3

    ee_scale: float = 0.8
    ee_default_left_pos_b: tuple[float, float, float] = (0.09729591, 0.21447651, -0.02440158)
    ee_default_right_pos_b: tuple[float, float, float] = (0.09729591, -0.21446651, -0.02440158)
    ee_default_left_quat_b_xyzw: tuple[float, float, float, float] = (
        0.08970304,
        0.38643646,
        0.04628095,
        0.91677604,
    )
    ee_default_right_quat_b_xyzw: tuple[float, float, float, float] = (
        -0.08970304,
        0.38643646,
        -0.04628095,
        0.91677604,
    )
    retarget_ee_pose: bool = False
    robot: str = "unitree_g1"
    actual_human_height: float = 1.6
    offset_to_ground: bool = True
    root_z_offset: float = 0.0
    anchor_body_name: str = "pelvis"
    left_ee_body_name: str = "left_wrist_yaw_link"
    right_ee_body_name: str = "right_wrist_yaw_link"

    stick_deadzone: float = 0.08
    trigger_deadzone: float = 0.1
    async_read: bool = True
    worker: PicoProcessWorkerCfg = PicoProcessWorkerCfg()


class PicoRetargetTrackingBfmCtrlCfg(CtrlCfg):
    ctrl_type: str = "PicoRetargetTrackingBfmCtrl"

    robot: str = "unitree_g1"
    actual_human_height: float = 1.6
    offset_to_ground: bool = True
    root_z_offset: float = 0.0

    anchor_body_name: str = "pelvis"
    left_ee_body_name: str = "left_wrist_yaw_link"
    right_ee_body_name: str = "right_wrist_yaw_link"
    async_read: bool = True
    worker: PicoProcessWorkerCfg = PicoProcessWorkerCfg()
    source_monitor: PicoSourceMonitorCfg = PicoSourceMonitorCfg()
    command_interpolation: CommandInterpolationCfg = CommandInterpolationCfg()

    triggers: dict[str, str] = {
        "LeftController.key_one": "[SHUTDOWN]",
        "LeftController.axis_click": "[SHUTDOWN]",
    }


class WbTeleopNpzPlaybackCtrlCfg(CtrlCfg):
    ctrl_type: str = "WbTeleopNpzPlaybackCtrl"

    motion_file: str = ""
    motion_type: str = "isaaclab"
    loop: bool = False
    auto_start: bool = False
