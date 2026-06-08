import logging
import time

import numpy as np
from unitree_cpp import RobotState, SportState, UnitreeController  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.tools.retarget import HandRetarget
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeCppEnv(Environment):
    cfg_env: UnitreeEnvCfg

    def __init__(self, cfg_env: UnitreeEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        self._last_pd_target_cmd: np.ndarray | None = None
        super().__init__(cfg_env=cfg_env, device=device)
        self.RemoteControllerHandler = None
        self.limit_pd_target_effort = cfg_env.limit_pd_target_effort
        self.clip_pd_target = cfg_env.clip_pd_target
        self.pd_target_max_delta = cfg_env.pd_target_max_delta

        cfg_unitree: UnitreeEnvCfg.UnitreeCfg = cfg_env.unitree

        cfg_unitree_dict: dict = cfg_unitree.to_dict()
        cfg_unitree_dict["num_dofs"] = self.num_dofs
        cfg_unitree_dict["stiffness"] = self.stiffness
        cfg_unitree_dict["damping"] = self.damping

        self.robot = cfg_unitree.robot
        self._dof_idx = cfg_env.joint2motor_idx
        self._odometry_type = cfg_env.odometry_type
        if self._odometry_type == "ZED":
            assert self.cfg_env.zed_cfg is not None, "zed_cfg must be set if odometry_type is 'ZED'"
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)
        elif self._odometry_type == "DUMMY":
            pass
        elif self._odometry_type == "UNITREE":
            pass

        self.hand_type = cfg_unitree.hand_type
        if self.hand_type == "Inspire":
            self.hand_retarget = HandRetarget(cfg_env.hand_retarget)
        elif self.hand_type == "Dex-3":
            self.hand_retarget = None  # TODO
        else:
            self.hand_retarget = None

        self.sport_state: SportState = None  # pyright: ignore[reportAttributeAccessIssue]
        self.robot_state: RobotState = None  # pyright: ignore[reportAttributeAccessIssue]

        self.unitree = UnitreeController(cfg_unitree_dict)

        # born place alignment extra for h1 torso
        if self.robot == "h1":
            self.torso_align = TransformAlignment()

        # time.sleep(1)  # wait for unitree init
        self.self_check()

    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                logger.info("UnitreeCppEnv self check passed!")
                break
        if not self.unitree.self_check():
            logger.critical("UnitreeCppEnv self check failed!")
            exit()

    def reset(self):
        self._last_pd_target_cmd = None
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

        if self.robot == "h1":
            self.torso_align.set_base(quat=self.torso_quat)

        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def update(self):
        # robot state
        self.robot_state = self.unitree.get_robot_state()
        if self._dof_idx is None:
            self._dof_pos = np.array(self.robot_state.motor_state.q, dtype=np.float32)
            self._dof_vel = np.array(self.robot_state.motor_state.dq, dtype=np.float32)
        else:
            self._dof_pos = np.array(
                [self.robot_state.motor_state.q[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )
            self._dof_vel = np.array(
                [self.robot_state.motor_state.dq[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )

        if self.robot == "g1":
            quat = np.array(self.robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)
            rpy = np.array(self.robot_state.imu_state.rpy, dtype=np.float32)

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = rpy

        elif self.robot == "h1":
            raise NotImplementedError("H1 robot with unitree_cpp not supported yet.")

        # odometry
        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                # born place aligned in zed_odometry
                self._base_pos = self.zed_odometry.pos
                self._base_lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.array([0.0, 0.0, 0.8])
            self._base_lin_vel = np.array([0.0, 0.0, 0.0])
        elif self._odometry_type == "UNITREE":
            self.sport_state = self.unitree.get_sport_state()
            base_pos = np.asarray(self.sport_state.position, dtype=np.float32)
            lin_vel = np.asarray(self.sport_state.velocity, dtype=np.float32)
            self._base_lin_vel = quat_rotate_inverse_np(self.base_quat, lin_vel)
            self._base_pos = self.base_align.align_pos(base_pos) if self.born_place_align else base_pos

        # FK
        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # controller
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        positions = self._filter_pd_target(pd_target)
        if self.enabled:
            self.unitree.step(positions.tolist())

        if hand_pose is not None:
            assert type(hand_pose) is np.ndarray, "hand_pose should be a numpy array"
            assert hand_pose.shape[0] == 2, "hand_pose should be of shape (2, -1)"
            if self.hand_retarget is not None:
                hand_pose = self.hand_retarget.from_pose_to_cmd(hand_pose)
                logger.debug(f"Hand pose retargeted: {hand_pose}")
            hand_pose = hand_pose.tolist()

            if self.enabled:
                self.unitree.step_hands(hand_pose[0], hand_pose[1])

    def _filter_pd_target(self, pd_target) -> np.ndarray:
        target = np.asarray(pd_target, dtype=np.float32).reshape(-1).copy()
        assert target.shape[0] == self.num_dofs, "pd_target len should be num_dofs of env"

        if not np.all(np.isfinite(target)):
            logger.error("Non-finite pd_target detected; replacing with current joint positions")
            target = self.dof_pos.astype(np.float32)

        if getattr(self, "clip_pd_target", False):
            limits = np.asarray(self.position_limits, dtype=np.float32)
            clipped = np.clip(target, limits[:, 0], limits[:, 1])
            if np.any(clipped != target):
                max_violation = float(np.max(np.abs(target - clipped)))
                logger.warning(f"PD target clipped to joint limits; max violation={max_violation:.4f} rad")
            target = clipped

        max_delta_raw = getattr(self, "pd_target_max_delta", None)
        if max_delta_raw is not None:
            max_delta = np.asarray(max_delta_raw, dtype=np.float32)
            if max_delta.ndim == 0:
                max_delta = np.full(self.num_dofs, float(max_delta), dtype=np.float32)
            assert max_delta.shape[0] == self.num_dofs, "pd_target_max_delta len should be num_dofs of env"

            reference = self._last_pd_target_cmd if self._last_pd_target_cmd is not None else self.dof_pos
            reference = np.asarray(reference, dtype=np.float32)
            limited = np.clip(target, reference - max_delta, reference + max_delta)
            if np.any(limited != target):
                max_limited = float(np.max(np.abs(target - reference)))
                logger.warning(f"PD target delta limited; max requested step={max_limited:.4f} rad")
            target = limited

        if getattr(self, "limit_pd_target_effort", True):
            target = self._limit_pd_target_effort(target)

        self._last_pd_target_cmd = target.copy()
        return target

    def _limit_pd_target_effort(self, target: np.ndarray) -> np.ndarray:
        stiffness = np.asarray(self.stiffness, dtype=np.float32)
        damping = np.asarray(self.damping, dtype=np.float32)
        effort = np.asarray(self.torque_limits, dtype=np.float32)
        pos = np.asarray(self.dof_pos, dtype=np.float32)
        vel = np.asarray(self.dof_vel, dtype=np.float32)

        valid = (stiffness > 0.0) & np.isfinite(stiffness) & np.isfinite(damping) & np.isfinite(effort)
        if not np.all(valid):
            logger.error("Invalid PD effort limit inputs; leaving affected pd_target joints unclamped")

        lower = np.full(self.num_dofs, -np.inf, dtype=np.float32)
        upper = np.full(self.num_dofs, np.inf, dtype=np.float32)
        lower[valid] = pos[valid] + (-effort[valid] + damping[valid] * vel[valid]) / stiffness[valid]
        upper[valid] = pos[valid] + (effort[valid] + damping[valid] * vel[valid]) / stiffness[valid]

        limited = np.clip(target, lower, upper)
        if np.any(limited != target):
            requested_effort = stiffness * (target - pos) - damping * vel
            excess = np.maximum(np.abs(requested_effort) - effort, 0.0)
            max_excess = float(np.max(excess[valid])) if np.any(valid) else 0.0
            logger.warning(f"PD target effort limited to training actuator range; max excess={max_excess:.4f} Nm")
        return limited.astype(np.float32, copy=False)

    def shutdown(self):
        # self.set_damping_mode()
        self.enabled = False
        self.unitree.shutdown()

    def set_gains(self, stiffness, damping):
        if not hasattr(self, "unitree"):  # TODO
            return
        if not self.enabled:
            return
        self.unitree.set_gains(stiffness, damping)


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg

    env = UnitreeCppEnv(cfg_env=G1RealEnvCfg())
    env.set_gains(
        stiffness=[kp * 0.0 for kp in env.stiffness],
        damping=[kd * 0.1 for kd in env.damping],
    )
    while 1:
        # env.step(np.zeros(29), np.ones((2, 7)) * -0)
        env.step(np.zeros(29), None)
        # if controller.remote_controller("A"):
        #     controller.shutdown()
        print(env.base_rpy)
        print(env.dof_pos)
        print(env.base_pos)
        env.update()
        # print(env.base_pos)
        time.sleep(0.1)
    # print("Exit")
    # from robojudo.controller import UnitreeCtrl
    # ctrl = UnitreeCtrl(env=env)

    # while True:
    #     env.update()
    #     state = ctrl.get_state()
    #     events = ctrl.get_events()
    #     print("State:", state)
    #     print("Events:", events)
    #     time.sleep(0.1)  # Simulate a control loop
