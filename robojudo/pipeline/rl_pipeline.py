import json
import logging
import time
from collections import defaultdict

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        pd_target = action + self.policy.default_pos
        pd_target = self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)
        if hasattr(self.policy, "override_pd_target"):
            pd_target = self.policy.override_pd_target(pd_target)
        return pd_target

    def get_init_dof_pos(self):
        init_dof_pos = self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)
        if hasattr(self.policy, "override_pd_target"):
            init_dof_pos = self.policy.override_pd_target(init_dof_pos)
        return init_dof_pos

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.hold_policy = None
        if self._is_wbteleop_task:
            self._policy_stiffness = np.asarray(self.env.stiffness, dtype=np.float32).copy()
            self._policy_damping = np.asarray(self.env.damping, dtype=np.float32).copy()
            self._policy_torque_limits = np.asarray(self.env.torque_limits, dtype=np.float32).copy()
            cfg_hold_policy = getattr(self.cfg, "hold_policy", None)
            if cfg_hold_policy is not None:
                self.hold_policy = PolicyWrapper(
                    cfg_policy=cfg_hold_policy,
                    env_dof_cfg=self.env.dof_cfg,
                    device=self.device,
                )
                self._hold_stiffness = self._fit_hold_dof_property("stiffness", self._policy_stiffness)
                self._hold_damping = self._fit_hold_dof_property("damping", self._policy_damping)
                self._hold_torque_limits = self._fit_hold_dof_property("torque_limits", self._policy_torque_limits)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq
        self._profile_sums = defaultdict(float)
        self._profile_count = 0
        self._default_pose_mode_enabled = False
        self._hold_to_policy_blend_start = None
        self._hold_to_policy_blend_step = 0
        self._hold_to_policy_blend_steps = 0
        self._last_pd_target = None

        self.reset()
        self.self_check()
        self.policy.reset()  # reset frame counter after dry-run steps
        if self.hold_policy is not None:
            self.hold_policy.reset()

    def _profile_enabled(self):
        return bool(getattr(self.cfg.debug, "profile_timing", False))

    def _profile_reset(self):
        self._profile_sums.clear()
        self._profile_count = 0

    def _profile_add(self, timings: dict[str, float]):
        if not self._profile_enabled():
            return
        for key, value in timings.items():
            self._profile_sums[key] += value
        self._profile_count += 1
        interval = max(1, int(getattr(self.cfg.debug, "profile_interval", 50)))
        if self._profile_count < interval:
            return

        total = sum(self._profile_sums.values())
        parts = [f"{key}={value / self._profile_count * 1000:.2f}ms" for key, value in self._profile_sums.items()]
        logger.warning(
            "Timing profile avg over %d frames: total=%.2fms, %s",
            self._profile_count,
            total / self._profile_count * 1000,
            ", ".join(parts),
        )
        self._profile_reset()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def _inner_policy(self):
        """Return the unwrapped inner Policy (e.g. ProtoMotionsTrackerPolicy)."""
        return getattr(self.policy, "policy", self.policy)

    @property
    def _has_default_pose_mode(self) -> bool:
        return hasattr(self._inner_policy(), "set_default_pose_mode")

    def _set_default_pose_mode(self, enabled: bool):
        """Enable/disable default-pose mode on the inner policy (if supported)."""
        self._default_pose_mode_enabled = bool(enabled)
        if enabled:
            self._hold_to_policy_blend_start = None
            self._hold_to_policy_blend_step = 0
            self._hold_to_policy_blend_steps = 0
        inner = self._inner_policy()
        if hasattr(inner, "set_default_pose_mode"):
            inner.set_default_pose_mode(enabled)
        self._set_default_pose_hold_gains(enabled)

    def _fit_hold_dof_property(self, prop_name: str, template: np.ndarray) -> np.ndarray:
        value = getattr(self.hold_policy.cfg_action_dof, prop_name)
        template = np.asarray(template, dtype=np.float32)
        if value is None:
            return template.copy()
        return self.hold_policy.actions_adapter.fit(value, template=template).astype(np.float32)

    @property
    def _is_wbteleop_sim2sim(self) -> bool:
        return (
            self.cfg.__class__.__name__ == "g1_wbteleop_sim2sim"
            and bool(getattr(self.cfg.env, "is_sim", False))
        )

    @property
    def _is_wbteleop_task(self) -> bool:
        return self.cfg.__class__.__name__ in {"g1_wbteleop_sim2sim", "g1_wbteleop_real"}

    def _set_wbteleop_sim2sim_default_qpos(self):
        if not self._is_wbteleop_sim2sim:
            return
        if not all(hasattr(self.env, name) for name in ("data", "model")):
            return

        default_pos = np.asarray(self.policy.default_pos, dtype=np.float64)
        if default_pos.shape[0] != self.env.num_dofs:
            raise ValueError(
                f"wbteleop sim2sim default pose length mismatch: default={default_pos.shape[0]}, "
                f"env={self.env.num_dofs}"
            )
        default_base_height = getattr(self.cfg, "wbteleop_default_base_height", None)
        if default_base_height is not None:
            self.env.data.qpos[2] = float(default_base_height)  # pyright: ignore[reportAttributeAccessIssue]
        self.env.data.qpos[-self.env.num_dofs :] = default_pos  # pyright: ignore[reportAttributeAccessIssue]
        self.env.data.qvel[:] = 0.0  # pyright: ignore[reportAttributeAccessIssue]
        self.env.data.ctrl[:] = 0.0  # pyright: ignore[reportAttributeAccessIssue]

        import mujoco

        mujoco.mj_forward(self.env.model, self.env.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.env.update()

    def _set_default_pose_hold_gains(self, enabled: bool):
        if not self._is_wbteleop_task:
            return

        if enabled and self.hold_policy is not None:
            stiffness = self._hold_stiffness
            damping = self._hold_damping
            torque_limits = self._hold_torque_limits
        elif enabled:
            hold_dof = self.cfg.env.dof
            stiffness = np.asarray(hold_dof.stiffness, dtype=np.float32)
            damping = np.asarray(hold_dof.damping, dtype=np.float32)
            torque_limits = np.asarray(hold_dof.torque_limits, dtype=np.float32)
        else:
            stiffness = self._policy_stiffness
            damping = self._policy_damping
            torque_limits = self._policy_torque_limits

        if stiffness.shape[0] != self.env.num_dofs or damping.shape[0] != self.env.num_dofs:
            raise ValueError(
                f"default-pose hold gain length mismatch: stiffness={stiffness.shape[0]}, "
                f"damping={damping.shape[0]}, env={self.env.num_dofs}"
            )
        if torque_limits.shape[0] != self.env.num_dofs:
            raise ValueError(
                f"default-pose hold torque limit length mismatch: torque_limits={torque_limits.shape[0]}, "
                f"env={self.env.num_dofs}"
            )

        self.env.set_gains(stiffness, damping)
        self.env.stiffness = stiffness
        self.env.damping = damping
        self.env.torque_limits = torque_limits

    @property
    def _use_wbteleop_hold_policy(self) -> bool:
        return self._is_wbteleop_task and self._default_pose_mode_enabled and self.hold_policy is not None

    def _policy_for_step(self):
        return self.hold_policy if self._use_wbteleop_hold_policy else self.policy

    def _default_pose_prepare_ctrl_data(self):
        ctrl_data = {"COMMANDS": []}
        ctrl_type = getattr(self.policy, "ctrl_type", None)
        if ctrl_type is not None:
            ctrl_data[ctrl_type] = {"state": "idle"}
        return Box(ctrl_data)

    def _prepare_ctrl_data(self, env_data):
        if self._has_default_pose_mode:
            return self._default_pose_prepare_ctrl_data()
        return self.ctrl_manager.get_ctrl_data(env_data)

    def _start_hold_to_policy_blend(self):
        if not self._use_wbteleop_hold_policy:
            return
        seconds = float(getattr(self.cfg, "hold_to_policy_blend_seconds", 0.0) or 0.0)
        steps = int(seconds * self.freq)
        if steps <= 0:
            return
        if self._last_pd_target is None:
            self._hold_to_policy_blend_start = np.asarray(self.env.dof_pos, dtype=np.float32)
        else:
            self._hold_to_policy_blend_start = np.asarray(self._last_pd_target, dtype=np.float32).copy()
        self._hold_to_policy_blend_step = 0
        self._hold_to_policy_blend_steps = steps

    @property
    def _hold_to_policy_blend_active(self) -> bool:
        return (
            self._hold_to_policy_blend_start is not None
            and self._hold_to_policy_blend_step < self._hold_to_policy_blend_steps
        )

    def _apply_hold_to_policy_blend(self, pd_target):
        if not self._hold_to_policy_blend_active:
            return pd_target
        alpha = (self._hold_to_policy_blend_step + 1) / max(self._hold_to_policy_blend_steps, 1)
        blended = (1 - alpha) * self._hold_to_policy_blend_start + alpha * pd_target
        self._hold_to_policy_blend_step += 1
        if self._hold_to_policy_blend_step >= self._hold_to_policy_blend_steps:
            self._hold_to_policy_blend_start = None
        return blended

    @staticmethod
    def _json_ready_debug_value(value):
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                return np.round(value.astype(np.float64), 6).tolist()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): RlPipeline._json_ready_debug_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [RlPipeline._json_ready_debug_value(val) for val in value]
        return value

    def _maybe_log_wbteleop_proprio_debug(self, env_data):
        debug_cfg = getattr(self.cfg, "debug", None)
        if not bool(getattr(debug_cfg, "wbteleop_proprio_debug", False)):
            return

        interval = max(1, int(getattr(debug_cfg, "wbteleop_proprio_debug_interval", 50) or 50))
        if self.timestep <= 0 or self.timestep % interval != 0:
            return

        if not hasattr(self.policy, "get_proprio_debug_terms"):
            return
        try:
            payload = self.policy.get_proprio_debug_terms(env_data)
        except Exception as exc:
            logger.warning("Failed to collect wbteleop proprio debug terms: %s", exc)
            return

        logger.warning(
            "WBTELEOP_PROPRIO_OBS frame=%d\n%s",
            self.timestep,
            json.dumps(self._json_ready_debug_value(payload), indent=2, sort_keys=True),
        )

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        self.policy.reset()
        if self.hold_policy is not None:
            self.hold_policy.reset()
        self.ctrl_manager.reset()
        self._set_wbteleop_sim2sim_default_qpos()
        if self._is_wbteleop_sim2sim and self._has_default_pose_mode:
            self._set_default_pose_mode(True)

        # Blend-out state: transitions policy → init pose at end of motion.
        self._blend_out_active = False
        self._blend_out_step = 0
        self._blend_out_duration = int(5.0 * self.freq)  # 5 seconds

        # For tracker policies with default-pose mode, ramp/blend target is
        # the env's default standing pose.  Otherwise, use motion frame 0.
        if self._has_default_pose_mode:
            self._init_dof_pos = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32)
        else:
            self._init_dof_pos = np.asarray(self.policy.get_init_dof_pos(), dtype=np.float32)

        self._pending_blend_in = False
        self._blend_in_completed = False
        self._user_fade_out = False  # True when fade-out was user-triggered (not auto)
        self._prepare_seconds = None  # set by prepare() for re-use on reset

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                self.policy.reset_alignment()
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                        self.policy.reset_alignment()
                case "[MOTION_RESET]" | "[MOTION_FADE_IN]":
                    self._blend_out_active = False
                    self._blend_out_step = 0
                    self._user_fade_out = False
                    if self._has_default_pose_mode:
                        # Policy is already active — just switch target
                        # from default pose to motion (instant, no blend).
                        logger.info(
                            f"{command} — starting motion from frame 0"
                        )
                        self._start_hold_to_policy_blend()
                        self._set_default_pose_mode(False)
                    else:
                        # Legacy path: full blend-in needed.
                        logger.info(
                            f"{command} — re-entering blend-in phase"
                        )
                        self._pending_blend_in = True
                case "[MOTION_FADE_OUT]":
                    if self._has_default_pose_mode:
                        logger.info("Fade out — switching to default pose mode")
                        self._set_default_pose_mode(True)
                        self._user_fade_out = True
                    elif not self._blend_out_active:
                        logger.info("Fade out — blending to default pose")
                        self._blend_out_active = True
                        self._blend_out_step = 0
                        self._user_fade_out = True
                        inner = self._inner_policy()
                        if hasattr(inner, "_paused"):
                            inner._paused = True

        self.ctrl_manager.post_step_callback(ctrl_data)

        active_policy = getattr(self, "_active_policy_for_callback", self.policy)
        active_policy.post_step_callback(commands)
        if self.visualizer is not None:
            active_policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )
        self._maybe_log_wbteleop_proprio_debug(env_data)

    def step(self, dry_run=False):
        timings = {}
        t_last = time.perf_counter()
        self.env.update()
        t_now = time.perf_counter()
        timings["env_update"] = t_now - t_last
        t_last = t_now
        env_data = self.env.get_data()
        t_now = time.perf_counter()
        timings["env_get_data"] = t_now - t_last
        t_last = t_now

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        t_now = time.perf_counter()
        timings["ctrl"] = t_now - t_last
        t_last = t_now

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        active_policy = self._policy_for_step()
        self._active_policy_for_callback = active_policy
        obs, extras = active_policy.get_observation(env_data, ctrl_data)
        t_now = time.perf_counter()
        timings["policy_obs"] = t_now - t_last
        t_last = t_now
        pd_target = active_policy.get_pd_target(obs)
        t_now = time.perf_counter()
        timings["policy_action"] = t_now - t_last
        t_last = t_now

        # -- Detect motion done --
        callbacks = extras.get("CALLBACK", [])
        if "[MOTION_DONE]" in callbacks and not self._blend_out_active:
            if self._has_default_pose_mode:
                logger.info("Motion done — switching to default pose mode")
                self._set_default_pose_mode(True)
            else:
                logger.info("Motion done — blending out to default pose")
                self._blend_out_active = True
                self._blend_out_step = 0

        # -- Blend policy output → init pose (frame 0) --
        if self._blend_out_active:
            alpha = min(self._blend_out_step / max(self._blend_out_duration, 1), 1.0)
            pd_target = (1 - alpha) * pd_target + alpha * self._init_dof_pos
            self._blend_out_step += 1

        pd_target = self._apply_hold_to_policy_blend(pd_target)
        self._last_pd_target = np.asarray(pd_target, dtype=np.float32).copy()

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
        t_now = time.perf_counter()
        timings["env_step"] = t_now - t_last
        t_last = t_now

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)
        t_now = time.perf_counter()
        timings["post_step"] = t_now - t_last
        self._profile_add(timings)

        # Handle pending blend-in (after MOTION_RESET / FADE_IN).
        if self._pending_blend_in:
            self._pending_blend_in = False
            self._run_blend_in()
            self._blend_in_completed = True

    def _run_blend_in(self):
        """Phase 2 of prepare: blend from default pose to policy output.

        Policy runs in default-pose mode (if supported); frame stays at 0
        (no post_step_callback).  After blend, switches to motion tracking.
        """
        secs = self._prepare_seconds or 3.0
        blend_steps = int(secs * self.freq)

        # Enter default-pose mode for the blend-in period.
        self._set_default_pose_mode(True)

        logger.warning(f"Blend-in: default DOF → policy ({blend_steps} steps, {secs:.1f}s)")
        pbar = ProgressBar("Blend in", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self._prepare_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            action = (1 - alpha) * self._init_dof_pos + alpha * policy_pd

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Switch to motion tracking — policy sees the jump.
        self._set_default_pose_mode(False)
        logger.warning("Blend-in done — motion starting")

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        elif self._has_default_pose_mode:
            # Ramp to the env's default standing pose (not motion frame 0).
            desired_motor_angle = np.array(
                self.env.dof_cfg.default_pos, dtype=np.float32
            )
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        # Convert seconds to steps (at policy frequency).
        # Default: 3s ramp + 5s blend.  CLI --prepare-seconds overrides both.
        if prepare_seconds is not None:
            ramp_steps = int(prepare_seconds * self.freq)
            blend_steps = int(prepare_seconds * self.freq)
        else:
            ramp_steps = int(3.0 * self.freq)
            blend_steps = int(5.0 * self.freq)

        # ── Phase 1: Ramp joints to default pose ──
        logger.warning(
            f"prepare: phase 1 — ramp joints ({ramp_steps} steps, "
            f"{ramp_steps / self.freq:.1f}s)"
        )
        pbar = ProgressBar("Prepare: ramp joints", ramp_steps)

        last_step_time = time.time()
        for t in range(ramp_steps):
            current_motor_angle = np.array(self.env.dof_pos)
            alpha = min(t / max(ramp_steps - 1, 1), 1.0)
            action = (1 - alpha) * current_motor_angle + alpha * desired_motor_angle

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Reset policy for a clean start — frame goes back to 0.
        self.reset()
        self._prepare_seconds = prepare_seconds  # restore after reset

        # ── Phase 2: Blend in policy (holding default pose) ──
        # Policy runs in default-pose mode (if supported): it sees synthetic
        # references for the standing pose, not the real motion.
        # Actions blend from raw default DOF to policy output.
        self._set_default_pose_mode(True)

        logger.warning(
            f"prepare: phase 2 — blend policy ({blend_steps} steps, "
            f"{blend_steps / self.freq:.1f}s)"
        )
        pbar = ProgressBar("Prepare: blend policy", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            timings = {}
            t_last = time.perf_counter()
            alpha = t / max(blend_steps - 1, 1)

            # Run policy observation + action (frame stays at 0).
            self.env.update()
            t_now = time.perf_counter()
            timings["prepare_env_update"] = t_now - t_last
            t_last = t_now
            env_data = self.env.get_data()
            t_now = time.perf_counter()
            timings["prepare_env_get_data"] = t_now - t_last
            t_last = t_now
            ctrl_data = self._prepare_ctrl_data(env_data)
            t_now = time.perf_counter()
            timings["prepare_ctrl"] = t_now - t_last
            t_last = t_now
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            t_now = time.perf_counter()
            timings["prepare_policy_obs"] = t_now - t_last
            t_last = t_now
            policy_pd = self.policy.get_pd_target(obs)
            t_now = time.perf_counter()
            timings["prepare_policy_action"] = t_now - t_last
            t_last = t_now

            # Blend: default DOF → policy output
            action = (1 - alpha) * desired_motor_angle + alpha * policy_pd

            self.env.step(action)
            t_now = time.perf_counter()
            timings["prepare_env_step"] = t_now - t_last
            self._profile_add(timings)

            # Do NOT call post_step_callback — frame stays at 0.

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # ── Phase 3: Hold default pose — wait for R to start motion ──
        # Stay in default-pose mode. Motion starts when [MOTION_RESET] is
        # received (user presses R), which calls _set_default_pose_mode(False).
        logger.warning("prepare done — holding default pose, press R to start motion")


if __name__ == "__main__":
    pass
