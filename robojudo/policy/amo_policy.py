from collections import deque

import numpy as np
import torch

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import AMOPolicyCfg
from robojudo.utils.util_func import command_remap, quatToEuler


@policy_registry.register
class AMOPolicy(Policy):
    """
    Only Unitree G1 supported now.
    """

    AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS = np.array(
        [0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3], dtype=np.float32
    )
    G1_29DOF_UPPER_BODY_ENV_INDICES = np.array([15, 16, 17, 18, 22, 23, 24, 25], dtype=np.int64)

    cfg_policy: AMOPolicyCfg

    def __init__(self, cfg_policy, device):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(cfg_policy=cfg_policy, device=device)

        # if cpu device:
        if self.device == "cpu":
            for node in self.model.graph.nodes():  # type: ignore
                if node.kind() == "prim::Constant":
                    node: torch.Node
                    if "value" in node.attributeNames():
                        value = node.output().toIValue()
                        if isinstance(value, torch.device):
                            node.removeAttribute("value")
                            node.s_("value", "cpu")

        self.scales_ang_vel = self.cfg_policy.obs_scales.ang_vel
        self.scales_dof_vel = self.cfg_policy.obs_scales.dof_vel
        self.control_dt = 1.0 / self.cfg_policy.freq
        self.commands_map = self.cfg_policy.commands_map

        # TODO: move robot specific to cfg
        self.nj = 23
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8
        self._upper_body_env_indices = self.G1_29DOF_UPPER_BODY_ENV_INDICES.copy()

        self.arm_action = self.default_dof_pos[-self._n_demo_dof :]
        self.upper_body_default_pos = self.AMO_ORIGINAL_UPPER_BODY_DEFAULT_POS.copy()
        self._upper_body_keyframes = np.stack(
            [
                self.upper_body_default_pos,
                self.upper_body_default_pos
                + np.array([-1.88, 0.18, -0.10, 0.45, -1.40, -0.04, 0.06, 0.18], dtype=np.float32),
                self.upper_body_default_pos
                + np.array([-1.88, -0.10, 0.18, 0.56, -1.40, 0.06, -0.05, 0.16], dtype=np.float32),
                self.upper_body_default_pos
                + np.array([-1.88, 0.22, -0.22, 0.30, -1.40, -0.08, 0.10, 0.12], dtype=np.float32),
            ],
            axis=0,
        )

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[: self._n_demo_dof] = self.upper_body_default_pos
        self.demo_obs_template[self._n_demo_dof + 6 : self._n_demo_dof + 9] = 0.75

        self.target_yaw = 0.0
        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        self.cmd = np.zeros(8, dtype=np.float32)
        self.cmd[0] = self.commands_map[0][1]
        self.cmd[1] = self.commands_map[1][1]
        self.cmd[2] = self.commands_map[2][1]
        self.cmd[3] = 0.0

        self.adapter = torch.jit.load(self.cfg_policy.policy_adapter_file, map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats = torch.load(self.cfg_policy.policy_adapter_norm_file, weights_only=False)
        self.input_mean = torch.tensor(norm_stats["input_mean"], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats["input_std"], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats["output_mean"], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats["output_std"], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        self.reset()

    def reset(self):
        self.timestep: int = 0
        self._upper_body_target = self.upper_body_default_pos.copy()
        self._upper_body_mode = "locked"
        self._upper_body_next_mode = None
        self._upper_body_transition_start = self.upper_body_default_pos.copy()
        self._upper_body_transition_end = self.upper_body_default_pos.copy()
        self._upper_body_transition_step = 0
        self._upper_body_transition_steps = max(1, int(1.0 / self.control_dt))
        self._upper_body_swing_step = 0
        self._upper_body_swing_period_steps = max(1, int(4.8 / self.control_dt))

    def post_step_callback(self, commands=None):
        self.timestep += 1
        if self._upper_body_mode == "swing":
            self._upper_body_swing_step = (self._upper_body_swing_step + 1) % self._upper_body_swing_period_steps
        elif self._upper_body_mode == "transition":
            self._upper_body_transition_step += 1
            if self._upper_body_transition_step >= self._upper_body_transition_steps:
                next_mode = self._upper_body_next_mode or "locked"
                self._upper_body_mode = next_mode
                self._upper_body_next_mode = None
                self._upper_body_target = self.upper_body_default_pos.copy()
                if next_mode == "swing":
                    self._upper_body_swing_step = 0

    def _get_upper_body_mode_event(self, ctrl_data):
        for controller_data in ctrl_data.values():
            if not isinstance(controller_data, dict):
                continue
            mode_event = controller_data.get("upper_body_mode_event")
            if mode_event is not None:
                return mode_event
        return None

    def _start_upper_body_transition(self, next_mode: str):
        self._upper_body_transition_start = self._upper_body_target.copy()
        self._upper_body_transition_end = self.upper_body_default_pos.copy()
        self._upper_body_transition_step = 0
        self._upper_body_next_mode = next_mode
        self._upper_body_mode = "transition"

    def _request_upper_body_mode(self, mode_event: str):
        if mode_event == "swing":
            if np.allclose(self._upper_body_target, self.upper_body_default_pos, atol=1e-6):
                self._upper_body_mode = "swing"
                self._upper_body_next_mode = None
                self._upper_body_swing_step = 0
                self._upper_body_target = self.upper_body_default_pos.copy()
            else:
                self._start_upper_body_transition("swing")
        elif mode_event == "lock":
            if np.allclose(self._upper_body_target, self.upper_body_default_pos, atol=1e-6):
                self._upper_body_mode = "locked"
                self._upper_body_next_mode = None
                self._upper_body_target = self.upper_body_default_pos.copy()
            else:
                self._start_upper_body_transition("locked")

    def _compute_swing_upper_body_target(self) -> np.ndarray:
        keyframes = self._upper_body_keyframes
        num_segments = keyframes.shape[0]
        phase = (self._upper_body_swing_step / self._upper_body_swing_period_steps) * num_segments
        seg_idx = int(np.floor(phase)) % num_segments
        next_idx = (seg_idx + 1) % num_segments
        local_t = phase - np.floor(phase)
        blend = 0.5 - 0.5 * np.cos(np.pi * local_t)
        return ((1.0 - blend) * keyframes[seg_idx] + blend * keyframes[next_idx]).astype(np.float32)

    def _compute_upper_body_target(self) -> np.ndarray:
        if self._upper_body_mode == "swing":
            target = self._compute_swing_upper_body_target()
        elif self._upper_body_mode == "transition":
            alpha = min(self._upper_body_transition_step / max(self._upper_body_transition_steps, 1), 1.0)
            target = (
                (1.0 - alpha) * self._upper_body_transition_start
                + alpha * self._upper_body_transition_end
            ).astype(np.float32)
        else:
            target = self.upper_body_default_pos.copy()
        self._upper_body_target = target.copy()
        return target

    def override_pd_target(self, pd_target: np.ndarray) -> np.ndarray:
        pd_target = pd_target.copy()
        pd_target[self._upper_body_env_indices] = self._compute_upper_body_target()
        return pd_target

    def _get_command_controller(self, ctrl_data):
        for controller_data in ctrl_data.values():
            if not isinstance(controller_data, dict):
                continue
            if "axes" in controller_data and "button_event" in controller_data:
                return controller_data
        return None

    def _get_commands(self, ctrl_data):
        for controller_data in ctrl_data.values():
            if not isinstance(controller_data, dict):
                continue
            direct_commands = controller_data.get("commands")
            if direct_commands is not None:
                return np.asarray(direct_commands, dtype=np.float32).copy()

        # if (ref_dof_pos := ctrl_data.get("ref_dof_pos", None)) is not None:
        #     self.arm_action = ref_dof_pos.copy()[-self._n_demo_dof :]

        commands = self.cmd.copy()
        controller_data = self._get_command_controller(ctrl_data)
        if controller_data is None:
            return commands

        axes = controller_data["axes"]
        lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

        commands[0] = command_remap(ly, self.commands_map[0])
        commands[1] = command_remap(rx, self.commands_map[1])
        commands[2] = command_remap(lx, self.commands_map[2])
        # commands[3] = command_remap(ry, self.commands_map[3]) # height

        button_event = controller_data["button_event"]
        for event in button_event:
            if event["type"] == "button" and event["name"] == "Y" and event["pressed"]:
                commands[7] = not commands[7]

        return commands

    def get_observation(self, env_data, ctrl_data):
        if (mode_event := self._get_upper_body_mode_event(ctrl_data)) is not None:
            self._request_upper_body_mode(mode_event)
        self.cmd = self._get_commands(ctrl_data)

        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        base_quat = env_data.base_quat
        base_ang_vel = env_data.base_ang_vel

        rpy = quatToEuler(base_quat)

        self._in_place_stand_flag = np.abs(self.cmd[0]) < 0.1
        self.target_yaw = self.cmd[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), dof_pos[15:]])

        self.adapter_input[0] = 0.75 + self.cmd[3]
        self.adapter_input[1] = self.cmd[4]
        self.adapter_input[2] = self.cmd[5]
        self.adapter_input[3] = self.cmd[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        last_action_full = np.concatenate(
            [self.last_action, (dof_pos - self.default_dof_pos)[-self._n_demo_dof :] / self.action_scale]
        )
        obs_prop = np.concatenate(
            [
                base_ang_vel * self.scales_ang_vel,
                rpy[:2],
                (np.sin(dyaw), np.cos(dyaw)),
                (dof_pos - self.default_dof_pos),
                dof_vel * self.scales_dof_vel,
                last_action_full,
                gait_obs,
                self.adapter_output.cpu().numpy().squeeze(),
            ]
        )
        obs_priv = np.zeros((self.n_priv,))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[: self._n_demo_dof] = dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.cmd[0]
        obs_demo[self._n_demo_dof + 1] = self.cmd[2]
        obs_demo[self._n_demo_dof + 3] = self.cmd[4]
        obs_demo[self._n_demo_dof + 4] = self.cmd[5]
        obs_demo[self._n_demo_dof + 5] = self.cmd[6]
        obs_demo[self._n_demo_dof + 6 : self._n_demo_dof + 9] = 0.75 + self.cmd[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        if self._in_place_stand_flag and (
            (np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
        ):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and (
            (np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
        ):
            self.gait_cycle = np.array([0.25, 0.75])
        extras = {
            "commands": self.cmd.copy(),
        }
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist)), extras

    def get_action(self, obs):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = (
                torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float)
                .view(1, -1)
                .to(self.device)
            )
            raw_action = self.model(obs_tensor, extra_hist).cpu().numpy().squeeze()

        raw_action = np.clip(raw_action, -40.0, 40.0)
        self.last_action = raw_action.copy()

        scaled_actions = raw_action * self.action_scale
        return scaled_actions
