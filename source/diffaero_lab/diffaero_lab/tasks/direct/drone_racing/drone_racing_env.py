# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from diffaero_lab.common.keys import EXTRA_CAPABILITIES, EXTRA_SIM_STATE, EXTRA_TASK_TERMS, OBS_POLICY
from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg
from diffaero_lab.tasks.direct.drone_racing.mdp import (
    compute_dones,
    compute_observations,
    compute_rewards,
    reset_body_state,
)
from diffaero_lab.tasks.direct.drone_racing.state.sim_state import build_sim_state
from diffaero_lab.uav.route_registry import RouteRegistry, RouteSpec


class DroneRacingEnv(DirectRLEnv):
    cfg: DroneRacingEnvCfg

    def __init__(self, cfg: DroneRacingEnvCfg, render_mode: str | None = None, **kwargs: Any):
        self._bridge = None
        self._route_spec: RouteSpec | None = None
        self._capabilities: dict[str, bool] = {}
        self._prev_action: torch.Tensor | None = None
        self._prev_position_w: torch.Tensor | None = None
        self._gate_index: torch.Tensor | None = None
        self._gates_passed: torch.Tensor | None = None
        self._gate_positions: torch.Tensor | None = None
        self._gate_yaws: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self) -> None:
        self.robot = self.scene.articulations["robot"]
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._bridge, self._route_spec = RouteRegistry.build_adapter(
            cfg=self.cfg,
            robot=self.robot,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._capabilities = self._route_spec.build_capabilities(supports_critic_state=self.cfg.state_space > 0)

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_action = torch.zeros_like(self.actions)
        self._prev_position_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._gate_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gates_passed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._init_position_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Build local gate positions from config
        track_cfg = self.cfg.scene.track
        num_gates = len(track_cfg.rigid_objects)
        local_gate_pos = torch.zeros(num_gates, 3, device=self.device)
        local_gate_yaw = torch.zeros(num_gates, device=self.device)
        for i, (gate_name, obj_cfg) in enumerate(track_cfg.rigid_objects.items()):
            local_gate_pos[i] = torch.tensor(obj_cfg.init_state.pos, device=self.device)
            quat_xyzw = torch.tensor(obj_cfg.init_state.rot, device=self.device)
            q_x, q_y, q_z, q_w = quat_xyzw.unbind(dim=-1)
            local_gate_yaw[i] = torch.atan2(
                2.0 * (q_w * q_z + q_x * q_y),
                1.0 - 2.0 * (q_y * q_y + q_z * q_z),
            )

        # Global gate positions: (num_envs, num_gates, 3)
        self._gate_positions = self.scene.env_origins.unsqueeze(1) + local_gate_pos.unsqueeze(0)
        self._gate_yaws = local_gate_yaw.unsqueeze(0).repeat(self.num_envs, 1)

    def _gate_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_gates = self._gate_positions.shape[1]
        target_gate_idx = torch.remainder(self._gate_index, num_gates)
        next_gate_idx = torch.remainder(target_gate_idx + 1, num_gates)
        env_ids = torch.arange(self.num_envs, device=self.device)
        return (
            self._gate_positions[env_ids, target_gate_idx],
            self._gate_yaws[env_ids, target_gate_idx],
            self._gate_positions[env_ids, next_gate_idx],
            self._gate_yaws[env_ids, next_gate_idx],
        )

    @staticmethod
    def _detach_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {key: DroneRacingEnv._detach_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DroneRacingEnv._detach_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DroneRacingEnv._detach_tree(item) for item in value)
        return value

    def detach(self) -> None:
        """Detach tensors kept across rollouts after an actor update."""
        for name in ("actions", "_prev_action", "_prev_position_w", "_init_position_w", "_gates_passed"):
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.detach())
        if self._bridge is not None and hasattr(self._bridge, "detach"):
            self._bridge.detach()
        self.extras = self._detach_tree(self.extras)

    @staticmethod
    def _split_reset_env_ids(env_ids: Sequence[int] | torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(env_ids, torch.Tensor):
            env_ids_sim = env_ids.to(device=device)
        else:
            env_ids_sim = torch.tensor(env_ids, device=device)
        return env_ids_sim.to(dtype=torch.int32), env_ids_sim.to(dtype=torch.long)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_action = self.actions.clone()
        self.actions = actions.clone()
        if self._bridge is not None:
            self._bridge.process_action(actions)

    def _apply_action(self) -> None:
        if self._bridge is not None:
            self._bridge.apply_to_sim()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_shared_extras()
        if self._bridge is None:
            return {OBS_POLICY: torch.zeros(self.num_envs, 0, device=self.device)}
        bridge_state = self._bridge.read_base_state()
        motor_state = self._bridge.read_motor_state()
        full_state = {**bridge_state, **motor_state}
        enable_critic = self.cfg.state_space > 0
        target_position_w, target_yaw, next_target_position_w, next_target_yaw = self._gate_targets()
        return compute_observations(
            full_state,
            enable_critic,
            target_position_w=target_position_w,
            target_yaw=target_yaw,
            next_target_position_w=next_target_position_w,
            next_target_yaw=next_target_yaw,
        )

    def _update_shared_extras(self) -> None:
        self.extras[EXTRA_CAPABILITIES] = self._capabilities

    def _get_rewards(self) -> torch.Tensor:
        if self._bridge is None:
            return torch.zeros(self.num_envs, device=self.device)
        bridge_state = self._bridge.read_base_state()
        motor_state = self._bridge.read_motor_state()
        dynamics_info = self._bridge.read_dynamics_info()
        if self._route_spec is not None:
            dynamics_info = {
                **dynamics_info,
                "physics_route": self._route_spec.physics_route,
                "tensor_backend": self._route_spec.tensor_backend,
                "write_mode": self._route_spec.write_mode,
                "quat_convention": self._route_spec.quat_convention,
            }

        position_w = bridge_state["position_w"]
        target_position_w, target_yaw, _, _ = self._gate_targets()

        total_reward, task_terms, updated_gate_index, updated_gates_passed, collision = compute_rewards(
            rew_scale_progress=self.cfg.rew_scale_progress,
            rew_scale_tracking=self.cfg.rew_scale_tracking,
            rew_scale_control_effort=self.cfg.rew_scale_control_effort,
            rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
            rew_scale_terminal=self.cfg.rew_scale_terminal,
            rew_scale_gate=self.cfg.rew_scale_gate,
            position_w=position_w,
            target_position_w=target_position_w,
            target_yaw=target_yaw,
            prev_position_w=self._prev_position_w,
            gate_index=self._gate_index,
            gates_passed=self._gates_passed,
            angular_velocity_b=bridge_state["angular_velocity_b"],
            last_action=self.actions,
            prev_action=self._prev_action if self._prev_action is not None else torch.zeros_like(self.actions),
            reset_terminated=self.reset_terminated,
            step_count=self.episode_length_buf.float(),
            gate_l1_radius=self.cfg.gate_l1_radius,
        )

        self._gate_index = updated_gate_index
        self._gates_passed = updated_gates_passed
        self._prev_position_w = position_w.clone().detach()

        sim_state = build_sim_state(
            position_w=position_w,
            quaternion_w=bridge_state["quaternion_w"],
            linear_velocity_w=bridge_state["linear_velocity_w"],
            angular_velocity_b=bridge_state["angular_velocity_b"],
            motor_omega=motor_state["motor_omega"],
            step_count=self.episode_length_buf.float(),
            last_action=self.actions,
            progress=task_terms["progress"],
            target_position_w=target_position_w,
            dynamics_info=dynamics_info,
        )

        self.extras[EXTRA_TASK_TERMS] = task_terms
        self.extras[EXTRA_SIM_STATE] = sim_state
        self.extras[EXTRA_CAPABILITIES] = self._capabilities

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._bridge is None:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            return terminated, time_out
        bridge_state = self._bridge.read_base_state()
        target_position_w, target_yaw, _, _ = self._gate_targets()
        return compute_dones(
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            prev_position_w=self._prev_position_w,
            position_w=bridge_state["position_w"],
            target_position_w=target_position_w,
            target_yaw=target_yaw,
            env_origins=self.scene.env_origins,
            gate_l1_radius=self.cfg.gate_l1_radius,
            oob_xy_limit=self.cfg.oob_xy_limit,
            oob_z_max=self.cfg.oob_z_max,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        env_ids_sim, env_ids_index = self._split_reset_env_ids(env_ids, self.device)

        super()._reset_idx(env_ids_sim)

        self._gate_index[env_ids_index] = 0
        self._gates_passed[env_ids_index] = 0.0
        target_position_w = self._gate_positions[env_ids_index, self._gate_index[env_ids_index]]
        target_yaw = self._gate_yaws[env_ids_index, self._gate_index[env_ids_index]]
        init_pos = target_position_w - torch.stack(
            [torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)], dim=-1
        )
        reset_body_state(self.robot, env_ids_sim, self.scene.env_origins, self.device, root_position_w=init_pos)

        if self._bridge is not None:
            self._bridge.reset(env_ids_index)

        self._prev_action[env_ids_index] = 0.0
        self.actions[env_ids_index] = 0.0

        if init_pos.shape[0] > 0:
            self._init_position_w[env_ids_index] = init_pos
        self._prev_position_w[env_ids_index] = init_pos
        self._update_shared_extras()
