# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from diffaero_common.keys import EXTRA_SIM_STATE, EXTRA_TASK_TERMS, OBS_POLICY
from diffaero_uav.adapters import build_newton_adapter
from diffaero_uav.adapters.newton import NewtonBackendAdapter

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from diffaero_env.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg
from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.pointmass_continuous import PMCDynamicsBridge
from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.pointmass_discrete import PMDDynamicsBridge
from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.quad import QuadDynamicsBridge
from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.simplified_quad import SimpleDynamicsBridge
from diffaero_env.tasks.direct.drone_racing.mdp import (
    compute_dones,
    compute_observations,
    compute_rewards,
    reset_body_state,
)
from diffaero_env.tasks.direct.drone_racing.state.sim_state import build_sim_state

_BRIDGE_CLASSES = {
    "quad": QuadDynamicsBridge,
    "pmd": PMDDynamicsBridge,
    "pmc": PMCDynamicsBridge,
    "simple": SimpleDynamicsBridge,
}


class DroneRacingEnv(DirectRLEnv):
    cfg: DroneRacingEnvCfg

    def __init__(self, cfg: DroneRacingEnvCfg, render_mode: str | None = None, **kwargs: Any):
        self._bridge: (
            QuadDynamicsBridge
            | PMDDynamicsBridge
            | PMCDynamicsBridge
            | SimpleDynamicsBridge
            | NewtonBackendAdapter
            | None
        ) = None
        self._prev_action: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self) -> None:
        self.robot = self.scene.articulations["robot"]
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        dynamics_model = getattr(self.cfg, "dynamics_model", "quad")
        # Check if this is a Warp/Newton backend by looking at sim.physics type
        sim_physics = getattr(self.cfg.sim, "physics", None)
        is_warp = sim_physics is not None and "Newton" in type(sim_physics).__name__

        if is_warp and dynamics_model == "quad":
            self._bridge = build_newton_adapter(
                cfg=self.cfg,
                robot=self.robot,
                num_envs=self.num_envs,
                device=self.device,
                backend="warp",
            )
        else:
            bridge_cls = _BRIDGE_CLASSES.get(dynamics_model, QuadDynamicsBridge)
            self._bridge = bridge_cls(
                cfg=self.cfg,
                robot=self.robot,
                num_envs=self.num_envs,
                device=self.device,
            )

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_action = torch.zeros_like(self.actions)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_action = self.actions.clone()
        self.actions = actions.clone()
        if self._bridge is not None:
            self._bridge.process_action(actions)

    def _apply_action(self) -> None:
        if self._bridge is not None:
            self._bridge.apply_to_sim()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self._bridge is None:
            return {OBS_POLICY: torch.zeros(self.num_envs, 0, device=self.device)}
        bridge_state = self._bridge.read_base_state()
        motor_state = self._bridge.read_motor_state()
        full_state = {**bridge_state, **motor_state}
        enable_critic = self.cfg.state_space > 0
        return compute_observations(full_state, self.actions, enable_critic)

    def _get_rewards(self) -> torch.Tensor:
        if self._bridge is None:
            return torch.zeros(self.num_envs, device=self.device)
        bridge_state = self._bridge.read_base_state()
        motor_state = self._bridge.read_motor_state()
        dynamics_info = self._bridge.read_dynamics_info()

        total_reward, task_terms = compute_rewards(
            rew_scale_progress=self.cfg.rew_scale_progress,
            rew_scale_tracking=self.cfg.rew_scale_tracking,
            rew_scale_control_effort=self.cfg.rew_scale_control_effort,
            rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
            rew_scale_terminal=self.cfg.rew_scale_terminal,
            rew_scale_gate=self.cfg.rew_scale_gate,
            angular_velocity_b=bridge_state["angular_velocity_b"],
            last_action=self.actions,
            prev_action=self._prev_action if self._prev_action is not None else torch.zeros_like(self.actions),
            reset_terminated=self.reset_terminated,
            step_count=self.episode_length_buf.float(),
        )

        sim_state = build_sim_state(
            position_w=bridge_state["position_w"],
            quaternion_w=bridge_state["quaternion_w"],
            linear_velocity_w=bridge_state["linear_velocity_w"],
            angular_velocity_b=bridge_state["angular_velocity_b"],
            motor_omega=motor_state["motor_omega"],
            step_count=self.episode_length_buf.float(),
            last_action=self.actions,
            progress=task_terms["progress"],
            dynamics_info=dynamics_info,
        )

        self.extras[EXTRA_TASK_TERMS] = task_terms
        self.extras[EXTRA_SIM_STATE] = sim_state

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_dones(
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            reset_terminated=self.reset_terminated,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        env_ids_t = env_ids
        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        super()._reset_idx(env_ids_t)

        reset_body_state(self.robot, env_ids_t, self.scene.env_origins, self.device)

        if self._bridge is not None:
            self._bridge.reset(env_ids_t)

        self._prev_action[env_ids_t] = 0.0
        self.actions[env_ids_t] = 0.0
