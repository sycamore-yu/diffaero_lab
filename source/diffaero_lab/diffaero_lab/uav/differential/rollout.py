# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native differentiable rollout for drone racing.

WarpDroneRollout owns the full differentiable pipeline inside wp.Tape():
- Newton model | Warp MLP actor | action→wrench | obs | loss | gate detection

IsaacLab is used only for initial state extraction and evaluation.
"""

from __future__ import annotations

import torch
import warp as wp

from diffaero_lab.uav.differential.kernels import (
    action_to_wrench_kernel,
    compute_loss_kernel,
    compute_obs_kernel,
    init_mlp_params,
    mlp_forward_kernel,
)
from diffaero_lab.uav.differential.model import WarpDroneModel


class RolloutConfig:
    """Configuration for Warp-native differentiable rollout."""

    def __init__(
        self,
        num_envs: int = 256,
        horizon: int = 32,
        sim_dt: float = 1.0 / 400.0,
        sim_substeps: int = 4,
        obs_dim: int = 13,
        action_dim: int = 4,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        thrust_scale: float = 1.9,
        moment_scale: float = 0.01,
        robot_weight: float | None = None,
        rew_scale_progress: float = 20.0,
        rew_scale_tracking: float = 1.0,
        rew_scale_ang_vel: float = 0.0001,
        rew_scale_control_effort: float = 0.0001,
        gate_l1_radius: float = 1.5,
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.sim_dt = sim_dt
        self.sim_substeps = sim_substeps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.thrust_scale = thrust_scale
        self.moment_scale = moment_scale
        self.robot_weight = robot_weight
        self.rew_scale_progress = rew_scale_progress
        self.rew_scale_tracking = rew_scale_tracking
        self.rew_scale_ang_vel = rew_scale_ang_vel
        self.rew_scale_control_effort = rew_scale_control_effort
        self.gate_l1_radius = gate_l1_radius


# -- State extraction kernels (body_q/qd → separate position/velocity arrays) --


@wp.kernel
def _extract_position(
    body_q: wp.array(dtype=wp.transform),
    pos: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
):
    """Extract position and quaternion from body transforms."""
    tid = wp.tid()
    tf = body_q[tid]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    pos[tid] = p
    quat[tid] = wp.vec4(wp.quat_x(q), wp.quat_y(q), wp.quat_z(q), wp.quat_w(q))


@wp.kernel
def _extract_velocity(
    body_qd: wp.array(dtype=wp.spatial_vector),
    lin_vel: wp.array(dtype=wp.vec3),
    ang_vel: wp.array(dtype=wp.vec3),
):
    """Extract linear and angular velocity from spatial vectors."""
    tid = wp.tid()
    sv = body_qd[tid]
    lin_vel[tid] = wp.vec3(
        wp.spatial_top(sv)[0],
        wp.spatial_top(sv)[1],
        wp.spatial_top(sv)[2],
    )
    ang_vel[tid] = wp.vec3(
        wp.spatial_bottom(sv)[0],
        wp.spatial_bottom(sv)[1],
        wp.spatial_bottom(sv)[2],
    )


@wp.kernel
def _copy_vec3(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
):
    """Copy vec3 array for prev_position tracking."""
    tid = wp.tid()
    dst[tid] = src[tid]


class WarpDroneRollout:
    """Warp-native differentiable rollout manager.

    Usage::

        cfg = RolloutConfig(num_envs=256, horizon=32)
        rollout = WarpDroneRollout(cfg)

        # Set initial state from IsaacLab
        rollout.set_initial_state(sim_state_dict)

        # Train one iteration
        loss = rollout.train_step(lr=3e-4)

        # Export for IsaacLab eval
        w, b = rollout.export_actor_pytorch()
    """

    def __init__(self, cfg: RolloutConfig, device: str = "cuda:0") -> None:
        wp.set_device(device)

        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.num_envs
        self.horizon = cfg.horizon
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.total_steps = cfg.horizon * cfg.sim_substeps

        # -- Newton model --
        self.model = WarpDroneModel(
            num_envs=cfg.num_envs,
            dt=cfg.sim_dt,
            requires_grad=True,
            device=device,
        )

        # -- Robot weight --
        if cfg.robot_weight is None:
            masses = wp.to_torch(self.model.model.body_mass)
            grav = torch.tensor(getattr(self.model.model, "gravity", (0.0, 0.0, -9.81)), device="cpu").norm()
            self.robot_weight = (masses[0].sum() * grav).item()
        else:
            self.robot_weight = cfg.robot_weight

        # -- Layer dimensions --
        layer_dims = [cfg.obs_dim] + list(cfg.hidden_dims) + [cfg.action_dim]

        # -- MLP parameters (single flat Warp array, requires_grad=True) --
        self.mlp_params = init_mlp_params(layer_dims, device=device, requires_grad=True)
        self.mlp_layer_dims = wp.array(layer_dims, dtype=int, device=device)

        # -- Hidden buffer for MLP scratch space --
        max_hidden = max(cfg.hidden_dims) if cfg.hidden_dims else cfg.action_dim
        self.mlp_hidden = wp.zeros((cfg.num_envs, max_hidden), dtype=float, device=device)

        # -- State ring buffer --
        self._states = [self.model.state() for _ in range(self.total_steps + 1)]
        self._controls = [self.model.control() for _ in range(self.total_steps)]
        self._contacts_list = [self.model.contacts() for _ in range(self.total_steps)]

        # -- Observation buffer (one per horizon step + initial) --
        self._obs_buf = [
            wp.zeros((cfg.num_envs, cfg.obs_dim), dtype=float, device=device) for _ in range(cfg.horizon + 1)
        ]

        # -- Action buffer --
        self._actions = [
            wp.zeros((cfg.num_envs, cfg.action_dim), dtype=float, device=device) for _ in range(cfg.horizon)
        ]

        # -- Previous action (starts as zeros) --
        self._prev_action = wp.zeros((cfg.num_envs, cfg.action_dim), dtype=float, device=device)

        # -- State views (extracted for obs/loss kernels) --
        self._pos = wp.zeros(cfg.num_envs, dtype=wp.vec3, device=device)
        self._quat = wp.zeros(cfg.num_envs, dtype=wp.vec4, device=device)
        self._lin_vel = wp.zeros(cfg.num_envs, dtype=wp.vec3, device=device)
        self._ang_vel = wp.zeros(cfg.num_envs, dtype=wp.vec3, device=device)
        self._prev_pos = wp.zeros(cfg.num_envs, dtype=wp.vec3, device=device)

        # -- Gate targets --
        self._target_pos = wp.zeros(cfg.num_envs, dtype=wp.vec3, device=device)
        self._target_yaw = wp.zeros(cfg.num_envs, dtype=float, device=device)
        self._next_target_pos = wp.zeros(cfg.num_envs, dtype=wp.vec3, device=device)
        self._next_target_yaw = wp.zeros(cfg.num_envs, dtype=float, device=device)

        # -- Loss accumulation --
        self._step_loss = wp.zeros(cfg.num_envs, dtype=float, device=device)
        self._total_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

        # -- Public aliases for WarpAPG --
        self.params = self.mlp_params
        self.loss = self._total_loss

        # Lazy: layer dims for export
        self._layer_dims = layer_dims

    # -- Public API ----------------------------------------------------------

    @property
    def layer_dims(self) -> list[int]:
        return self._layer_dims

    def forward(self) -> None:
        """Run the full differentiable rollout forward pass.

        Must be called inside a wp.Tape() context by WarpAPG.
        """
        self._forward()

    def set_initial_state(self, sim_state: dict[str, torch.Tensor]) -> None:
        """Set initial drone state from IsaacLab sim_state dict.

        Args:
            sim_state: Dict with keys position_w (N,3), quaternion_w (N,4 xyzw),
                linear_velocity_w (N,3), angular_velocity_b (N,3),
                target_position_w (N,3), and optionally target_yaw etc.
        """
        pos_w = sim_state["position_w"]
        quat_w = sim_state["quaternion_w"]
        lin_vel_w = sim_state["linear_velocity_w"]
        ang_vel_b = sim_state.get("angular_velocity_b")
        if ang_vel_b is None:
            ang_vel_b = torch.zeros(self.num_envs, 3, device="cuda:0")

        state0 = self._states[0]
        for env_id in range(self.num_envs):
            state0.body_q.assign(
                wp.transform(
                    wp.vec3(float(pos_w[env_id, 0]), float(pos_w[env_id, 1]), float(pos_w[env_id, 2])),
                    wp.quat(
                        float(quat_w[env_id, 0]),
                        float(quat_w[env_id, 1]),
                        float(quat_w[env_id, 2]),
                        float(quat_w[env_id, 3]),
                    ),
                ),
                index=env_id,
            )
            state0.body_qd.assign(
                wp.spatial_vector(
                    float(lin_vel_w[env_id, 0]),
                    float(lin_vel_w[env_id, 1]),
                    float(lin_vel_w[env_id, 2]),
                    float(ang_vel_b[env_id, 0]),
                    float(ang_vel_b[env_id, 1]),
                    float(ang_vel_b[env_id, 2]),
                ),
                index=env_id,
            )

        # Collision detection for initial state
        self.model.collision_pipeline.collide(state0, self._contacts_list[0])

        # Set gate targets
        tp = sim_state.get("target_position_w")
        if tp is not None:
            for env_id in range(self.num_envs):
                self._target_pos.assign(
                    wp.vec3(float(tp[env_id, 0]), float(tp[env_id, 1]), float(tp[env_id, 2])),
                    index=env_id,
                )

        # Store initial position as previous for progress loss
        for env_id in range(self.num_envs):
            self._prev_pos.assign(
                wp.vec3(float(pos_w[env_id, 0]), float(pos_w[env_id, 1]), float(pos_w[env_id, 2])),
                index=env_id,
            )

        # Compute initial observation
        self._extract_and_obs(0, is_initial=True)

    def set_gate_targets(
        self,
        target_pos: torch.Tensor,
        target_yaw: torch.Tensor,
        next_target_pos: torch.Tensor,
        next_target_yaw: torch.Tensor,
    ) -> None:
        """Update gate target positions and yaws."""
        for env_id in range(self.num_envs):
            self._target_pos.assign(
                wp.vec3(float(target_pos[env_id, 0]), float(target_pos[env_id, 1]), float(target_pos[env_id, 2])),
                index=env_id,
            )
            self._target_yaw.assign(float(target_yaw[env_id]), index=env_id)
            self._next_target_pos.assign(
                wp.vec3(
                    float(next_target_pos[env_id, 0]),
                    float(next_target_pos[env_id, 1]),
                    float(next_target_pos[env_id, 2]),
                ),
                index=env_id,
            )
            self._next_target_yaw.assign(float(next_target_yaw[env_id]), index=env_id)

    def export_actor_pytorch(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Export actor weights/biases as PyTorch tensors.

        Returns:
            (weights_list, biases_list) for use in PyTorch MLP evaluation.
        """
        layer_dims = [self.obs_dim] + list(self.cfg.hidden_dims) + [self.cfg.action_dim]
        num_layers = len(layer_dims) - 1
        flat = wp.to_torch(self.mlp_params).clone().detach().cpu()

        weights = []
        biases = []
        offset = 0
        for li in range(num_layers):
            in_d = layer_dims[li]
            out_d = layer_dims[li + 1]
            w_size = out_d * in_d
            w = flat[offset : offset + w_size].reshape(out_d, in_d).clone()
            offset += w_size
            b = flat[offset : offset + out_d].clone()
            offset += out_d
            weights.append(w)
            biases.append(b)

        return weights, biases

    # -- Internal ------------------------------------------------------------

    def _extract_state(self, state_idx: int) -> None:
        """Extract position/quat/velocity from state buffer."""
        state = self._states[state_idx]
        wp.launch(_extract_position, dim=self.num_envs, inputs=[state.body_q], outputs=[self._pos, self._quat])
        wp.launch(_extract_velocity, dim=self.num_envs, inputs=[state.body_qd], outputs=[self._lin_vel, self._ang_vel])

    def _extract_and_obs(self, state_idx: int, *, is_initial: bool = False) -> None:
        """Extract state and compute observation, storing in obs_buf[step_t]."""
        self._extract_state(state_idx)
        if is_initial:
            wp.launch(
                compute_obs_kernel,
                dim=self.num_envs,
                inputs=[
                    self._pos,
                    self._quat,
                    self._lin_vel,
                    self._target_pos,
                    self._target_yaw,
                    self._next_target_pos,
                    self._next_target_yaw,
                ],
                outputs=[self._obs_buf[0]],
            )

    def _forward(self) -> None:
        """Run the full differentiable rollout forward pass.

        For each t in 0..horizon-1:
            1. MLP: obs[t] → action[t]
            2. For each substep: clear forces, action→wrench, solver.step()
            3. Extract state, compute obs[t+1]
            4. Compute loss for this step, accumulate into total_loss
        """
        self._total_loss.zero_()

        num_layers = len(self.cfg.hidden_dims) + 1

        for step_t in range(self.horizon):
            sub_base = step_t * self.cfg.sim_substeps

            # -- 1. MLP forward: obs → action --
            wp.launch(
                mlp_forward_kernel,
                dim=self.num_envs,
                inputs=[
                    self._obs_buf[step_t],
                    self.mlp_params,
                    self.mlp_layer_dims,
                    num_layers,
                ],
                outputs=[self.mlp_hidden, self._actions[step_t]],
            )

            # -- 2. Substeps: apply wrench + solver step --
            for sub in range(self.cfg.sim_substeps):
                t = sub_base + sub
                state = self._states[t]
                next_state = self._states[t + 1]
                control = self._controls[t]
                contacts = self._contacts_list[t]

                state.clear_forces()

                wp.launch(
                    action_to_wrench_kernel,
                    dim=self.num_envs,
                    inputs=[
                        self._actions[step_t],
                        self.cfg.thrust_scale,
                        self.cfg.moment_scale,
                        self.robot_weight,
                        self.num_envs,
                    ],
                    outputs=[state.body_f],
                )

                self.model.solver.step(state, next_state, control, contacts, self.cfg.sim_dt)

            # -- 3. Extract state from final substep + compute observation --
            final_state_idx = sub_base + self.cfg.sim_substeps
            self._extract_state(final_state_idx)

            if step_t + 1 < self.horizon:
                wp.launch(
                    compute_obs_kernel,
                    dim=self.num_envs,
                    inputs=[
                        self._pos,
                        self._quat,
                        self._lin_vel,
                        self._target_pos,
                        self._target_yaw,
                        self._next_target_pos,
                        self._next_target_yaw,
                    ],
                    outputs=[self._obs_buf[step_t + 1]],
                )

            # -- 4. Compute loss for this step --
            wp.launch(
                compute_loss_kernel,
                dim=self.num_envs,
                inputs=[
                    self._pos,
                    self._prev_pos,
                    self._target_pos,
                    self._ang_vel,
                    self._actions[step_t],
                    self._prev_action,
                    self.cfg.rew_scale_progress,
                    self.cfg.rew_scale_tracking,
                    self.cfg.rew_scale_ang_vel,
                    self.cfg.rew_scale_control_effort,
                ],
                outputs=[self._step_loss],
            )

            # Accumulate step loss into total
            wp.launch(
                self._accumulate_loss_kernel,
                dim=self.num_envs,
                inputs=[self._step_loss],
                outputs=[self._total_loss],
            )

            # Update prev_pos and prev_action for next step
            wp.launch(_copy_vec3, dim=self.num_envs, inputs=[self._pos], outputs=[self._prev_pos])
            wp.copy(self._prev_action, self._actions[step_t])

    @wp.kernel
    def _accumulate_loss_kernel(
        step_loss: wp.array(dtype=float),
        total_loss: wp.array(dtype=wp.float32),
    ):
        """Accumulate per-env losses into single scalar."""
        tid = wp.tid()
        wp.atomic_add(total_loss, 0, step_loss[tid])
