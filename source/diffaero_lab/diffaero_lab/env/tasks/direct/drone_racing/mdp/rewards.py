# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_lab.common.terms import (
    TERM_ANGULAR_RATE,
    TERM_CONTROL_EFFORT,
    TERM_CONTROL_SMOOTHNESS,
    TERM_GATE_PASS,
    TERM_PROGRESS,
    TERM_TERMINAL,
    TERM_TIME_PENALTY,
    TERM_TRACKING_ERROR,
)
from diffaero_lab.env.tasks.direct.drone_racing.state.task_terms import build_task_terms

# Minimal implicit gate model parameters
_GATE_SPACING: float = 5.0  # Distance between gates along x-axis
_GATE_HEIGHT: float = 1.0  # Default gate height (z)
_GATE_WIDTH: float = 2.0  # Half-width of gate opening (for tracking error)
_GROUND_COLLISION_THRESHOLD: float = 0.05  # z position below this = collision


def compute_rewards(
    rew_scale_progress: float,
    rew_scale_tracking: float,
    rew_scale_control_effort: float,
    rew_scale_ang_vel: float,
    rew_scale_terminal: float,
    rew_scale_gate: float,
    position_w: Tensor,
    target_position_w: Tensor,
    prev_position_w: Tensor,
    gate_index: Tensor,
    gates_passed: Tensor,
    angular_velocity_b: Tensor,
    last_action: Tensor,
    prev_action: Tensor,
    reset_terminated: Tensor,
    step_count: Tensor,
) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor, Tensor]:
    """Compute rewards with real racing semantics.

    Args:
        rew_scale_progress: Reward scale for progress toward target
        rew_scale_tracking: Reward scale for tracking error (negative = penalty)
        rew_scale_control_effort: Reward scale for control effort (negative = penalty)
        rew_scale_ang_vel: Reward scale for angular velocity (negative = penalty)
        rew_scale_terminal: Reward scale for terminal states (negative = penalty)
        rew_scale_gate: Reward scale for gate passage bonus
        position_w: Current drone position [num_envs, 3]
        target_position_w: Target gate position [num_envs, 3]
        prev_position_w: Previous step drone position [num_envs, 3]
        gate_index: Current gate index for each env [num_envs]
        gates_passed: Cumulative gates passed count [num_envs]
        angular_velocity_b: Angular velocity in body frame [num_envs, 3]
        last_action: Last action taken [num_envs, action_dim]
        prev_action: Previous action taken [num_envs, action_dim]
        reset_terminated: Which envs are terminated [num_envs]
        step_count: Current step count per env [num_envs]

    Returns:
        total_reward: Sum of all reward terms [num_envs]
        task_terms: Dict of individual term values for logging
        updated_gate_index: Updated gate index (may increment if gate crossed)
        updated_gates_passed: Updated gates passed count
        collision: Collision indicator [num_envs]
    """
    num_envs = position_w.shape[0]
    device = position_w.device

    # Compute distance to target
    diff_to_target = target_position_w - position_w
    dist_to_target = torch.norm(diff_to_target, dim=-1)

    # Progress: inverse distance to target (closer = higher progress)
    progress = 1.0 / (dist_to_target + 1e-6)

    # Tracking error: perpendicular distance from the direct path to target
    # Vector from drone to target
    dir_to_target = diff_to_target / (dist_to_target.unsqueeze(-1) + 1e-6)
    # Perpendicular component = lateral deviation
    # tracking_error = ||position - projection_onto_path|| = ||diff - dir * dot(diff, dir)||
    along_track = torch.sum(diff_to_target * dir_to_target, dim=-1, keepdim=True)
    projection = dir_to_target * along_track
    tracking_error = torch.norm(diff_to_target - projection, dim=-1)

    # Gate pass: z-plane crossing detection
    # Gate N is passed when drone's z-position exceeds the gate's z-position
    # prev_z < target_z AND curr_z >= target_z indicates crossing from below
    gate_crossed = (prev_position_w[:, 2] < target_position_w[:, 2]) & (position_w[:, 2] >= target_position_w[:, 2])
    updated_gate_index = gate_index + gate_crossed.long()
    updated_gates_passed = gates_passed + gate_crossed.float()

    gate_pass = gates_passed.float()

    # Collision: ground contact when z position is below threshold
    collision = (position_w[:, 2] < _GROUND_COLLISION_THRESHOLD).float()

    # Other reward terms
    terminal = reset_terminated.float()
    control_effort = torch.sum(torch.abs(last_action), dim=-1)
    control_smoothness = torch.sum(torch.abs(last_action - prev_action), dim=-1)
    angular_rate = torch.sum(torch.abs(angular_velocity_b), dim=-1)
    time_penalty = torch.ones(num_envs, device=device) * -0.001

    # Compute individual reward components
    rew_progress = rew_scale_progress * progress
    rew_tracking = rew_scale_tracking * tracking_error
    rew_gate = rew_scale_gate * gate_pass
    rew_control = rew_scale_control_effort * control_effort
    rew_ang_vel = rew_scale_ang_vel * angular_rate
    rew_terminal = rew_scale_terminal * terminal

    total_reward = rew_progress + rew_tracking + rew_gate + rew_control + rew_ang_vel + rew_terminal + time_penalty

    task_terms = build_task_terms(
        progress=progress,
        tracking_error=tracking_error,
        gate_pass=gate_pass,
        collision=collision,
        terminal=terminal,
        control_effort=control_effort,
        control_smoothness=control_smoothness,
        angular_rate=angular_rate,
        time_penalty=time_penalty,
    )

    return total_reward, task_terms, updated_gate_index, updated_gates_passed, collision
