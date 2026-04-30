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
from diffaero_lab.tasks.direct.drone_racing.mdp.gates import gate_crossing
from diffaero_lab.tasks.direct.drone_racing.state.task_terms import build_task_terms

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
    target_yaw: Tensor,
    prev_position_w: Tensor,
    gate_index: Tensor,
    gates_passed: Tensor,
    angular_velocity_b: Tensor,
    last_action: Tensor,
    prev_action: Tensor,
    reset_terminated: Tensor,
    step_count: Tensor,
    gate_l1_radius: float,
) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor, Tensor]:
    """Compute rewards with differentiable loss for APG training.

    The key design follows diffaero/env/racing.py:
    - progress_loss = d2g_new - d2g_old.detach()
      d2g_new retains gradients through the physics dynamics,
      d2g_old is detached from the previous step.
      This gives the actor a gradient signal to move toward the gate.

    Returns:
        total_reward: Sum of all reward terms [num_envs] (detached, for logging)
        task_terms: Dict including 'loss' (differentiable) and 'progress' (for logging)
        updated_gate_index, updated_gates_passed, collision
    """
    num_envs = position_w.shape[0]
    device = position_w.device

    # ── Differentiable loss (retains gradients through position_w) ──
    # Distance to target gate: current (grad-enabled) vs previous (detached)
    d2g_new = torch.norm(position_w - target_position_w, dim=-1)
    d2g_old = torch.norm(prev_position_w - target_position_w, dim=-1)
    # progress_loss > 0 means the drone moved away from the gate
    progress_loss = d2g_new - d2g_old.detach()

    # Position loss: 1 - exp(-distance), saturates far from gate
    pos_loss = 1.0 - torch.exp(-d2g_new)

    # Angular rate penalty for stability
    ang_rate_loss = torch.norm(angular_velocity_b, dim=-1)

    # Action smoothness penalty (jerk)
    jerk_loss = torch.sum((last_action - prev_action) ** 2, dim=-1)

    # Combined differentiable loss for APG backward pass
    diff_loss = (
        rew_scale_progress * progress_loss
        + abs(rew_scale_tracking) * pos_loss
        + abs(rew_scale_ang_vel) * ang_rate_loss
        + abs(rew_scale_control_effort) * jerk_loss
    )

    # ── Gate crossing detection: same gate-frame plane crossing as diffaero ──
    gate_crossed, gate_collision = gate_crossing(
        prev_position_w=prev_position_w,
        position_w=position_w,
        gate_position_w=target_position_w,
        gate_yaw=target_yaw,
        gate_l1_radius=gate_l1_radius,
    )
    updated_gate_index = gate_index + gate_crossed.long()
    updated_gates_passed = gates_passed + gate_crossed.float()

    collision = gate_collision.float()

    # ── Reward for RL logging (detached) ──
    terminal = reset_terminated.float()
    progress = 1.0 / (d2g_new.detach() + 1e-6)
    rew_progress = rew_scale_progress * (-progress_loss.detach())
    rew_tracking = rew_scale_tracking * pos_loss.detach()
    rew_control = rew_scale_control_effort * jerk_loss.detach()
    rew_ang_vel = rew_scale_ang_vel * ang_rate_loss.detach()
    rew_gate = rew_scale_gate * gate_crossed.float()
    rew_terminal = rew_scale_terminal * terminal
    total_reward = rew_progress + rew_tracking + rew_control + rew_ang_vel + rew_gate + rew_terminal

    task_terms = build_task_terms(
        progress=progress,
        tracking_error=d2g_new.detach(),
        gate_pass=gates_passed,
        collision=collision,
        terminal=terminal,
        control_effort=torch.sum(torch.abs(last_action), dim=-1),
        control_smoothness=jerk_loss.detach(),
        angular_rate=ang_rate_loss.detach(),
        time_penalty=torch.zeros(num_envs, device=device),
    )
    # The differentiable loss: this is what the APG trainer backprops through
    task_terms["loss"] = diff_loss
    task_terms["reward"] = total_reward

    return total_reward, task_terms, updated_gate_index, updated_gates_passed, collision
