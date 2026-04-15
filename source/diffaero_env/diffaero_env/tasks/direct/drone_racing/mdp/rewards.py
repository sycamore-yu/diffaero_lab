# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_common.terms import (
    TERM_ANGULAR_RATE,
    TERM_CONTROL_EFFORT,
    TERM_CONTROL_SMOOTHNESS,
    TERM_GATE_PASS,
    TERM_PROGRESS,
    TERM_TERMINAL,
    TERM_TIME_PENALTY,
    TERM_TRACKING_ERROR,
)
from diffaero_env.tasks.direct.drone_racing.state.task_terms import build_task_terms


def compute_rewards(
    rew_scale_progress: float,
    rew_scale_tracking: float,
    rew_scale_control_effort: float,
    rew_scale_ang_vel: float,
    rew_scale_terminal: float,
    rew_scale_gate: float,
    angular_velocity_b: Tensor,
    last_action: Tensor,
    prev_action: Tensor,
    reset_terminated: Tensor,
    step_count: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    progress = torch.ones_like(angular_velocity_b[:, 0]) * 0.0
    tracking_error = torch.zeros_like(angular_velocity_b[:, 0])
    gate_pass = torch.zeros_like(angular_velocity_b[:, 0])
    collision = torch.zeros_like(angular_velocity_b[:, 0])
    terminal = reset_terminated.float()
    control_effort = torch.sum(torch.abs(last_action), dim=-1)
    control_smoothness = torch.sum(torch.abs(last_action - prev_action), dim=-1)
    angular_rate = torch.sum(torch.abs(angular_velocity_b), dim=-1)
    time_penalty = torch.ones_like(angular_velocity_b[:, 0]) * -0.001

    # Phase 1: progress, tracking_error, gate_pass, and collision are placeholder zeros.
    # Real gate-tracking and collision logic will be added in a follow-up phase.
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

    return total_reward, task_terms
