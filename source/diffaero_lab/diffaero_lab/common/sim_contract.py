# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared simulation contract for Task Scene and Differential Algorithm modules."""

from __future__ import annotations

from typing import Any

import torch

from diffaero_lab.common.capabilities import (
    SUPPORTS_CRITIC_STATE,
    SUPPORTS_DIFFERENTIAL_ROLLOUT,
    SUPPORTS_DYNAMICS_SWITCH,
    SUPPORTS_SIM_STATE,
    SUPPORTS_TASK_TERMS,
    SUPPORTS_TERMINAL_STATE,
    SUPPORTS_WARP_BACKEND,
)

DEFAULT_QUAT_CONVENTION = "xyzw"
DEFAULT_STATE_LAYOUT_VERSION = "1.0"

_STATE_LAYOUTS: dict[str, tuple[tuple[str, int], ...]] = {
    "quad": (
        ("position_w", 3),
        ("quaternion_w", 4),
        ("linear_velocity_w", 3),
        ("angular_velocity_b", 3),
        ("motor_omega", 4),
    ),
    "pmd": (
        ("position_w", 3),
        ("quaternion_w", 4),
        ("linear_velocity_w", 3),
        ("angular_velocity_b", 3),
    ),
    "pmc": (
        ("position_w", 3),
        ("quaternion_w", 4),
        ("linear_velocity_w", 3),
        ("angular_velocity_b", 3),
    ),
    "simple": (
        ("position_w", 3),
        ("quaternion_w", 4),
        ("linear_velocity_w", 3),
        ("angular_velocity_b", 3),
    ),
}


def state_layout(model_name: str) -> tuple[tuple[str, int], ...]:
    if model_name not in _STATE_LAYOUTS:
        raise ValueError(f"Unknown dynamics model '{model_name}'")
    return _STATE_LAYOUTS[model_name]


def default_action_dim(model_name: str) -> int:
    return 4 if model_name in ("quad", "simple") else 3


def build_dynamics_info(
    *,
    model_name: str,
    tensor_backend: str,
    write_mode: str,
    quat_convention: str = DEFAULT_QUAT_CONVENTION,
    state_layout_version: str = DEFAULT_STATE_LAYOUT_VERSION,
    physics_route: str | None = None,
) -> dict[str, Any]:
    info = {
        "model_name": model_name,
        "state_layout_version": state_layout_version,
        "quat_convention": quat_convention,
        "tensor_backend": tensor_backend,
        "write_mode": write_mode,
    }
    if physics_route is not None:
        info["physics_route"] = physics_route
    return info


def build_capabilities(
    *,
    supports_critic_state: bool,
    supports_sim_state: bool = True,
    supports_task_terms: bool = True,
    supports_terminal_state: bool = False,
    supports_differential_rollout: bool = False,
    supports_dynamics_switch: bool = True,
    supports_warp_backend: bool = False,
) -> dict[str, bool]:
    return {
        SUPPORTS_CRITIC_STATE: supports_critic_state,
        SUPPORTS_SIM_STATE: supports_sim_state,
        SUPPORTS_TASK_TERMS: supports_task_terms,
        SUPPORTS_TERMINAL_STATE: supports_terminal_state,
        SUPPORTS_DIFFERENTIAL_ROLLOUT: supports_differential_rollout,
        SUPPORTS_DYNAMICS_SWITCH: supports_dynamics_switch,
        SUPPORTS_WARP_BACKEND: supports_warp_backend,
    }


def build_sim_state(
    *,
    position_w: torch.Tensor,
    quaternion_w: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    angular_velocity_b: torch.Tensor,
    motor_omega: torch.Tensor | None,
    step_count: torch.Tensor,
    last_action: torch.Tensor,
    progress: torch.Tensor,
    target_position_w: torch.Tensor,
    dynamics_info: dict[str, Any],
) -> dict[str, torch.Tensor | dict[str, Any]]:
    sim_state: dict[str, torch.Tensor | dict[str, Any]] = {
        "position_w": position_w,
        "quaternion_w": quaternion_w,
        "linear_velocity_w": linear_velocity_w,
        "angular_velocity_b": angular_velocity_b,
        "step_count": step_count,
        "last_action": last_action,
        "progress": progress,
        "target_position_w": target_position_w,
        "dynamics": dynamics_info,
    }
    if motor_omega is not None:
        sim_state["motor_omega"] = motor_omega
    return sim_state


def build_zero_sim_state(
    *,
    batch_size: int,
    model_name: str = "quad",
    backend: str = "physx",
    write_mode: str = "indexed",
    quat_convention: str = DEFAULT_QUAT_CONVENTION,
    action_dim: int | None = None,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    if action_dim is None:
        action_dim = default_action_dim(model_name)
    state = {
        "position_w": torch.zeros(batch_size, 3),
        "quaternion_w": torch.zeros(batch_size, 4),
        "linear_velocity_w": torch.zeros(batch_size, 3),
        "angular_velocity_b": torch.zeros(batch_size, 3),
        "target_position_w": torch.zeros(batch_size, 3),
        "last_action": torch.zeros(batch_size, action_dim),
        "progress": torch.zeros(batch_size, 1),
        "step_count": torch.zeros(batch_size, 1, dtype=torch.long),
    }
    if model_name == "quad":
        state["motor_omega"] = torch.zeros(batch_size, 4)
    state["dynamics"] = build_dynamics_info(
        model_name=model_name,
        tensor_backend=backend,
        write_mode=write_mode,
        quat_convention=quat_convention,
    )
    return state
