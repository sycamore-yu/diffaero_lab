# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""sim_state builder producing PhysX/Warp-shaped state dicts with dynamics metadata."""

import torch
from typing import Dict, Literal

_STATE_LAYOUT_QUAD = (
    "position_w",
    "quaternion_w",
    "linear_velocity_w",
    "angular_velocity_b",
    "motor_omega",
)


def build_sim_state(
    batch_size: int, model_name: str = "quad", backend: Literal["physx", "warp"] = "physx"
) -> Dict[str, torch.Tensor]:
    """Return a sim_state dict with correctly shaped zero tensors and dynamics metadata.

    Args:
        batch_size: number of environment instances.
        model_name: "quad", "pmd", or "pmc".
        backend: Physics backend - "physx" (IsaacLab/PhysX) or "warp" (Newton/Warp).

    Returns:
        Dict mapping field names to (batch_size, N) tensors.
    """
    if model_name == "quad":
        state = {
            "position_w": torch.zeros(batch_size, 3),
            "quaternion_w": torch.zeros(batch_size, 4),
            "linear_velocity_w": torch.zeros(batch_size, 3),
            "angular_velocity_b": torch.zeros(batch_size, 3),
            "motor_omega": torch.zeros(batch_size, 4),
            "target_position_w": torch.zeros(batch_size, 3),
            "last_action": torch.zeros(batch_size, 4),
            "progress": torch.zeros(batch_size, 1),
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long),
        }
        state["dynamics"] = {
            "model_name": "quad",
            "state_layout_version": "1.0",
            "quat_convention": "wxyz",
            "tensor_backend": backend,
            "write_mode": "indexed",
        }
    elif model_name in ("pmd", "pmc", "simple"):
        state = {
            "position_w": torch.zeros(batch_size, 3),
            "quaternion_w": torch.zeros(batch_size, 4),
            "linear_velocity_w": torch.zeros(batch_size, 3),
            "angular_velocity_b": torch.zeros(batch_size, 3),
            "target_position_w": torch.zeros(batch_size, 3),
            "last_action": torch.zeros(batch_size, 3),
            "progress": torch.zeros(batch_size, 1),
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long),
        }
        state["dynamics"] = {
            "model_name": model_name,
            "state_layout_version": "1.0",
            "quat_convention": "wxyz",
            "tensor_backend": backend,
            "write_mode": "indexed",
        }
    else:
        raise ValueError(f"Unknown model_name {model_name!r}")
    return state
