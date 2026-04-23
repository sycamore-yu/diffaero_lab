# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX-first flatten / unflatten adapters for sim_state dicts.

The QUAD_LAYOUT defines the canonical (field_name, num_components) ordering used
by the quadrotor model in PhysX.  All flatten / unflatten operations are
implemented as simple concatenation so that tensors can be reconstructed
without any metadata beyond the model name.
"""

import torch
from typing import Dict

_QUAD_LAYOUT = (
    ("position_w", 3),
    ("quaternion_w", 4),
    ("linear_velocity_w", 3),
    ("angular_velocity_b", 3),
    ("motor_omega", 4),
)

_LAYOUTS = {
    "quad": _QUAD_LAYOUT,
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


def flatten_sim_state(sim_state: Dict[str, torch.Tensor], model_name: str) -> torch.Tensor:
    """Concatenate all fields of sim_state along the feature axis (dim=1)."""
    layout = _LAYOUTS[model_name]
    tensors = []
    for field_name, _ in layout:
        tensors.append(sim_state[field_name])
    return torch.cat(tensors, dim=1)


def unflatten_sim_state(flat: torch.Tensor, model_name: str) -> Dict[str, torch.Tensor]:
    """Reconstruct a sim_state dict from a flattened tensor for model_name."""
    layout = _LAYOUTS[model_name]
    result = {}
    offset = 0
    for field_name, num_components in layout:
        result[field_name] = flat[:, offset : offset + num_components].clone()
        offset += num_components
    return result
