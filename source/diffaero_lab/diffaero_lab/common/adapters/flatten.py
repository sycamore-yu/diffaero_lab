# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flatten / unflatten adapters for sim_state dicts."""

import torch
from typing import Dict

from diffaero_lab.common.sim_contract import state_layout


def flatten_sim_state(sim_state: Dict[str, torch.Tensor], model_name: str) -> torch.Tensor:
    """Concatenate all fields of sim_state along the feature axis (dim=1)."""
    layout = state_layout(model_name)
    tensors = []
    for field_name, _ in layout:
        tensors.append(sim_state[field_name])
    return torch.cat(tensors, dim=1)


def unflatten_sim_state(flat: torch.Tensor, model_name: str) -> Dict[str, torch.Tensor]:
    """Reconstruct a sim_state dict from a flattened tensor for model_name."""
    layout = state_layout(model_name)
    result = {}
    offset = 0
    for field_name, num_components in layout:
        result[field_name] = flat[:, offset : offset + num_components].clone()
        offset += num_components
    return result
