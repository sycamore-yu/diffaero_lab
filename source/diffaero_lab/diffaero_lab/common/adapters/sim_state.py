# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared zero-state builder for sim_state contracts."""

from typing import Dict, Literal

import torch

from diffaero_lab.common.sim_contract import build_zero_sim_state


def build_sim_state(
    batch_size: int,
    model_name: str = "quad",
    backend: Literal["physx", "warp"] = "physx",
) -> Dict[str, torch.Tensor]:
    return build_zero_sim_state(batch_size=batch_size, model_name=model_name, backend=backend)
