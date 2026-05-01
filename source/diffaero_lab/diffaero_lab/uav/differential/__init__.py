# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native differentiable drone training core.

This package contains the Newton/Warp differentiable physics stack
that serves as the training core, while IsaacLab provides the task
shell (assets, visualization, evaluation).

Modules:
    model: Standalone Newton differentiable drone model
    kernels: Warp kernels for actor, obs, reward, wrench
    rollout: WarpDroneRollout managing state buffers and tape
"""

from diffaero_lab.uav.differential.model import WarpDroneModel
from diffaero_lab.uav.differential.rollout import RolloutConfig, WarpDroneRollout

__all__ = [
    "WarpDroneModel",
    "WarpDroneRollout",
    "RolloutConfig",
]
