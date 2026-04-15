# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor


def build_critic_obs(
    position_w: Tensor,
    quaternion_w: Tensor,
    linear_velocity_w: Tensor,
    angular_velocity_b: Tensor,
) -> Tensor:
    return torch.cat(
        [
            position_w,
            quaternion_w,
            linear_velocity_w,
            angular_velocity_b,
        ],
        dim=-1,
    )
