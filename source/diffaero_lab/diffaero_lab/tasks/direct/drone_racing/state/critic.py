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
    target_position_w: Tensor,
    target_yaw: Tensor,
    next_target_position_w: Tensor,
    next_target_yaw: Tensor,
) -> Tensor:
    from diffaero_lab.tasks.direct.drone_racing.state.policy import build_policy_obs

    zero_action = torch.zeros(position_w.shape[0], 0, device=position_w.device, dtype=position_w.dtype)
    return build_policy_obs(
        position_w=position_w,
        quaternion_w=quaternion_w,
        linear_velocity_w=linear_velocity_w,
        angular_velocity_b=angular_velocity_b,
        last_action=zero_action,
        target_position_w=target_position_w,
        target_yaw=target_yaw,
        next_target_position_w=next_target_position_w,
        next_target_yaw=next_target_yaw,
    )
