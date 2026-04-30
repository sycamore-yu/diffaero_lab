# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_lab.common.keys import OBS_CRITIC, OBS_POLICY
from diffaero_lab.tasks.direct.drone_racing.state.critic import build_critic_obs
from diffaero_lab.tasks.direct.drone_racing.state.policy import build_policy_obs


def compute_observations(
    bridge_state: dict[str, Tensor],
    last_action: Tensor,
    enable_critic: bool,
    target_position_w: Tensor,
    target_yaw: Tensor,
    next_target_position_w: Tensor,
    next_target_yaw: Tensor,
) -> dict[str, Tensor]:
    base = bridge_state
    policy_obs = build_policy_obs(
        position_w=base["position_w"],
        quaternion_w=base["quaternion_w"],
        linear_velocity_w=base["linear_velocity_w"],
        angular_velocity_b=base["angular_velocity_b"],
        last_action=last_action,
        target_position_w=target_position_w,
        target_yaw=target_yaw,
        next_target_position_w=next_target_position_w,
        next_target_yaw=next_target_yaw,
    )
    obs = {OBS_POLICY: policy_obs}
    if enable_critic:
        critic_obs = build_critic_obs(
            position_w=base["position_w"],
            quaternion_w=base["quaternion_w"],
            linear_velocity_w=base["linear_velocity_w"],
            angular_velocity_b=base["angular_velocity_b"],
            target_position_w=target_position_w,
            target_yaw=target_yaw,
            next_target_position_w=next_target_position_w,
            next_target_yaw=next_target_yaw,
        )
        obs[OBS_CRITIC] = critic_obs
    return obs
