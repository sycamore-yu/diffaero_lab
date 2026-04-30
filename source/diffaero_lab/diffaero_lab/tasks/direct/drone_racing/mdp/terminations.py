# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_lab.tasks.direct.drone_racing.mdp.gates import gate_crossing, out_of_bounds


def compute_dones(
    episode_length_buf: Tensor,
    max_episode_length: int,
    prev_position_w: Tensor,
    position_w: Tensor,
    target_position_w: Tensor,
    target_yaw: Tensor,
    env_origins: Tensor,
    gate_l1_radius: float,
    oob_xy_limit: float,
    oob_z_max: float,
) -> tuple[Tensor, Tensor]:
    time_out = episode_length_buf >= max_episode_length - 1
    _, gate_collision = gate_crossing(
        prev_position_w=prev_position_w,
        position_w=position_w,
        gate_position_w=target_position_w,
        gate_yaw=target_yaw,
        gate_l1_radius=gate_l1_radius,
    )
    truncated = time_out | out_of_bounds(position_w, env_origins, oob_xy_limit, oob_z_max)
    return gate_collision, truncated
