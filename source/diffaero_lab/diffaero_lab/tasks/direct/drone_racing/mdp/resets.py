# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp
from torch import Tensor


def reset_body_state(
    robot,
    env_ids_sim: Tensor,
    env_origins: Tensor,
    device: str,
    root_position_w: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    env_ids_index = env_ids_sim.to(device=device, dtype=torch.long)
    joint_pos = wp.to_torch(robot.data.default_joint_pos)[env_ids_index]
    joint_vel = wp.to_torch(robot.data.default_joint_vel)[env_ids_index]

    default_root_state = wp.to_torch(robot.data.default_root_state)[env_ids_index].clone()
    default_root_state[:, :3] += env_origins[env_ids_index]
    if root_position_w is not None:
        default_root_state[:, :3] = root_position_w

    joint_pos[:, :] = 0.0
    joint_vel[:, :] = 0.0

    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_sim)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_sim)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_sim)

    return joint_pos, joint_vel
