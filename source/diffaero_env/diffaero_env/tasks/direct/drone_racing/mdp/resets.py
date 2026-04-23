# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp
from torch import Tensor

from isaaclab.utils.math import sample_uniform


def reset_body_state(
    robot,
    env_ids: Tensor,
    env_origins: Tensor,
    device: str,
) -> tuple[Tensor, Tensor]:
    joint_pos = wp.to_torch(robot.data.default_joint_pos)[env_ids]
    joint_vel = wp.to_torch(robot.data.default_joint_vel)[env_ids]

    default_root_state = wp.to_torch(robot.data.default_root_state)[env_ids].clone()
    default_root_state[:, :3] += env_origins[env_ids]

    joint_pos[:, :] = 0.0
    joint_vel[:, :] = 0.0

    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    return joint_pos, joint_vel
