# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor


def build_policy_obs(
    position_w: Tensor,
    quaternion_w: Tensor,
    linear_velocity_w: Tensor,
    angular_velocity_b: Tensor,
    target_position_w: Tensor,
    target_yaw: Tensor,
    next_target_position_w: Tensor,
    next_target_yaw: Tensor,
    last_action: Tensor | None = None,
) -> Tensor:
    from diffaero_lab.tasks.direct.drone_racing.mdp.gates import gate_frame_state, gate_rotmat_w2g, transform_w_to_gate, wrap_pi

    pos_g, vel_g, rpy_g = gate_frame_state(
        position_w=position_w,
        velocity_w=linear_velocity_w,
        quaternion_xyzw=quaternion_w,
        gate_position_w=target_position_w,
        gate_yaw=target_yaw,
    )
    next_gate_rel_pos = transform_w_to_gate(gate_rotmat_w2g(target_yaw), next_target_position_w - target_position_w)
    next_gate_rel_yaw = wrap_pi(next_target_yaw - target_yaw).unsqueeze(-1)
    return torch.cat(
        [
            pos_g,
            vel_g,
            rpy_g,
            next_gate_rel_pos,
            next_gate_rel_yaw,
        ],
        dim=-1,
    )
