# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from torch import Tensor

from diffaero_lab.common.sim_contract import build_sim_state as build_shared_sim_state


def build_sim_state(
    position_w: Tensor,
    quaternion_w: Tensor,
    linear_velocity_w: Tensor,
    angular_velocity_b: Tensor,
    motor_omega: Tensor,
    step_count: Tensor,
    last_action: Tensor,
    progress: Tensor,
    target_position_w: Tensor,
    dynamics_info: dict,
) -> dict[str, Tensor | dict]:
    return build_shared_sim_state(
        position_w=position_w,
        quaternion_w=quaternion_w,
        linear_velocity_w=linear_velocity_w,
        angular_velocity_b=angular_velocity_b,
        motor_omega=motor_omega,
        step_count=step_count,
        last_action=last_action,
        progress=progress,
        target_position_w=target_position_w,
        dynamics_info=dynamics_info,
    )
