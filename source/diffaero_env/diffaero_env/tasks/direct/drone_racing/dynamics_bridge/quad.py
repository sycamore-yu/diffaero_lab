# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp
from torch import Tensor

from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.base import DynamicsBridgeBase


class QuadDynamicsBridge(DynamicsBridgeBase):
    """PhysX quadrotor bridge driven by DirectRLEnv hooks.

    Implements the dynamics bridge for a quadrotor using the CRAZYFLIE asset.
    Action is [thrust_scale * roll, thrust_scale * pitch, thrust_scale * yaw, thrust_scale * thrust].
    """

    def __init__(self, cfg: "DroneRacingEnvCfg", robot, num_envs: int, device: str):
        super().__init__(cfg, robot, num_envs, device)
        self.thrust_scale = cfg.thrust_scale
        self.moment_scale = cfg.moment_scale
        # Phase 1: motor_omega is always zero. No per-motor physics or motor model is simulated.
        self._motor_omega = torch.zeros(num_envs, 4, device=device)
        self._thrust_body = torch.zeros(num_envs, 1, 3, device=device)
        self._torque_body = torch.zeros(num_envs, 1, 3, device=device)
        # Body index 0 = root body of Crazyflie SDF (single-body articulation)
        self._body_id = torch.zeros(num_envs, 1, dtype=torch.int32, device=device)

    def reset(self, env_ids: Tensor) -> None:
        self._motor_omega[env_ids] = 0.0
        self._thrust_body[env_ids] = 0.0
        self._torque_body[env_ids] = 0.0

    def process_action(self, actions: Tensor) -> None:
        self._action_buf = actions.clone()

    def apply_to_sim(self) -> None:
        if self._action_buf is None:
            return
        actions = self._action_buf
        roll = actions[:, 0] * self.thrust_scale
        pitch = actions[:, 1] * self.thrust_scale
        yaw = actions[:, 2] * self.thrust_scale
        thrust = actions[:, 3] * self.thrust_scale

        self._thrust_body[:, 0, 2] = thrust
        self._torque_body[:, 0, 0] = roll * self.moment_scale
        self._torque_body[:, 0, 1] = pitch * self.moment_scale
        self._torque_body[:, 0, 2] = yaw * self.moment_scale

        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            body_ids=self._body_id, forces=self._thrust_body, torques=self._torque_body
        )

        # Compute motor_omega from control action using simplified quadrotor mixing.
        # Motor layout (X-config): front-right (0), rear-left (1), front-left (2), rear-right (3)
        # Basic mixing: base spin + roll/pitch corrections + yaw differential
        # Phase 3 Task 4: meaningful motor-state derived from control action, not physics.
        self._motor_omega[:, 0] = thrust + roll * 0.3 + pitch * 0.3  # FR motor
        self._motor_omega[:, 1] = thrust - roll * 0.3 - pitch * 0.3  # RL motor
        self._motor_omega[:, 2] = thrust - roll * 0.3 + pitch * 0.3  # FL motor
        self._motor_omega[:, 3] = thrust + roll * 0.3 - pitch * 0.3  # RR motor
        # Yaw correction: differential spin between CW and CCW motor pairs
        self._motor_omega[:, 0] += yaw * self.moment_scale * 0.5
        self._motor_omega[:, 1] += yaw * self.moment_scale * 0.5
        self._motor_omega[:, 2] -= yaw * self.moment_scale * 0.5
        self._motor_omega[:, 3] -= yaw * self.moment_scale * 0.5

    def read_base_state(self) -> dict[str, Tensor]:
        root_pos = wp.to_torch(self.robot.data.root_pos_w)
        root_quat = wp.to_torch(self.robot.data.root_quat_w)
        root_lin_vel = wp.to_torch(self.robot.data.root_lin_vel_w)
        root_ang_vel = wp.to_torch(self.robot.data.root_ang_vel_b)
        return {
            "position_w": root_pos,
            "quaternion_w": root_quat,
            "linear_velocity_w": root_lin_vel,
            "angular_velocity_b": root_ang_vel,
        }

    def read_motor_state(self) -> dict[str, Tensor]:
        return {"motor_omega": self._motor_omega}

    def read_dynamics_info(self) -> dict:
        return {
            "model_name": "quad",
            "state_layout_version": "1.0",
            "quat_convention": "wxyz",
            "tensor_backend": "torch",
            "write_mode": "indexed",
        }
