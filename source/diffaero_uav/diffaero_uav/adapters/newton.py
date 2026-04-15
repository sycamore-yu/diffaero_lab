# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton/Warp backend adapter for drone racing with Warp physics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class NewtonBackendAdapter:
    """Warp/Newton backend adapter for drone racing.

    This adapter provides the same interface as PhysX bridges (process_action,
    apply_to_sim, read_base_state, read_motor_state, read_dynamics_info, reset)
    but uses Warp/Newton physics internally.

    The adapter is instantiated via build_newton_adapter() when the environment
    physics backend is configured for Warp/Newton execution.
    """

    def __init__(
        self,
        cfg: Any | None = None,
        robot: Any | None = None,
        num_envs: int = 1,
        device: str = "cpu",
        backend: str = "warp",
    ):
        self.cfg = cfg
        self.robot = robot
        self.device = device
        self.backend = backend
        self._num_envs = num_envs
        self._action_buf: Tensor | None = None
        self._thrust_body: Tensor | None = None
        self._torque_body: Tensor | None = None
        self._body_id: Tensor | None = None
        self._motor_omega: Tensor | None = None

    @staticmethod
    def _wp_to_torch(val):
        """Convert warp array to torch tensor, or pass torch tensors through unchanged.

        This lets the adapter work with both real Warp-backed robot data and
        plain-tensor FakeRobot objects used in contract tests.
        """
        import warp as wp

        if isinstance(val, wp.types.array):
            return wp.to_torch(val)
        return val

    def reset(self, env_ids: Tensor) -> None:
        """Reset dynamics state for specified environments."""
        if self._motor_omega is not None:
            self._motor_omega[env_ids] = 0.0
        if self._thrust_body is not None:
            self._thrust_body[env_ids] = 0.0
        if self._torque_body is not None:
            self._torque_body[env_ids] = 0.0

    def process_action(self, actions: Tensor) -> None:
        """Pre-process actions before physics step."""
        self._action_buf = actions.clone()

    def apply_to_sim(self) -> None:
        """Apply processed actions to the simulation via permanent_wrench_composer."""
        if self._action_buf is None or self.robot is None:
            return
        actions = self._action_buf
        thrust_scale = getattr(self.cfg, "thrust_scale", 1.0) if self.cfg else 1.0
        moment_scale = getattr(self.cfg, "moment_scale", 0.01) if self.cfg else 0.01

        roll = actions[:, 0] * thrust_scale
        pitch = actions[:, 1] * thrust_scale
        yaw = actions[:, 2] * thrust_scale
        thrust = actions[:, 3] * thrust_scale

        if self._thrust_body is None:
            self._thrust_body = torch.zeros(self._num_envs, 1, 3, device=self.device)
            self._torque_body = torch.zeros(self._num_envs, 1, 3, device=self.device)
            self._body_id = torch.zeros(self._num_envs, 1, dtype=torch.int32, device=self.device)

        self._thrust_body[:, 0, 2] = thrust
        self._torque_body[:, 0, 0] = roll * moment_scale
        self._torque_body[:, 0, 1] = pitch * moment_scale
        self._torque_body[:, 0, 2] = yaw * moment_scale

        if hasattr(self.robot, "permanent_wrench_composer"):
            self.robot.permanent_wrench_composer.set_forces_and_torques_index(
                body_ids=self._body_id, forces=self._thrust_body, torques=self._torque_body
            )

    def read_base_state(self) -> dict[str, Tensor]:
        """Read current drone state from simulation.

        Returns:
            dict with keys: position_w, quaternion_w, linear_velocity_w, angular_velocity_b
        """
        if self.robot is not None and hasattr(self.robot, "data"):
            root_pos = self._wp_to_torch(self.robot.data.root_pos_w)
            root_quat = self._wp_to_torch(self.robot.data.root_quat_w)
            root_lin_vel = self._wp_to_torch(self.robot.data.root_lin_vel_w)
            root_ang_vel = self._wp_to_torch(self.robot.data.root_ang_vel_b)
            self._num_envs = root_pos.shape[0]
        else:
            root_pos = torch.zeros(self._num_envs, 3, device=self.device)
            root_quat = torch.zeros(self._num_envs, 4, device=self.device)
            root_lin_vel = torch.zeros(self._num_envs, 3, device=self.device)
            root_ang_vel = torch.zeros(self._num_envs, 3, device=self.device)
        return {
            "position_w": root_pos,
            "quaternion_w": root_quat,
            "linear_velocity_w": root_lin_vel,
            "angular_velocity_b": root_ang_vel,
        }

    def read_motor_state(self) -> dict[str, Tensor]:
        """Read motor state from simulation.

        Returns:
            dict with key: motor_omega
        """
        if self._motor_omega is None:
            self._motor_omega = torch.zeros(self._num_envs, 4, device=self.device)
        return {"motor_omega": self._motor_omega}

    def read_dynamics_info(self) -> dict:
        """Read dynamics model metadata.

        Returns:
            dict with keys: model_name, state_layout_version, tensor_backend, write_mode, quat_convention
        """
        return {
            "model_name": "quad",
            "state_layout_version": "1.0",
            "tensor_backend": "warp",
            "write_mode": "indexed",
            "quat_convention": "wxyz",
        }


def build_newton_adapter(
    cfg: Any | None = None,
    robot: Any | None = None,
    num_envs: int = 1,
    device: str = "cpu",
    backend: str = "warp",
) -> NewtonBackendAdapter:
    """Build and return a NewtonBackendAdapter for Warp/Newton physics execution.

    Args:
        cfg: Configuration object for the adapter (optional).
        robot: Robot articulation object (optional, enables real state reading).
        num_envs: Number of parallel environments.
        device: Device to run computations on (e.g., "cpu", "cuda").
        backend: Physics backend to use ("warp" for Warp/Newton).

    Returns:
        NewtonBackendAdapter instance configured for Warp execution.
    """
    return NewtonBackendAdapter(cfg=cfg, robot=robot, num_envs=num_envs, device=device, backend=backend)
