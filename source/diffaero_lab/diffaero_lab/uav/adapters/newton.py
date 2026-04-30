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
        self._robot_weight: float | None = None
        self._motor_omega: Tensor | None = None

    def _ensure_body_wrench_params(self) -> None:
        if self._body_id is None:
            if self.robot is not None and hasattr(self.robot, "find_bodies"):
                body_ids = None
                for body_name in ("body", "base_link"):
                    try:
                        candidate = self.robot.find_bodies(body_name)[0]
                    except ValueError:
                        continue
                    if len(candidate) > 0:
                        body_ids = candidate
                        break
                if body_ids is None:
                    body_ids = [0]
                if not torch.is_tensor(body_ids):
                    body_ids = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
                self._body_id = body_ids.to(device=self.device, dtype=torch.int32)
            else:
                self._body_id = torch.zeros(1, dtype=torch.int32, device=self.device)
        if self._robot_weight is None:
            if self.robot is not None and hasattr(self.robot, "data") and hasattr(self.robot.data, "body_mass"):
                masses = self._wp_to_torch(self.robot.data.body_mass)
                gravity_cfg = getattr(getattr(self.cfg, "sim", None), "gravity", (0.0, 0.0, -9.81))
                gravity = torch.tensor(gravity_cfg, device=self.device).norm()
                self._robot_weight = (masses[0].sum() * gravity).item()
            else:
                self._robot_weight = 1.0

    def _map_quad_action(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._ensure_body_wrench_params()
        thrust_to_weight = getattr(self.cfg, "thrust_scale", 1.9) if self.cfg else 1.9
        moment_scale = getattr(self.cfg, "moment_scale", 0.01) if self.cfg else 0.01
        roll = actions[:, 0] * moment_scale
        pitch = actions[:, 1] * moment_scale
        yaw = actions[:, 2] * moment_scale
        thrust = thrust_to_weight * self._robot_weight * (actions[:, 3] + 1.0) / 2.0
        return roll, pitch, yaw, thrust

    def _compute_motor_omega(self, actions: Tensor) -> None:
        if self._motor_omega is None or self._motor_omega.shape[0] != actions.shape[0]:
            self._motor_omega = torch.zeros(actions.shape[0], 4, device=self.device)

        roll, pitch, yaw, thrust = self._map_quad_action(actions)

        self._motor_omega[:, 0] = thrust + roll * 0.3 + pitch * 0.3
        self._motor_omega[:, 1] = thrust - roll * 0.3 - pitch * 0.3
        self._motor_omega[:, 2] = thrust - roll * 0.3 + pitch * 0.3
        self._motor_omega[:, 3] = thrust + roll * 0.3 - pitch * 0.3
        self._motor_omega[:, 0] += yaw * 0.5
        self._motor_omega[:, 1] += yaw * 0.5
        self._motor_omega[:, 2] -= yaw * 0.5
        self._motor_omega[:, 3] -= yaw * 0.5

    @staticmethod
    def _wp_to_torch(val):
        """Convert warp array to torch tensor, or pass torch tensors through unchanged.

        This lets the adapter work with both real Warp-backed robot data and
        plain-tensor FakeRobot objects used in contract tests.
        """
        import warp as wp

        if torch.is_tensor(val):
            return val
        try:
            return wp.to_torch(val)
        except Exception:
            pass
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
        self._action_buf = actions.clone().detach()
        self._compute_motor_omega(actions)

    def apply_to_sim(self) -> None:
        """Apply processed actions to the simulation via permanent_wrench_composer."""
        if self._action_buf is None:
            return
        actions = self._action_buf
        if self._thrust_body is None:
            self._thrust_body = torch.zeros(self._num_envs, 1, 3, device=self.device)
            self._torque_body = torch.zeros(self._num_envs, 1, 3, device=self.device)
        roll, pitch, yaw, thrust = self._map_quad_action(actions)

        self._compute_motor_omega(actions)

        self._thrust_body[:, 0, 2] = thrust
        self._torque_body[:, 0, 0] = roll
        self._torque_body[:, 0, 1] = pitch
        self._torque_body[:, 0, 2] = yaw

        if self.robot is not None and hasattr(self.robot, "permanent_wrench_composer"):
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
            "quat_convention": "xyzw",
        }

    def detach(self) -> None:
        """Detach cached adapter tensors at rollout boundaries."""
        for name in ("_action_buf", "_motor_omega", "_thrust_body", "_torque_body"):
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.detach())


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
