# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

import torch
import warp as wp
from torch import Tensor


class DynamicsBridgeBase(ABC):
    """Bridge interface between DirectRLEnv lifecycle and a specific dynamics backend.

    Subclasses implement this interface for each dynamics model (quad, pmd, etc.).
    The bridge is driven by DirectRLEnv._pre_physics_step() and _apply_action()
    rather than running its own environment loop.
    """

    def __init__(self, cfg: "DroneRacingEnvCfg", robot, num_envs: int, device: str):
        self.cfg = cfg
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        self._action_buf: Tensor | None = None
        self._body_id: Tensor | None = None
        self._robot_weight: float | None = None

    def _ensure_body_wrench_params(self) -> None:
        """Initialize body ids and robot weight for Crazyflie wrench control."""
        if self._body_id is None:
            body_ids = self.robot.find_bodies("body")[0]
            if not torch.is_tensor(body_ids):
                body_ids = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
            self._body_id = body_ids.to(device=self.device, dtype=torch.int32)
        if self._robot_weight is None:
            masses = self._wp_to_torch(self.robot.data.body_mass)
            gravity = torch.tensor(getattr(self.cfg.sim, "gravity", (0.0, 0.0, -9.81)), device=self.device).norm()
            self._robot_weight = (masses[0].sum() * gravity).item()

    def _map_quad_action(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Map normalized [roll, pitch, yaw, thrust] actions to wrench commands."""
        self._ensure_body_wrench_params()
        thrust_to_weight = getattr(self.cfg, "thrust_scale", 1.9)
        moment_scale = getattr(self.cfg, "moment_scale", 0.01)
        thrust = thrust_to_weight * self._robot_weight * (actions[:, 3] + 1.0) / 2.0
        roll = actions[:, 0] * moment_scale
        pitch = actions[:, 1] * moment_scale
        yaw = actions[:, 2] * moment_scale
        return roll, pitch, yaw, thrust

    @abstractmethod
    def reset(self, env_ids: Tensor) -> None:
        """Reset dynamics state for specified environments."""
        raise NotImplementedError

    @abstractmethod
    def process_action(self, actions: Tensor) -> None:
        """Pre-process actions before physics step."""
        raise NotImplementedError

    @abstractmethod
    def apply_to_sim(self) -> None:
        """Apply processed actions to the simulation."""
        raise NotImplementedError

    @abstractmethod
    def read_base_state(self) -> dict[str, Tensor]:
        """Read current drone state from simulation.

        Returns:
            dict with keys: position_w, quaternion_w, linear_velocity_w, angular_velocity_b
        """
        raise NotImplementedError

    @abstractmethod
    def read_motor_state(self) -> dict[str, Tensor]:
        """Read motor state from simulation.

        Returns:
            dict with key: motor_omega
        """
        raise NotImplementedError

    @abstractmethod
    def read_dynamics_info(self) -> dict:
        """Read dynamics model metadata.

        Returns:
            dict with keys: model_name, state_layout_version, tensor_backend, write_mode
        """
        raise NotImplementedError

    def detach(self) -> None:
        """Detach cached bridge tensors at rollout boundaries."""
        for name in ("_action_buf", "_motor_omega", "_thrust_body", "_torque_body"):
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.detach())

    @staticmethod
    def _wp_to_torch(val):
        """Convert warp array to torch tensor, or pass torch tensors through unchanged.

        This lets bridges work with both real Warp-backed robot data and plain-tensor
        FakeRobot objects used in contract tests.
        """
        if torch.is_tensor(val):
            return val
        try:
            return wp.to_torch(val)
        except Exception:
            pass
        return val
