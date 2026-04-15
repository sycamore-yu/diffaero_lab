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

    @staticmethod
    def _wp_to_torch(val):
        """Convert warp array to torch tensor, or pass torch tensors through unchanged.

        This lets bridges work with both real Warp-backed robot data and plain-tensor
        FakeRobot objects used in contract tests.
        """
        if isinstance(val, wp.types.array):
            return wp.to_torch(val)
        return val
