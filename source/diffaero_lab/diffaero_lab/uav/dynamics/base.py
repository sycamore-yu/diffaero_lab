# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base dynamics interface for UAV dynamics models."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseDynamics(ABC):
    """Abstract base class for UAV dynamics models.

    This class defines the interface that all dynamics models must implement
    for use with DiffAero UAV extension on IsaacLab.
    """

    def __init__(self, cfg: "DictConfig", device: torch.device):
        """Initialize the dynamics model.

        Args:
            cfg: Configuration object with model parameters.
            device: torch device for tensor operations.
        """
        self.cfg = cfg
        self.device = device

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action vector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the dynamics model."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: Tensor) -> None:
        """Reset the dynamics state for specified environments.

        Args:
            env_ids: Tensor of environment IDs to reset.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: Tensor) -> None:
        """Step the dynamics forward with given actions.

        Args:
            actions: Tensor of actions to apply.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs) -> Tensor:
        """Compute dynamics outputs (e.g., wrench, thrust, torque)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def min_action(self) -> Tensor:
        """Minimum action values."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_action(self) -> Tensor:
        """Maximum action values."""
        raise NotImplementedError
