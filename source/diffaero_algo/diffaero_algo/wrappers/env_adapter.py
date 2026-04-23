# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment adapter that wraps IsaacLab DirectRLEnv for differential learning.

This adapter:
1. Validates the environment contract (policy/critic observations, task_terms, sim_state)
2. Provides a Batch dataclass for consistent data passing
3. Handles action scaling and environment reset
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from diffaero_common.keys import EXTRA_SIM_STATE, EXTRA_TASK_TERMS, OBS_CRITIC, OBS_POLICY


@dataclass
class Batch:
    """Container for observations and extras from environment."""

    observations: dict[str, torch.Tensor]
    extras: dict[str, Any]


class DifferentialEnvAdapter:
    """Adapter that wraps an IsaacLab DirectRLEnv for differential learning.

    This adapter validates and exposes the environment contract:
    - observations[OBS_POLICY]: policy observation tensor
    - observations[OBS_CRITIC]: critic observation tensor (optional)
    - extras[EXTRA_TASK_TERMS]: dict of task reward terms
    - extras[EXTRA_SIM_STATE]: dict of simulation state for differentiable physics
    """

    def __init__(self, env: gym.Env):
        self.env = env
        unwrapped = getattr(env, "unwrapped", env)
        self.num_envs = getattr(unwrapped, "num_envs", 1)
        self.device = getattr(unwrapped, "device", "cuda:0")
        self.action_space = env.action_space
        self.action_dim = (
            self.action_space.shape[-1] if hasattr(self.action_space, "shape") else int(self.action_space.n)
        )
        high = self.action_space.high
        low = self.action_space.low
        if isinstance(high, torch.Tensor):
            high = high.cpu().numpy()
        if isinstance(low, torch.Tensor):
            low = low.cpu().numpy()
        if np.isinf(high).any() or np.isinf(low).any():
            self._action_scale = None
            self._action_bias = None
        else:
            self._action_scale = torch.tensor(
                (high - low) / 2.0,
                device=self.device,
                dtype=torch.float32,
            )
            self._action_bias = torch.tensor(
                (high + low) / 2.0,
                device=self.device,
                dtype=torch.float32,
            )

    @classmethod
    def make(cls, task_id: str, cfg: Any = None):
        """Factory method to create adapter from task ID.

        Args:
            task_id: Gymnasium task ID
            cfg: Optional environment configuration. If None, loads default config for task.

        Returns:
            DifferentialEnvAdapter instance
        """
        if cfg is None:
            # Use task-specific default config instead of hardcoding PhysX config
            if "Warp" in task_id:
                from diffaero_env.tasks.direct.drone_racing.drone_racing_env_warp_cfg import DroneRacingWarpEnvCfg

                cfg = DroneRacingWarpEnvCfg()
            else:
                from diffaero_env.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

                cfg = DroneRacingEnvCfg()
            cfg.scene.num_envs = 64
        env = gym.make(task_id, cfg=cfg)
        return cls(env)

    def reset(self) -> Batch:
        """Reset environment and return initial batch.

        Returns:
            Batch with observations and extras
        """
        observations, extras = self.env.reset()
        self._validate(observations, extras)
        return Batch(observations=observations, extras=extras)

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute action in environment.

        Args:
            action: Action tensor of shape (num_envs, action_dim)

        Returns:
            observations: Dict of observation tensors
            rewards: Reward tensor of shape (num_envs,)
            terminated: Termination tensor of shape (num_envs,)
            truncated: Truncation tensor of shape (num_envs,)
            extras: Dict with task_terms and sim_state
        """
        action = action.clone()
        action = torch.clamp(action, -1.0, 1.0)
        action = action.detach()
        observations, rewards, terminated, truncated, extras = self.env.step(action)
        self._validate(observations, extras)
        return observations, rewards, terminated, truncated, extras

    def _validate(self, observations: dict[str, torch.Tensor], extras: dict[str, Any]) -> None:
        """Validate environment contract.

        Args:
            observations: Observations dict from env
            extras: Extras dict from env

        Raises:
            AssertionError: If contract fields are missing
        """
        assert OBS_POLICY in observations, f"observations must contain '{OBS_POLICY}'"
        assert OBS_CRITIC in observations, f"observations must contain '{OBS_CRITIC}'"
        if extras:
            assert EXTRA_TASK_TERMS in extras, f"extras must contain '{EXTRA_TASK_TERMS}'"
            assert EXTRA_SIM_STATE in extras, f"extras must contain '{EXTRA_SIM_STATE}'"

    def get_policy_action(self) -> torch.Tensor:
        """Get zero action tensor of correct shape for policy.

        Returns:
            Zero action tensor of shape (num_envs, action_dim)
        """
        return torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Rescale action from [-1, 1] to environment's action space.

        Args:
            action: Normalized action tensor

        Returns:
            Rescaled action tensor
        """
        if self._action_scale is None:
            return action
        return self._action_scale * action + self._action_bias

    def close(self) -> None:
        """Close the environment."""
        self.env.close()
