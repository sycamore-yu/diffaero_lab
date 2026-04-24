# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class APGConfig:
    """Configuration for APG (Augmented Policy Gradient) algorithm."""

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    rollout_horizon: int = 32
    hidden_dims: tuple[int, ...] = (256, 128, 64)


class DeterministicActor(nn.Module):
    """Simple deterministic actor network for APG."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self._action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    @property
    def action_dim(self) -> int:
        return self._action_dim


class APG:
    """Deterministic APG (Augmented Policy Gradient) for differentiable simulation.

    This implementation consumes IsaacLab-shaped observations:
    - observations[OBS_POLICY]: policy observation tensor
    - observations[OBS_CRITIC]: critic observation tensor (optional)
    - extras[EXTRA_TASK_TERMS]: dict of task-specific reward terms
    - extras[EXTRA_SIM_STATE]: dict of simulation state for differentiable physics
    """

    def __init__(
        self,
        cfg: APGConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cuda:0",
    ):
        self.cfg = cfg
        self.actor = DeterministicActor(obs_dim, action_dim, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.max_grad_norm = cfg.max_grad_norm
        self.rollout_horizon = cfg.rollout_horizon
        self.device = device
        self.actor_loss = torch.zeros(1, device=device)

    def act(self, obs: torch.Tensor, test: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute action from policy observation.

        Args:
            obs: Policy observation tensor of shape (batch, obs_dim)
            test: Whether in test mode (not used for deterministic actor)

        Returns:
            action: Action tensor of shape (batch, action_dim)
            policy_info: Dict with additional info (empty for deterministic)
        """
        action = self.actor(obs)
        return action, {}

    def record_loss(self, loss: torch.Tensor, policy_info: dict[str, Any], env_info: dict[str, Any]) -> None:
        """Accumulate loss for the rollout.

        Args:
            loss: Scalar loss tensor
            policy_info: Dict with policy information (unused)
            env_info: Dict with env information including task_terms
        """
        self.actor_loss += loss.mean()

    def update_actor(self) -> tuple[dict[str, float], dict[str, float]]:
        """Compute gradients and update actor network.

        For non-differentiable physics, uses policy gradient approach.

        Returns:
            losses: Dict with actor loss value
            grad_norms: Dict with gradient norm
        """
        self.actor_loss = self.actor_loss / self.rollout_horizon

        grad_norm = 0.0
        if self.actor_loss.requires_grad:
            self.optimizer.zero_grad()
            self.actor_loss.backward()

            for p in self.actor.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm().item() ** 2
            grad_norm = grad_norm**0.5

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)

            self.optimizer.step()
        else:
            grad_norm = 0.0

        actor_loss = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)

        return {"actor_loss": actor_loss}, {"actor_grad_norm": grad_norm}

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        """Reset actor state for RNN-based actors (no-op for feedforward)."""
        pass

    def detach(self) -> None:
        """Detach tensors for next rollout (no-op for feedforward)."""
        pass
