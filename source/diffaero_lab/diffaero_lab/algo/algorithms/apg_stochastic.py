# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import math

import torch
import torch.nn as nn


@dataclass
class APGStochasticConfig:
    """Configuration for stochastic APG with Gaussian policy and tanh squashing."""

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    rollout_horizon: int = 32
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    init_log_std: float = 0.0
    entropy_coef: float = 0.0


class GaussianActor(nn.Module):
    """Gaussian actor with reparameterized sampling and tanh squashing.

    The actor predicts mean and log_std, samples using the reparameterization trick,
    and squashes output through tanh to produce bounded actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        init_log_std: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, action_dim)
        self._action_dim = action_dim
        # Learnable log_std initialized near init_log_std
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std for Gaussian distribution.

        Args:
            obs: Observation tensor of shape (batch, obs_dim)

        Returns:
            mean: Action mean of shape (batch, action_dim)
            std: Action std of shape (batch, action_dim)
        """
        h = self.network(obs)
        mean = self.mean_layer(h)
        # Clamp log_std for numerical stability; broadcast to batch
        log_std = torch.clamp(self.log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        return mean, std

    @property
    def action_dim(self) -> int:
        return self._action_dim


class APGStochastic:
    """Stochastic APG using a Gaussian policy with reparameterized sampling and tanh squashing.

    This implementation extends the deterministic APG pattern to support stochastic policies
    while keeping the sampled action connected to differentiable environment losses.
    """

    def __init__(
        self,
        cfg: APGStochasticConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cuda:0",
    ):
        self.cfg = cfg
        self.actor = GaussianActor(
            obs_dim,
            action_dim,
            cfg.hidden_dims,
            cfg.init_log_std,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.max_grad_norm = cfg.max_grad_norm
        self.rollout_horizon = cfg.rollout_horizon
        self.entropy_coef = cfg.entropy_coef
        self.device = device
        self.actor_loss = torch.zeros(1, device=device)

    def act(self, obs: torch.Tensor, test: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample action from Gaussian policy with tanh squashing.

        Args:
            obs: Policy observation tensor of shape (batch, obs_dim)
            test: Whether in test mode (uses mean action without exploration noise)

        Returns:
            action: Squashed action tensor of shape (batch, action_dim)
            policy_info: Dict containing "log_prob" and "entropy"
        """
        mean, std = self.actor(obs)
        if test:
            # Deterministic mean action for evaluation
            raw_action = mean
        else:
            # Reparameterized sampling: action = mean + std * epsilon
            epsilon = torch.randn_like(mean)
            raw_action = mean + std * epsilon

        # Tanh squashing
        action = torch.tanh(raw_action)

        # Log probability with tanh correction (change of variables)
        # log_prob = log_prob_raw - log(1 - action^2 + eps)
        eps = 1e-6
        log_prob_raw = torch.distributions.Normal(mean, std).log_prob(raw_action).sum(dim=-1)
        log_prob_corrected = log_prob_raw - torch.log(1 - action.pow(2) + eps).sum(dim=-1)

        # Entropy of the Gaussian before squashing
        entropy = 0.5 + torch.log(torch.tensor(2 * math.pi, device=self.device)) + torch.log(std)
        entropy = entropy.sum(dim=-1)

        policy_info: dict[str, Any] = {
            "log_prob": log_prob_corrected,
            "entropy": entropy,
            "mean": mean,
            "std": std,
        }
        return action, policy_info

    def record_loss(self, loss: torch.Tensor, policy_info: dict[str, Any], env_info: dict[str, Any]) -> None:
        """Accumulate differentiable rollout loss for direct APG backprop."""
        actor_loss = loss.mean()
        if self.entropy_coef:
            actor_loss = actor_loss - self.entropy_coef * policy_info["entropy"].mean()
        self.actor_loss += actor_loss

    def record_policy_gradient_loss(self, reward: torch.Tensor, policy_info: dict[str, Any]) -> None:
        """Accumulate score-function loss for non-differentiable simulator backends."""
        advantage = reward.detach()
        if advantage.numel() > 1:
            advantage = advantage - advantage.mean()
            std = advantage.std(unbiased=False)
            if std > 1e-6:
                advantage = advantage / (std + 1e-6)
        actor_loss = -(policy_info["log_prob"] * advantage).mean()
        if self.entropy_coef:
            actor_loss = actor_loss - self.entropy_coef * policy_info["entropy"].mean()
        self.actor_loss += actor_loss

    def update_actor(self) -> tuple[dict[str, float], dict[str, float]]:
        """Compute gradients and update actor network.

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
