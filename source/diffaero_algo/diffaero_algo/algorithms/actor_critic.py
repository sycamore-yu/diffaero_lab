# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

import math

OBS_POLICY = "policy"
OBS_CRITIC = "critic"

_LOG_2PI = math.log(2 * math.pi)


@dataclass
class SharedActorCriticConfig:
    """Configuration for shared actor-critic with separate observation paths."""

    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    init_log_std: float = 0.0


class GaussianActorHead(nn.Module):
    """Gaussian actor head that predicts mean and log_std for policy distribution."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 128, 64), init_log_std: float = 0.0
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
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std, -20.0, 2.0)
        log_std = log_std.unsqueeze(0).expand(obs.shape[0], -1)
        return mean, log_std

    def sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        epsilon = torch.randn_like(mean)
        raw_action = mean + std * epsilon
        return torch.tanh(raw_action)

    @property
    def action_dim(self) -> int:
        return self._action_dim


class ValueCriticHead(nn.Module):
    """Value critic head for state-value or Q-value estimation."""

    def __init__(self, obs_dim: int, action_dim: int | None = None, hidden_dims: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        self._has_action = action_dim is not None
        self._action_dim = action_dim

        state_value_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            state_value_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        state_value_layers.append(nn.Linear(in_dim, 1))
        self.state_value_net = nn.Sequential(*state_value_layers)

        if action_dim is not None:
            q_value_layers = []
            in_dim = obs_dim + action_dim
            for h_dim in hidden_dims:
                q_value_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
                in_dim = h_dim
            q_value_layers.append(nn.Linear(in_dim, 1))
            self.q_value_net = nn.Sequential(*q_value_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if action is not None and self._has_action:
            return self.q_value_net(torch.cat([obs, action], dim=-1))
        return self.state_value_net(obs)


class SharedActorCritic(nn.Module):
    """Shared actor-critic module with separate policy and critic observation paths.

    This module provides the base architecture for SHAC and SHA2C algorithms,
    with separate input paths for actor (OBS_POLICY) and critic (OBS_CRITIC).
    """

    def __init__(
        self,
        cfg: SharedActorCriticConfig,
        policy_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.policy_obs_dim = policy_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

        self.actor = GaussianActorHead(
            obs_dim=policy_obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            init_log_std=cfg.init_log_std,
        )
        self.critic = ValueCriticHead(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.critic_hidden_dims,
        )

    def actor_act(self, policy_obs: torch.Tensor, with_log_prob: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        mean, log_std = self.actor(policy_obs)
        action = self.actor.sample(mean, log_std)

        policy_info: dict[str, Any] = {
            "mean": mean,
            "log_std": log_std,
        }

        if with_log_prob:
            entropy = 0.5 + 0.5 * _LOG_2PI + log_std
            policy_info["entropy"] = entropy.sum(dim=-1)

        return action, policy_info

    def critic_forward(self, critic_obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(critic_obs, action)
