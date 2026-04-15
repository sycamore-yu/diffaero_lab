# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SHAC (Soft Hierarchical Actor-Critic) algorithm for IsaacLab environments."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from diffaero_algo.algorithms.actor_critic import (
    SharedActorCritic,
    SharedActorCriticConfig,
    ValueCriticHead,
)
from diffaero_common.keys import EXTRA_TASK_TERMS


@dataclass
class SHACConfig:
    """Configuration for SHAC (Soft Hierarchical Actor-Critic) algorithm."""

    lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0
    rollout_horizon: int = 32
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lmbda: float = 0.95


class CriticNetwork(nn.Module):
    """Critic network for SHAC value estimation."""

    def __init__(
        self,
        critic_obs_dim: int,
        action_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.network = ValueCriticHead(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )

    def forward(self, critic_obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(critic_obs, action)


class SHAC:
    """SHAC (Soft Hierarchical Actor-Critic) algorithm.

    Uses separate observation paths for actor and critic:
    - Actor consumes observations["policy"]
    - Critic consumes observations["critic"]
    - Entropy regularization for exploration
    - Value bootstrap for advantage estimation via GAE
    """

    def __init__(
        self,
        cfg: SHACConfig,
        policy_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cuda:0",
    ):
        self.cfg = cfg
        self.policy_obs_dim = policy_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.device = device

        shared_cfg = SharedActorCriticConfig(
            actor_hidden_dims=cfg.hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
        )
        self.ac = SharedActorCritic(
            cfg=shared_cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
        ).to(device)

        self.actor_optim = torch.optim.Adam(self.ac.actor.parameters(), lr=cfg.lr)
        self.critic_optim = torch.optim.Adam(self.ac.critic.parameters(), lr=cfg.critic_lr)

        self.rollout_horizon = cfg.rollout_horizon
        self.gamma = cfg.gamma
        self.lmbda = cfg.lmbda
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm

        self._values_buffer: list[torch.Tensor] = []
        self._critic_obs_buffer: list[torch.Tensor] = []
        self._actor_loss = torch.tensor(0.0, device=device)
        self._critic_loss = torch.tensor(0.0, device=device)
        self._entropy_loss = torch.tensor(0.0, device=device)
        self._n_records = 0

        self._rewards_buffer: list[torch.Tensor] = []
        self._terminated_buffer: list[torch.Tensor] = []

    def actor_act(self, policy_obs: torch.Tensor, test: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute action from policy observation."""
        return self.ac.actor_act(policy_obs, with_log_prob=True)

    def critic_forward(self, critic_obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Compute value estimate from critic observation."""
        return self.ac.critic_forward(critic_obs, action)

    def record_value(self, critic_obs: torch.Tensor, value: torch.Tensor) -> None:
        """Record critic observation and value estimate for advantage computation."""
        self._critic_obs_buffer.append(critic_obs.detach())
        self._values_buffer.append(value.detach())
        self._n_records += 1

    def record_loss(
        self,
        loss: torch.Tensor,
        policy_info: dict[str, Any],
        extras: dict[str, Any],
        terminated: torch.Tensor | None = None,
    ) -> None:
        """Record loss for actor update with value bootstrap."""
        self._actor_loss = self._actor_loss + loss.mean()

        if "entropy" in policy_info:
            entropy = policy_info["entropy"].mean()
            self._entropy_loss = self._entropy_loss - entropy

        if EXTRA_TASK_TERMS in extras:
            task_terms = extras[EXTRA_TASK_TERMS]
            if "reward" in task_terms:
                self._rewards_buffer.append(task_terms["reward"].detach())
            elif "final_reward" in task_terms:
                self._rewards_buffer.append(task_terms["final_reward"].detach())

        if terminated is not None:
            self._terminated_buffer.append(terminated.float().detach())

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Perform one update step for actor and critic."""
        advantages, target_values = self._compute_gae()

        actor_losses, actor_grad_norms = self._update_actor(advantages)
        critic_losses, critic_grad_norms = self._update_critic(target_values)

        losses = {**actor_losses, **critic_losses}
        grad_norms = {**actor_grad_norms, **critic_grad_norms}

        self._reset_buffers()

        return losses, grad_norms

    def _compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation."""
        if len(self._values_buffer) < 2:
            T = max(len(self._values_buffer), 1)
            batch_size = self._values_buffer[0].shape[0] if self._values_buffer else 1
            return (
                torch.zeros(T, batch_size, device=self.device),
                torch.zeros(T, batch_size, device=self.device),
            )

        T = len(self._values_buffer)
        batch_size = self._values_buffer[0].shape[0]

        values_list = []
        for v in self._values_buffer[:T]:
            if v.dim() == 0:
                values_list.append(v.unsqueeze(0).unsqueeze(0).expand(T, batch_size))
            elif v.dim() == 1:
                values_list.append(v.unsqueeze(0).expand(T, -1))
            else:
                values_list.append(v)
        values = torch.stack([v.squeeze(-1) if v.dim() > 1 and v.shape[-1] == 1 else v for v in values_list], dim=0)

        if values.dim() == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.dim() == 2 and values.shape[-1] != batch_size:
            values = values.squeeze(-1) if values.dim() > 1 else values.unsqueeze(-1).expand(T, batch_size)

        if values.dim() == 1:
            values = values.unsqueeze(-1).expand(T, batch_size)

        rewards = torch.zeros(T, batch_size, device=self.device)
        for i, r in enumerate(self._rewards_buffer[:T]):
            if r.dim() == 0:
                rewards[i] = r.unsqueeze(0).expand(batch_size)
            elif r.dim() == 1:
                rewards[i] = r
            else:
                rewards[i] = r.squeeze(-1) if r.shape[-1] == 1 else r

        terminated = torch.zeros(T, batch_size, device=self.device)
        for i, term in enumerate(self._terminated_buffer[:T]):
            if term.dim() == 0:
                terminated[i] = term.unsqueeze(0).expand(batch_size)
            elif term.dim() == 1:
                terminated[i] = term
            else:
                terminated[i] = term.squeeze(-1) if term.shape[-1] == 1 else term

        advantages = torch.zeros_like(values)
        lastgaelam = 0

        for t in reversed(range(T - 1)):
            nextnonterminal = 1.0 - terminated[t]
            nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lmbda * nextnonterminal * lastgaelam

        target_values = advantages + values

        return advantages, target_values

    def _update_actor(self, advantages: torch.Tensor) -> tuple[dict[str, float], dict[str, float]]:
        """Update actor network."""
        actor_loss = self._actor_loss / max(self._n_records, 1)
        entropy_loss = self._entropy_loss / max(self._n_records, 1)
        total_loss = actor_loss + self.entropy_coef * entropy_loss

        self.actor_optim.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), max_norm=self.max_grad_norm)

        self.actor_optim.step()

        return {"actor_loss": actor_loss.item(), "entropy_loss": entropy_loss.item()}, {
            "actor_grad_norm": grad_norm.item()
        }

    def _update_critic(self, target_values: torch.Tensor) -> tuple[dict[str, float], dict[str, float]]:
        """Update critic network using MSE loss against GAE target values."""
        if len(self._critic_obs_buffer) < 2:
            return {"critic_loss": 0.0}, {"critic_grad_norm": 0.0}

        T = len(self._critic_obs_buffer)
        batch_size = self._critic_obs_buffer[0].shape[0]

        critic_obs_flat = torch.cat([obs.unsqueeze(0) for obs in self._critic_obs_buffer[:T]], dim=0).view(
            -1, self.critic_obs_dim
        )
        target_values_flat = target_values[: T * batch_size].view(-1)

        predicted_values = self.critic_forward(critic_obs_flat).squeeze(-1)

        critic_loss = torch.nn.functional.mse_loss(predicted_values, target_values_flat)

        self.critic_optim.zero_grad()
        critic_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), max_norm=self.max_grad_norm)

        self.critic_optim.step()

        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm.item()}

    def _reset_buffers(self) -> None:
        """Reset all accumulated buffers after an update."""
        self._values_buffer.clear()
        self._critic_obs_buffer.clear()
        self._rewards_buffer.clear()
        self._terminated_buffer.clear()
        self._actor_loss = self._actor_loss.detach().fill_(0.0)
        self._critic_loss = self._critic_loss.detach().fill_(0.0)
        self._entropy_loss = self._entropy_loss.detach().fill_(0.0)
        self._n_records = 0

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        """Reset actor state for RNN-based actors (no-op for feedforward)."""
        pass

    def detach(self) -> None:
        """Detach tensors for next rollout."""
        pass
