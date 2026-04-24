# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SHA2C (Soft Actor-Critic with Asymmetric Actor-Critic) algorithm for IsaacLab environments."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffaero_lab.algo.algorithms.actor_critic import ValueCriticHead


@dataclass
class SHA2CConfig:
    """Configuration for SHA2C (Soft Actor-Critic with Asymmetric Actor-Critic) algorithm."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0
    rollout_horizon: int = 32
    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    init_log_std: float = 0.0
    target_entropy: float = -4.0
    soft_update_coef: float = 0.005
    gamma: float = 0.99


class GaussianActor(nn.Module):
    """Gaussian actor head that predicts mean and log_std for policy distribution."""

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


class AsymmetricActor(nn.Module):
    """Asymmetric actor network for SHA2C.

    Takes policy observations and produces action distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.actor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            init_log_std=init_log_std,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs)

    def sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return self.actor.sample(mean, log_std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.actor(obs)
        if deterministic:
            return torch.tanh(mean)
        return self.actor.sample(mean, log_std)


class AsymmetricCritic(nn.Module):
    """Asymmetric critic network for SHA2C.

    Takes critic observations and action as input, produces Q-values.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()
        self.critic = ValueCriticHead(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(obs, action)


class SHA2C:
    """SHA2C (Soft Actor-Critic with Asymmetric Actor-Critic) algorithm.

    Uses separate observation paths for actor and critic:
    - Actor consumes observations["policy"]
    - Critic consumes observations["critic"]
    - Entropy regularization with automatic temperature tuning
    - Target critic network with soft updates for stable Q-learning
    """

    def __init__(
        self,
        cfg: SHA2CConfig,
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

        self.actor = AsymmetricActor(
            obs_dim=policy_obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            init_log_std=cfg.init_log_std,
        ).to(device)

        self.critic = AsymmetricCritic(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.critic_hidden_dims,
        ).to(device)

        self.target_critic = AsymmetricCritic(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.critic_hidden_dims,
        ).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=device))
        self.target_entropy = cfg.target_entropy

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.actor_lr)

        self.rollout_horizon = cfg.rollout_horizon
        self.gamma = cfg.gamma
        self.soft_update_coef = cfg.soft_update_coef
        self.max_grad_norm = cfg.max_grad_norm

        self._rewards_buffer: list[torch.Tensor] = []
        self._values_buffer: list[torch.Tensor] = []
        self._terminated_buffer: list[torch.Tensor] = []
        self._critic_obs_buffer: list[torch.Tensor] = []
        self._action_buffer: list[torch.Tensor] = []
        self._policy_obs_buffer: list[torch.Tensor] = []
        self._n_records = 0

    def actor_act(self, policy_obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute action from policy observation.

        Args:
            policy_obs: Policy observation tensor
            deterministic: If True, use mean action instead of sampling

        Returns:
            action: Action tensor
            policy_info: Dict with mean, log_std, and entropy
        """
        mean, log_std = self.actor(policy_obs)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            action = self.actor.sample(mean, log_std)
            std = torch.exp(log_std)
            # Gaussian log_prob
            gaussian_log_prob = (
                -0.5 * ((action - mean) / (std + 1e-6)) ** 2
                - log_std
                - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=self.device))
            )
            gaussian_log_prob = gaussian_log_prob.sum(dim=-1, keepdim=True)
            # Tanh squash Jacobian correction: log|dx/du| = log(1 - tanh^2) = log(1 - a^2)
            tanh_correction = torch.log(1.0 - action.square() + 1e-6)
            log_prob = gaussian_log_prob + tanh_correction.sum(dim=-1, keepdim=True)

        entropy = 0.5 + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=self.device)) + log_std
        entropy = entropy.sum(dim=-1, keepdim=True).mean()

        policy_info: dict[str, Any] = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": entropy,
        }

        return action, policy_info

    def critic_forward(self, critic_obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Q-value estimate from critic observation and action.

        Args:
            critic_obs: Critic observation tensor
            action: Action tensor (optional)

        Returns:
            Q-value tensor
        """
        return self.critic(critic_obs, action)

    def target_critic_forward(self, critic_obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Compute target Q-value from target critic network.

        Args:
            critic_obs: Critic observation tensor
            action: Action tensor (optional)

        Returns:
            Target Q-value tensor
        """
        return self.target_critic(critic_obs, action)

    def record_transition(
        self,
        critic_obs: torch.Tensor,
        policy_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        batch_extras: dict[str, Any],
    ) -> None:
        self._critic_obs_buffer.append(critic_obs.detach())
        self._policy_obs_buffer.append(policy_obs.detach())
        self._action_buffer.append(action.detach())
        self._rewards_buffer.append(reward.detach())
        self._terminated_buffer.append(terminated.float().detach())
        self._values_buffer.append(value.detach())
        self._n_records += 1

    def record_loss(self, loss: torch.Tensor, policy_info: dict[str, Any]) -> None:
        pass

    def update_critic(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        batch_extras: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if self._n_records < 1:
            return {"critic_loss": 0.0}, {"critic_grad_norm": 0.0}

        alpha = torch.exp(self.log_alpha).item()

        T = len(self._rewards_buffer)
        batch_size = self._rewards_buffer[0].shape[0]

        rewards_flat = torch.stack(
            [r.squeeze(-1) if r.dim() > 1 and r.shape[-1] == 1 else r for r in self._rewards_buffer[:T]], dim=0
        ).view(-1)
        terminated_flat = torch.stack(
            [t.squeeze(-1) if t.dim() > 1 and t.shape[-1] == 1 else t for t in self._terminated_buffer[:T]], dim=0
        ).view(-1)
        critic_obs_flat = torch.cat([obs.unsqueeze(0) for obs in self._critic_obs_buffer[:T]], dim=0).view(
            -1, self.critic_obs_dim
        )
        policy_obs_flat = torch.cat([obs.unsqueeze(0) for obs in self._policy_obs_buffer[:T]], dim=0).view(
            -1, self.policy_obs_dim
        )
        action_flat = torch.cat([a.unsqueeze(0) for a in self._action_buffer[:T]], dim=0).view(-1, self.action_dim)

        with torch.no_grad():
            next_action, next_policy_info = self.actor_act(policy_obs_flat)
            next_q_target = self.target_critic_forward(critic_obs_flat, next_action)
            log_prob = next_policy_info.get("log_prob")
            if log_prob is not None and log_prob.dim() == 1:
                log_prob = log_prob.unsqueeze(-1)
            if log_prob is None:
                log_prob = torch.zeros_like(next_q_target)
            next_value = next_q_target - alpha * log_prob

            next_nonterminal = 1.0 - terminated_flat.unsqueeze(-1)
            q_target = rewards_flat.unsqueeze(-1) + self.gamma * next_nonterminal * next_value

        q_pred = self.critic_forward(critic_obs_flat, action_flat)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)

        self.critic_optimizer.step()

        self._soft_update_target_critic()

        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm.item()}

    def update_actor(self) -> tuple[dict[str, float], dict[str, float]]:
        if self._n_records < 1:
            return {"actor_loss": 0.0, "entropy_loss": 0.0}, {"actor_grad_norm": 0.0}

        alpha = torch.exp(self.log_alpha)

        critic_obs_flat = torch.cat(
            [obs.unsqueeze(0) for obs in self._critic_obs_buffer[: self._n_records]], dim=0
        ).view(-1, self.critic_obs_dim)
        policy_obs_flat = torch.cat(
            [obs.unsqueeze(0) for obs in self._policy_obs_buffer[: self._n_records]], dim=0
        ).view(-1, self.policy_obs_dim)

        sampled_action, policy_info = self.actor_act(policy_obs_flat)
        log_prob = policy_info["log_prob"]

        q_value = self.target_critic_forward(critic_obs_flat, sampled_action)

        actor_loss = (alpha * log_prob - q_value).mean()
        alpha_loss = -self.log_alpha * (log_prob.detach() + self.target_entropy).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {"actor_loss": actor_loss.item(), "entropy_loss": alpha_loss.item()}, {
            "actor_grad_norm": grad_norm.item()
        }

    def _soft_update_target_critic(self) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.soft_update_coef * param.data + (1.0 - self.soft_update_coef) * target_param.data
            )

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        pass

    def detach(self) -> None:
        self._critic_obs_buffer.clear()
        self._policy_obs_buffer.clear()
        self._action_buffer.clear()
        self._rewards_buffer.clear()
        self._terminated_buffer.clear()
        self._values_buffer.clear()
        self._n_records = 0
