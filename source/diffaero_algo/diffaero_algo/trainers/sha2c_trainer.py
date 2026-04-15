# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SHA2CTrainer for SHA2C algorithm."""

import torch

from diffaero_algo.algorithms.sha2c import SHA2C, SHA2CConfig
from diffaero_algo.wrappers.env_adapter import DifferentialEnvAdapter


class SHA2CTrainer:
    """Trainer for SHA2C algorithm.

    SHA2C uses:
    - observations["policy"] for the actor
    - observations["critic"] for the critic network
    - Target critic with soft updates for stable Q-learning
    - Automatic entropy temperature tuning
    """

    def __init__(self, env: DifferentialEnvAdapter, cfg: SHA2CConfig):
        self.env = env
        self.cfg = cfg

        batch = env.reset()
        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = env.action_dim

        self.sha2c = SHA2C(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=env.device,
        )
        self.rollout_horizon = cfg.rollout_horizon

    def train(self, max_iterations: int = 100) -> None:
        for iteration in range(max_iterations):
            self._rollout()
            critic_losses, critic_grad_norms = self.sha2c.update_critic(
                rewards=torch.zeros(self.env.num_envs, device=self.env.device),
                terminated=torch.zeros(self.env.num_envs, device=self.env.device),
                batch_extras={},
            )
            actor_losses, actor_grad_norms = self.sha2c.update_actor()
            self.sha2c.detach()

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: actor_loss={actor_losses.get('actor_loss', 0.0):.4f}, "
                    f"critic_loss={critic_losses.get('critic_loss', 0.0):.4f}, "
                    f"grad_norm={actor_grad_norms.get('actor_grad_norm', 0.0):.4f}"
                )

    def _rollout(self) -> None:
        batch = self.env.reset()

        for _ in range(self.rollout_horizon):
            policy_obs = batch.observations["policy"]
            action, policy_info = self.sha2c.actor_act(policy_obs)

            batch.observations, rewards, terminated, truncated, batch.extras = self.env.step(action)

            critic_obs = batch.observations["critic"]
            value = self.sha2c.critic_forward(critic_obs, action)

            self.sha2c.record_transition(
                critic_obs=critic_obs,
                policy_obs=policy_obs,
                action=action,
                reward=rewards,
                terminated=terminated,
                value=value,
                batch_extras=batch.extras,
            )

            if "progress" in batch.extras.get("task_terms", {}):
                loss = -batch.extras["task_terms"]["progress"].mean()
            else:
                loss = -rewards.mean()

            self.sha2c.record_loss(loss, policy_info)
