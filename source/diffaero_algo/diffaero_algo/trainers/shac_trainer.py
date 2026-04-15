# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SHAC trainer that orchestrates rollout and backward pass with value bootstrap."""

import torch

from diffaero_algo.algorithms.shac import SHAC, SHACConfig
from diffaero_algo.wrappers.env_adapter import DifferentialEnvAdapter
from diffaero_common.keys import EXTRA_TASK_TERMS


class SHACTrainer:
    """Trainer that orchestrates SHAC rollout and backward pass.

    SHAC uses:
    - observations["policy"] for the actor
    - observations["critic"] for the critic network
    - Value bootstrap via critic estimates for advantage estimation
    """

    def __init__(self, env: DifferentialEnvAdapter, cfg: SHACConfig):
        self.env = env
        self.cfg = cfg

        batch = env.reset()
        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = env.action_dim

        self.shac = SHAC(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=env.device,
        )
        self.rollout_horizon = cfg.rollout_horizon

    def train(self, max_iterations: int = 100) -> None:
        """Run training loop."""
        for iteration in range(max_iterations):
            self._rollout()
            losses, grad_norms = self.shac.update()
            self.shac.detach()

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: actor_loss={losses['actor_loss']:.4f}, "
                    f"critic_loss={losses.get('critic_loss', 0.0):.4f}, "
                    f"grad_norm={grad_norms['actor_grad_norm']:.4f}"
                )

    def _rollout(self) -> None:
        """Execute one SHAC rollout with value bootstrap."""
        batch = self.env.reset()

        for _ in range(self.rollout_horizon):
            policy_obs = batch.observations["policy"]
            action, policy_info = self.shac.actor_act(policy_obs)

            batch.observations, rewards, terminated, truncated, batch.extras = self.env.step(action)

            critic_obs = batch.observations["critic"]
            value = self.shac.critic_forward(critic_obs)
            self.shac.record_value(critic_obs, value)

            if EXTRA_TASK_TERMS in batch.extras:
                task_terms = batch.extras[EXTRA_TASK_TERMS]
                if "progress" in task_terms:
                    loss = -task_terms["progress"].mean()
                else:
                    loss = -rewards.mean()
            else:
                loss = -rewards.mean()

            self.shac.record_loss(loss, policy_info, batch.extras, terminated)
