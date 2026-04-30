# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""APG Stochastic trainer that orchestrates rollout and backward pass with stochastic policies."""

from diffaero_lab.algo.algorithms.apg_stochastic import APGStochastic, APGStochasticConfig
from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter
from diffaero_lab.common.keys import EXTRA_SIM_STATE, EXTRA_TASK_TERMS


class APGStochasticTrainer:
    """Trainer that orchestrates stochastic APG rollout and backward pass.

    Mirrors APGTrainer but uses APGStochastic for Gaussian policy with
    reparameterized sampling and tanh squashing.
    """

    def __init__(self, env: DifferentialEnvAdapter, cfg: APGStochasticConfig):
        self.env = env
        self.cfg = cfg

        batch = env.reset()
        obs_dim = batch.observations["policy"].shape[-1]
        action_dim = env.action_dim

        self.apg = APGStochastic(
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=env.device,
        )
        self.rollout_horizon = cfg.rollout_horizon

    def train(self, max_iterations: int = 100) -> None:
        """Run training loop."""
        for iteration in range(max_iterations):
            self._rollout()
            losses, grad_norms = self.apg.update_actor()
            self.apg.detach()
            self.env.detach()

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: actor_loss={losses['actor_loss']:.4f}, "
                    f"grad_norm={grad_norms['actor_grad_norm']:.4f}"
                )

    def _rollout(self) -> None:
        """Execute one rollout and backpropagate the environment's differentiable loss."""
        batch = self.env.reset()

        for _ in range(self.rollout_horizon):
            action, policy_info = self.apg.act(batch.observations["policy"])
            batch.observations, rewards, terminated, truncated, batch.extras = self.env.step(action)

            task_terms = batch.extras[EXTRA_TASK_TERMS]
            sim_state = batch.extras.get(EXTRA_SIM_STATE, {})
            dynamics = sim_state.get("dynamics", {}) if isinstance(sim_state, dict) else {}
            if dynamics.get("tensor_backend") == "warp":
                self.apg.record_loss(task_terms["loss"], policy_info, batch.extras)
            else:
                self.apg.record_policy_gradient_loss(rewards, policy_info)
