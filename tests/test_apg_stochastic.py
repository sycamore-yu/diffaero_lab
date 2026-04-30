# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager

import diffaero_lab.tasks  # noqa: F401
from tests.isaac_test_utils import make_cpu_test_adapter

get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)


@pytest.fixture(scope="module")
def shared_env():
    """Shared adapter for all tests in this module."""
    adapter = make_cpu_test_adapter()
    yield adapter
    adapter.close()


def test_apg_stochastic_actor_outputs_action_and_log_prob():
    """Test that APGStochastic actor produces action and log_prob via stochastic sampling."""
    from diffaero_lab.algo.algorithms.apg_stochastic import APGStochastic, APGStochasticConfig

    policy = APGStochastic(APGStochasticConfig(), obs_dim=13, action_dim=4, device="cpu")
    obs = torch.zeros(8, 13)
    action, info = policy.act(obs)
    assert action.shape == (8, 4), f"Expected action shape (8, 4), got {action.shape}"
    assert "log_prob" in info, "Expected 'log_prob' key in policy_info"


def test_apg_stochastic_rollout_and_update_smoke(shared_env):
    """Integration smoke test: adapter + APGStochasticTrainer stochastic rollout and update."""
    from diffaero_lab.algo.algorithms.apg_stochastic import APGStochastic, APGStochasticConfig
    from diffaero_lab.algo.trainers.apg_stochastic_trainer import APGStochasticTrainer

    cfg = APGStochasticConfig(
        lr=3e-4,
        max_grad_norm=1.0,
        rollout_horizon=8,
    )
    trainer = APGStochasticTrainer(env=shared_env, cfg=cfg)

    assert trainer is not None
    assert trainer.rollout_horizon == 8

    batch = shared_env.reset()

    obs_dim = batch.observations["policy"].shape[-1]
    action_dim = shared_env.action_dim

    apg = APGStochastic(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim, device=shared_env.device)

    for _ in range(4):
        action, policy_info = apg.act(batch.observations["policy"])
        assert "log_prob" in policy_info, "Expected 'log_prob' in policy_info during rollout"
        batch.observations, rewards, terminated, truncated, batch.extras = shared_env.step(action)
        if "task_terms" in batch.extras:
            loss = -rewards.mean()
            apg.record_loss(loss, policy_info, batch.extras)

    losses, grad_norms = apg.update_actor()

    assert "actor_loss" in losses
    assert losses["actor_loss"] is not None
