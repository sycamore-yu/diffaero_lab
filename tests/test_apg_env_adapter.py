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


def test_apg_adapter_reads_policy_critic(shared_env):
    """Test that adapter exposes policy and critic observations."""
    batch = shared_env.reset()
    assert "policy" in batch.observations
    assert "critic" in batch.observations


def test_apg_adapter_step_produces_sim_state(shared_env):
    """Test that step() produces sim_state and task_terms in extras."""
    shared_env.reset()

    action = torch.randn_like(shared_env.get_policy_action())
    observations, rewards, terminated, truncated, extras = shared_env.step(action)

    assert "policy" in observations
    assert "critic" in observations
    assert "sim_state" in extras
    assert "task_terms" in extras
    assert rewards.shape[0] == shared_env.num_envs


def test_apg_adapter_contract_keys(shared_env):
    """Test that extras contain required sim_state fields."""
    shared_env.reset()

    action = torch.randn_like(shared_env.get_policy_action())
    observations, rewards, terminated, truncated, extras = shared_env.step(action)

    sim_state = extras["sim_state"]
    assert isinstance(sim_state, dict)
    assert "position_w" in sim_state
    assert "linear_velocity_w" in sim_state


def test_apgtrainer_initialization(shared_env):
    """Test that APGTrainer can be initialized."""
    from diffaero_lab.algo.algorithms.apg import APGConfig
    from diffaero_lab.algo.trainers.apg_trainer import APGTrainer

    cfg = APGConfig(
        lr=3e-4,
        max_grad_norm=1.0,
        rollout_horizon=32,
    )
    trainer = APGTrainer(env=shared_env, cfg=cfg)

    assert trainer is not None
    assert trainer.rollout_horizon == 32


def test_apg_actor_forward(shared_env):
    """Test that APG actor can forward pass."""
    from diffaero_lab.algo.algorithms.apg import APG, APGConfig

    batch = shared_env.reset()

    obs_dim = batch.observations["policy"].shape[-1]
    action_dim = shared_env.action_dim

    cfg = APGConfig(
        lr=3e-4,
        max_grad_norm=1.0,
        rollout_horizon=32,
    )
    apg = APG(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim, device=shared_env.device)

    action, policy_info = apg.act(batch.observations["policy"])

    assert action.shape[-1] == action_dim


def test_apg_rollout_and_backward(shared_env):
    """Test APG rollout and backward pass."""
    from diffaero_lab.algo.algorithms.apg import APG, APGConfig

    batch = shared_env.reset()

    obs_dim = batch.observations["policy"].shape[-1]
    action_dim = shared_env.action_dim

    cfg = APGConfig(
        lr=3e-4,
        max_grad_norm=1.0,
        rollout_horizon=8,
    )
    apg = APG(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim, device=shared_env.device)

    for _ in range(4):
        action, policy_info = apg.act(batch.observations["policy"])
        batch.observations, rewards, terminated, truncated, batch.extras = shared_env.step(action)
        if "task_terms" in batch.extras:
            loss = -rewards.mean()
            apg.record_loss(loss, policy_info, batch.extras)

    losses, grad_norms = apg.update_actor()

    assert "actor_loss" in losses
    assert losses["actor_loss"] is not None
