# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared actor-critic base module.

This module provides the shared actor-critic architecture used by SHAC and SHA2C
algorithms, with separate observation paths for policy and critic networks.
"""

import torch


class TestSharedActorCriticImports:
    """Tests for actor_critic module imports."""

    def test_actor_critic_imports(self):
        """Test that actor_critic module can be imported."""
        from diffaero_algo.algorithms.actor_critic import (
            OBS_CRITIC,
            OBS_POLICY,
            GaussianActorHead,
            SharedActorCritic,
            SharedActorCriticConfig,
            ValueCriticHead,
        )

        assert SharedActorCritic is not None
        assert SharedActorCriticConfig is not None
        assert GaussianActorHead is not None
        assert ValueCriticHead is not None
        assert OBS_POLICY == "policy"
        assert OBS_CRITIC == "critic"


class TestSharedActorCriticConfig:
    """Tests for SharedActorCriticConfig dataclass."""

    def test_default_config(self):
        """Test that default config has expected values."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCriticConfig

        cfg = SharedActorCriticConfig()
        assert cfg.actor_hidden_dims == (256, 128, 64)
        assert cfg.critic_hidden_dims == (256, 128, 64)
        assert cfg.init_log_std == 0.0

    def test_custom_config(self):
        """Test that custom config values are preserved."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCriticConfig

        cfg = SharedActorCriticConfig(
            actor_hidden_dims=(512, 256),
            critic_hidden_dims=(1024, 512, 256),
            init_log_std=-0.5,
        )
        assert cfg.actor_hidden_dims == (512, 256)
        assert cfg.critic_hidden_dims == (1024, 512, 256)
        assert cfg.init_log_std == -0.5


class TestGaussianActorHead:
    """Tests for GaussianActorHead (policy head)."""

    def test_actor_forward_shape(self):
        """Test that actor forward pass returns correct shapes."""
        from diffaero_algo.algorithms.actor_critic import GaussianActorHead

        batch_size = 8
        obs_dim = 64
        action_dim = 12

        actor = GaussianActorHead(obs_dim=obs_dim, action_dim=action_dim)
        obs = torch.randn(batch_size, obs_dim)

        mean, log_std = actor(obs)

        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)

    def test_actor_sample(self):
        """Test that actor can sample actions."""
        from diffaero_algo.algorithms.actor_critic import GaussianActorHead

        batch_size = 8
        obs_dim = 64
        action_dim = 12

        actor = GaussianActorHead(obs_dim=obs_dim, action_dim=action_dim)
        obs = torch.randn(batch_size, obs_dim)

        mean, log_std = actor(obs)
        action = actor.sample(mean, log_std)

        assert action.shape == (batch_size, action_dim)
        assert action.abs().max() <= 1.0 + 1e-5  # Tanh squashes to [-1, 1]


class TestValueCriticHead:
    """Tests for ValueCriticHead (value head)."""

    def test_critic_forward_shape(self):
        """Test that critic forward pass returns correct shapes."""
        from diffaero_algo.algorithms.actor_critic import ValueCriticHead

        batch_size = 8
        obs_dim = 128

        critic = ValueCriticHead(obs_dim=obs_dim)
        obs = torch.randn(batch_size, obs_dim)

        value = critic(obs)

        assert value.shape == (batch_size, 1)

    def test_critic_q_value_shape(self):
        """Test that critic Q-value computation with actions returns correct shapes."""
        from diffaero_algo.algorithms.actor_critic import ValueCriticHead

        batch_size = 8
        obs_dim = 128
        action_dim = 12

        critic = ValueCriticHead(obs_dim=obs_dim, action_dim=action_dim)
        obs = torch.randn(batch_size, obs_dim)
        action = torch.randn(batch_size, action_dim)

        q_value = critic(obs, action)

        assert q_value.shape == (batch_size, 1)


class TestSharedActorCritic:
    """Tests for SharedActorCritic (full actor-critic module)."""

    def test_actor_critic_initialization(self):
        """Test that SharedActorCritic can be initialized with separate dims."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCritic, SharedActorCriticConfig

        policy_obs_dim = 64
        critic_obs_dim = 128
        action_dim = 12

        cfg = SharedActorCriticConfig()
        ac = SharedActorCritic(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
        )

        assert ac is not None
        assert ac.policy_obs_dim == policy_obs_dim
        assert ac.critic_obs_dim == critic_obs_dim
        assert ac.action_dim == action_dim

    def test_actor_critic_separate_observation_paths(self):
        """Test that actor and critic use separate observation inputs."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCritic, SharedActorCriticConfig

        policy_obs_dim = 64
        critic_obs_dim = 128
        action_dim = 12

        cfg = SharedActorCriticConfig()
        ac = SharedActorCritic(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
        )

        policy_obs = torch.randn(8, policy_obs_dim)
        critic_obs = torch.randn(8, critic_obs_dim)

        # Actor should consume policy observations
        action, policy_info = ac.actor_act(policy_obs)
        assert action.shape == (8, action_dim)
        assert "mean" in policy_info
        assert "log_std" in policy_info

        # Critic should consume critic observations
        value = ac.critic_forward(critic_obs)
        assert value.shape == (8, 1)

    def test_actor_critic_q_value_with_action(self):
        """Test that critic can compute Q-value given observations and actions."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCritic, SharedActorCriticConfig

        policy_obs_dim = 64
        critic_obs_dim = 128
        action_dim = 12

        cfg = SharedActorCriticConfig()
        ac = SharedActorCritic(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
        )

        policy_obs = torch.randn(8, policy_obs_dim)
        critic_obs = torch.randn(8, critic_obs_dim)

        action, _ = ac.actor_act(policy_obs)
        q_value = ac.critic_forward(critic_obs, action)

        assert q_value.shape == (8, 1)

    def test_actor_critic_entropy(self):
        """Test that actor_critic produces entropy for exploration."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCritic, SharedActorCriticConfig

        policy_obs_dim = 64
        critic_obs_dim = 128
        action_dim = 12

        cfg = SharedActorCriticConfig(init_log_std=0.0)
        ac = SharedActorCritic(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
        )

        policy_obs = torch.randn(8, policy_obs_dim)
        action, policy_info = ac.actor_act(policy_obs, with_log_prob=True)

        assert "entropy" in policy_info
        assert policy_info["entropy"].shape == (8,)

    def test_actor_critic_to_device(self):
        """Test that actor_critic can be moved to device."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCritic, SharedActorCriticConfig

        cfg = SharedActorCriticConfig()
        ac = SharedActorCritic(
            cfg=cfg,
            policy_obs_dim=64,
            critic_obs_dim=128,
            action_dim=12,
        )

        # Just verify it has the method and doesn't error
        ac = ac.to("cpu")
        assert next(ac.actor.parameters()).device.type == "cpu"

    def test_actor_critic_train_eval_mode(self):
        """Test that actor_critic can switch between train and eval modes."""
        from diffaero_algo.algorithms.actor_critic import SharedActorCritic, SharedActorCriticConfig

        cfg = SharedActorCriticConfig()
        ac = SharedActorCritic(
            cfg=cfg,
            policy_obs_dim=64,
            critic_obs_dim=128,
            action_dim=12,
        )

        ac.train()
        assert ac.training

        ac.eval()
        assert not ac.training
