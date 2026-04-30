# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest
import torch
import numpy as np

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


class TestSHA2CAsymmetricActorCritic:
    """Tests for SHA2C (Soft Actor-Critic with Asymmetric Actor-Critic) paths.

    SHA2C uses different observation inputs for actor and critic networks:
    - Actor uses observations["policy"]
    - Critic uses observations["critic"]
    This asymmetry requires careful handling in the update rules.
    """

    def test_sha2c_imports(self):
        """Test that SHA2C module can be imported."""
        from diffaero_lab.algo.algorithms.sha2c import SHA2C, SHA2CConfig
        from diffaero_lab.algo.algorithms.sha2c import AsymmetricActor, AsymmetricCritic

        assert SHA2C is not None
        assert SHA2CConfig is not None
        assert AsymmetricActor is not None
        assert AsymmetricCritic is not None

    def test_sha2c_config(self):
        """Test that SHA2CConfig can be instantiated with asymmetric parameters."""
        from diffaero_lab.algo.algorithms.sha2c import SHA2CConfig

        cfg = SHA2CConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=32,
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(512, 256, 128),
            init_log_std=-0.5,
            target_entropy=-4.0,
            soft_update_coef=0.005,
        )
        assert cfg.actor_lr == 3e-4
        assert cfg.critic_lr == 3e-4
        assert cfg.actor_hidden_dims == (256, 128, 64)
        assert cfg.critic_hidden_dims == (512, 256, 128)

    def test_sha2c_asymmetric_architecture(self, shared_env):
        """Test that SHA2C has separate actor and critic networks.

        SHA2C should have:
        - AsymmetricActor with potentially different architecture than critic
        - AsymmetricCritic that takes different input (critic observations)
        - Separate optimizers for actor and critic
        """
        from diffaero_lab.algo.algorithms.sha2c import SHA2C, SHA2CConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHA2CConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=8,
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(512, 256, 128),
        )
        sha2c = SHA2C(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        assert sha2c.actor is not None
        assert sha2c.critic is not None
        assert sha2c.target_critic is not None
        assert sha2c.actor_optimizer is not None
        assert sha2c.critic_optimizer is not None

    def test_sha2c_separate_observation_paths(self, shared_env):
        """Test that actor and critic use separate observation inputs.

        Actor should use policy observations while critic uses critic observations.
        """
        from diffaero_lab.algo.algorithms.sha2c import SHA2C, SHA2CConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHA2CConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=8,
        )
        sha2c = SHA2C(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        policy_obs = batch.observations["policy"]
        critic_obs = batch.observations["critic"]

        action, policy_info = sha2c.actor_act(policy_obs)
        q_value = sha2c.critic_forward(critic_obs, action)

        assert action.shape[-1] == action_dim
        assert q_value.shape[-1] == 1

    def test_sha2c_soft_update_target_critic(self, shared_env):
        """Test that SHA2C uses soft updates for target critic network.

        Soft update: target = tau * current + (1 - tau) * target
        """
        from diffaero_lab.algo.algorithms.sha2c import SHA2C, SHA2CConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHA2CConfig(
            actor_lr=3e-4,
            critic_lr=0.01,
            max_grad_norm=1.0,
            rollout_horizon=8,
            soft_update_coef=0.1,
        )
        sha2c = SHA2C(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        policy_obs = batch.observations["policy"]
        action, policy_info = sha2c.actor_act(policy_obs)

        old_target_params = {k: v.clone().cpu().detach().numpy() for k, v in sha2c.target_critic.state_dict().items()}

        batch.observations, rewards, terminated, truncated, batch.extras = shared_env.step(action)

        critic_obs = batch.observations["critic"]
        policy_obs_new = batch.observations["policy"]
        action_for_critic, _ = sha2c.actor_act(policy_obs_new)
        value = sha2c.critic_forward(critic_obs, action_for_critic)
        sha2c.record_transition(
            critic_obs=critic_obs,
            policy_obs=policy_obs_new,
            action=action_for_critic,
            reward=rewards,
            terminated=terminated,
            value=value,
            batch_extras=batch.extras,
        )
        old_critic_params = {k: v.clone().cpu().numpy() for k, v in sha2c.critic.state_dict().items()}
        for i in range(100):
            losses, grads = sha2c.update_critic(rewards, terminated, batch.extras)
        new_critic_params = {k: v.clone().cpu().numpy() for k, v in sha2c.critic.state_dict().items()}

        new_target_params = {k: v.clone().cpu().detach().numpy() for k, v in sha2c.target_critic.state_dict().items()}

        for k in old_target_params:
            diff = np.abs(new_target_params[k] - old_target_params[k]).max()
            if "q_value_net" in k:
                assert diff > 1e-6, f"target_critic parameter '{k}' did not change (max_diff={diff})"

    def test_sha2c_trainer_initialization(self, shared_env):
        """Test that SHA2CTrainer can be initialized."""
        from diffaero_lab.algo.algorithms.sha2c import SHA2CConfig
        from diffaero_lab.algo.trainers.sha2c_trainer import SHA2CTrainer

        cfg = SHA2CConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=32,
        )
        trainer = SHA2CTrainer(env=shared_env, cfg=cfg)

        assert trainer is not None
        assert trainer.rollout_horizon == 32

    def test_sha2c_entropy_and_target_entropy(self, shared_env):
        """Test that SHA2C supports automatic entropy tuning.

        SHA2C should adapt entropy coefficient to achieve target_entropy.
        """
        from diffaero_lab.algo.algorithms.sha2c import SHA2C, SHA2CConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHA2CConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=8,
            target_entropy=-4.0,
        )
        sha2c = SHA2C(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        assert sha2c.log_alpha is not None
        assert sha2c.target_entropy == -4.0
        assert isinstance(sha2c.log_alpha, torch.nn.Parameter)
