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

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager

import diffaero_env.tasks  # noqa: F401
from diffaero_algo.wrappers.env_adapter import DifferentialEnvAdapter

get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)


@pytest.fixture(scope="module")
def shared_env():
    """Shared adapter for all tests in this module."""
    from diffaero_env.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    sim_utils.create_new_stage()
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)
    cfg = DroneRacingEnvCfg()
    cfg.scene = cfg.scene.replace(num_envs=8, replicate_physics=False)
    cfg.sim.device = "cpu"
    adapter = DifferentialEnvAdapter.make("Isaac-Drone-Racing-Direct-v0", cfg=cfg)
    yield adapter
    adapter.close()


class TestSHACCriticStateConsumption:
    """Tests for SHAC (Soft Hierarchical Actor-Critic) critic-state consumption.

    SHAC should consume observations["critic"] for value estimation separate from
    the policy's observations["policy"] path.
    """

    def test_shac_imports(self):
        """Test that SHAC module can be imported."""
        from diffaero_algo.algorithms.shac import SHAC, SHACConfig
        from diffaero_algo.algorithms.shac import CriticNetwork

        assert SHAC is not None
        assert SHACConfig is not None
        assert CriticNetwork is not None

    def test_shac_config(self):
        """Test that SHACConfig can be instantiated."""
        from diffaero_algo.algorithms.shac import SHACConfig

        cfg = SHACConfig(
            lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=32,
            hidden_dims=(256, 128, 64),
            critic_hidden_dims=(256, 128, 64),
            entropy_coef=0.01,
        )
        assert cfg.lr == 3e-4
        assert cfg.critic_lr == 3e-4
        assert cfg.entropy_coef == 0.01

    def test_shac_actor_critic_separate_inputs(self, shared_env):
        """Test that SHAC actor and critic consume separate observation paths.

        SHAC should use observations["policy"] for the actor and
        observations["critic"] for the critic network.
        """
        from diffaero_algo.algorithms.shac import SHAC, SHACConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHACConfig(
            lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=8,
        )
        shac = SHAC(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        # Actor should consume policy observations
        policy_obs = batch.observations["policy"]
        action, policy_info = shac.actor_act(policy_obs)
        assert action.shape[-1] == action_dim

        # Critic should consume critic observations
        critic_obs = batch.observations["critic"]
        value = shac.critic_forward(critic_obs)
        assert value.shape[-1] == 1

    def test_shac_rollout_with_value_bootstrap(self, shared_env):
        """Test SHAC rollout with value bootstrap using critic observations.

        SHAC should compute value estimates from critic observations and use
        them for advantage estimation.
        """
        from diffaero_algo.algorithms.shac import SHAC, SHACConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHACConfig(
            lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=8,
        )
        shac = SHAC(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        for _ in range(4):
            policy_obs = batch.observations["policy"]
            action, policy_info = shac.actor_act(policy_obs)
            batch.observations, rewards, terminated, truncated, batch.extras = shared_env.step(action)

            critic_obs = batch.observations["critic"]
            value = shac.critic_forward(critic_obs)
            shac.record_value(critic_obs, value)

            loss = -rewards.mean()
            shac.record_loss(loss, policy_info, batch.extras, reward=rewards)

        losses, grad_norms = shac.update()

        assert "actor_loss" in losses
        assert "critic_loss" in losses

    def test_shac_trainer_initialization(self, shared_env):
        """Test that SHACTrainer can be initialized."""
        from diffaero_algo.algorithms.shac import SHACConfig
        from diffaero_algo.trainers.shac_trainer import SHACTrainer

        cfg = SHACConfig(
            lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=32,
        )
        trainer = SHACTrainer(env=shared_env, cfg=cfg)

        assert trainer is not None
        assert trainer.rollout_horizon == 32

    def test_shac_entropy_regularization(self, shared_env):
        """Test that SHAC supports entropy regularization for exploration."""
        from diffaero_algo.algorithms.shac import SHAC, SHACConfig

        batch = shared_env.reset()

        policy_obs_dim = batch.observations["policy"].shape[-1]
        critic_obs_dim = batch.observations["critic"].shape[-1]
        action_dim = shared_env.action_dim

        cfg = SHACConfig(
            lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            rollout_horizon=8,
            entropy_coef=0.01,
        )
        shac = SHAC(
            cfg=cfg,
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=shared_env.device,
        )

        policy_obs = batch.observations["policy"]
        action, policy_info = shac.actor_act(policy_obs, test=False)

        assert "entropy" in policy_info
        assert isinstance(policy_info["entropy"], torch.Tensor)
        assert policy_info["entropy"].shape == (shared_env.num_envs,)
