# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch


def test_drone_racing_env_is_registered():
    import diffaero_lab.tasks  # noqa: F401

    assert "Isaac-Drone-Racing-Direct-v0" in gym.registry


def test_drone_racing_env_spec_entry_point():
    import diffaero_lab.tasks  # noqa: F401

    spec = gym.registry["Isaac-Drone-Racing-Direct-v0"]
    assert "DroneRacingEnv" in spec.entry_point
    assert "env_cfg_entry_point" in spec.kwargs
    assert "skrl_cfg_entry_point" in spec.kwargs


def test_drone_racing_env_cfg_has_correct_spaces():
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    cfg = DroneRacingEnvCfg()
    assert cfg.action_space == 4
    assert cfg.observation_space == 13
    assert cfg.state_space == 13


def test_drone_racing_env_creates_correct_extras_keys():
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg
    from diffaero_lab.common.keys import EXTRA_CAPABILITIES, EXTRA_SIM_STATE, EXTRA_TASK_TERMS

    cfg = DroneRacingEnvCfg()
    cfg.scene.num_envs = 4

    assert EXTRA_CAPABILITIES == "capabilities"
    assert EXTRA_SIM_STATE == "sim_state"
    assert EXTRA_TASK_TERMS == "task_terms"
    assert hasattr(cfg, "rew_scale_progress")
    assert hasattr(cfg, "rew_scale_tracking")
    assert hasattr(cfg, "rew_scale_control_effort")
    assert hasattr(cfg, "rew_scale_terminal")


def test_drone_racing_bridge_exports_required_keys():
    import warp as wp

    from diffaero_lab.tasks.direct.drone_racing.dynamics_bridge.quad import QuadDynamicsBridge
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    wp.init()
    cfg = DroneRacingEnvCfg()
    cfg.scene.num_envs = 4

    class FakeRobot:
        def __init__(self):
            self.data = type(
                "data",
                (),
                {
                    "root_pos_w": wp.from_torch(torch.zeros(4, 3), dtype=wp.vec3),
                    "root_quat_w": wp.from_torch(torch.zeros(4, 4), dtype=wp.quat),
                    "root_lin_vel_w": wp.from_torch(torch.zeros(4, 3), dtype=wp.vec3),
                    "root_ang_vel_b": wp.from_torch(torch.zeros(4, 3), dtype=wp.vec3),
                },
            )()

    bridge = QuadDynamicsBridge(cfg=cfg, robot=FakeRobot(), num_envs=4, device="cpu")
    base_state = bridge.read_base_state()
    assert "position_w" in base_state
    assert "quaternion_w" in base_state
    assert "linear_velocity_w" in base_state
    assert "angular_velocity_b" in base_state

    motor_state = bridge.read_motor_state()
    assert "motor_omega" in motor_state

    dyn_info = bridge.read_dynamics_info()
    assert dyn_info["model_name"] == "quad"
    assert dyn_info["tensor_backend"] == "physx"
