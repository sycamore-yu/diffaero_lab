# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failing tests for Warp/Newton task registration (Phase 2D Task 1).

These tests verify that the Isaac-Drone-Racing-Direct-Warp-v0 task is registered.
They should FAIL until Phase 2D Task 2 implements the Warp/Newton task registration.
"""

import gymnasium as gym


def test_warp_task_id_is_registered():
    """Verify Isaac-Drone-Racing-Direct-Warp-v0 is registered in the Gym registry.

    This test FAILS because the Warp/Newton task registration is not yet implemented.
    Expected failure reason: KeyError - task ID not found in gym.registry
    """
    import diffaero_lab.tasks  # noqa: F401

    assert "Isaac-Drone-Racing-Direct-Warp-v0" in gym.registry, (
        "Isaac-Drone-Racing-Direct-Warp-v0 is not registered. "
        "Phase 2D Task 2 must register this task ID with a Warp/Newton cfg path."
    )


def test_warp_task_spec_entry_point():
    """Verify the Warp task entry point resolves to a DroneRacingEnv with Warp cfg.

    This test FAILS because the Warp task registration does not exist.
    Expected failure reason: gym.error.UnregisteredEnv - task ID not registered
    """
    import diffaero_lab.tasks  # noqa: F401

    spec = gym.spec("Isaac-Drone-Racing-Direct-Warp-v0")
    assert "DroneRacingEnv" in spec.entry_point, "Warp task entry point should resolve to DroneRacingEnv"
    assert "env_cfg_entry_point" in spec.kwargs, "Warp task should have env_cfg_entry_point kwarg"


def test_warp_task_cfg_entry_point_refers_to_warp_config():
    """Verify the Warp task config entry point references a Warp-specific config.

    This test FAILS because the Warp task registration does not exist.
    Expected failure reason: gym.error.UnregisteredEnv - task ID not registered
    """
    import diffaero_lab.tasks  # noqa: F401

    spec = gym.spec("Isaac-Drone-Racing-Direct-Warp-v0")
    env_cfg_entry_point = spec.kwargs["env_cfg_entry_point"]
    # The Warp config should be distinguishable from the PhysX config
    # Current PhysX config: diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg:DroneRacingEnvCfg
    # Expected Warp config: should reference Warp or Newton in the path
    assert "warp" in env_cfg_entry_point.lower() or "newton" in env_cfg_entry_point.lower(), (
        f"Warp task env_cfg_entry_point should reference Warp/Newton config, got: {env_cfg_entry_point}"
    )


def test_warp_task_has_skrl_cfg():
    """Verify the Warp task has a skrl_cfg_entry_point kwarg.

    This test FAILS because the Warp task registration does not exist.
    Expected failure reason: gym.error.UnregisteredEnv - task ID not registered
    """
    import diffaero_lab.tasks  # noqa: F401

    spec = gym.spec("Isaac-Drone-Racing-Direct-Warp-v0")
    assert "skrl_cfg_entry_point" in spec.kwargs, "Warp task should have skrl_cfg_entry_point kwarg for RL training"
