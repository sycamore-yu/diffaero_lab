# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-dynamics backend contract preservation.

These tests verify that pmd, pmc, and simple backends export the required
common sim_state keys plus their backend-specific fields. They fail until
the backends are implemented per Phase 2C.
"""

import torch


COMMON_BASE_STATE_KEYS = {"position_w", "quaternion_w", "linear_velocity_w", "angular_velocity_b"}


def test_pmd_bridge_exports_common_base_state_keys():
    """Verify PMD bridge exports all required common base state keys."""
    from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.pointmass_discrete import PMDDynamicsBridge

    class FakeRobot:
        def __init__(self):
            self.data = type(
                "data",
                (),
                {
                    "root_pos_w": torch.zeros(4, 3),
                    "root_quat_w": torch.zeros(4, 4),
                    "root_lin_vel_w": torch.zeros(4, 3),
                    "root_ang_vel_b": torch.zeros(4, 3),
                },
            )()

        def permanent_wrench_composer(self):
            return self

        def set_forces_and_torques_index(self, *args, **kwargs):
            pass

    class FakeCfg:
        thrust_scale = 1.0
        moment_scale = 0.01

    robot = FakeRobot()
    bridge = PMDDynamicsBridge(cfg=FakeCfg(), robot=robot, num_envs=4, device="cpu")
    base_state = bridge.read_base_state()

    for key in COMMON_BASE_STATE_KEYS:
        assert key in base_state, f"PMD bridge must export '{key}' in base_state"


def test_pmc_bridge_exports_common_base_state_keys():
    """Verify PMC bridge exports all required common base state keys."""
    from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.pointmass_continuous import PMCDynamicsBridge

    class FakeRobot:
        def __init__(self):
            self.data = type(
                "data",
                (),
                {
                    "root_pos_w": torch.zeros(4, 3),
                    "root_quat_w": torch.zeros(4, 4),
                    "root_lin_vel_w": torch.zeros(4, 3),
                    "root_ang_vel_b": torch.zeros(4, 3),
                },
            )()

        def permanent_wrench_composer(self):
            return self

        def set_forces_and_torques_index(self, *args, **kwargs):
            pass

    class FakeCfg:
        thrust_scale = 1.0
        moment_scale = 0.01

    robot = FakeRobot()
    bridge = PMCDynamicsBridge(cfg=FakeCfg(), robot=robot, num_envs=4, device="cpu")
    base_state = bridge.read_base_state()

    for key in COMMON_BASE_STATE_KEYS:
        assert key in base_state, f"PMC bridge must export '{key}' in base_state"


def test_simple_bridge_exports_common_base_state_keys():
    """Verify Simple bridge exports all required common base state keys."""
    from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.simplified_quad import SimpleDynamicsBridge

    class FakeRobot:
        def __init__(self):
            self.data = type(
                "data",
                (),
                {
                    "root_pos_w": torch.zeros(4, 3),
                    "root_quat_w": torch.zeros(4, 4),
                    "root_lin_vel_w": torch.zeros(4, 3),
                    "root_ang_vel_b": torch.zeros(4, 3),
                },
            )()

        def permanent_wrench_composer(self):
            return self

        def set_forces_and_torques_index(self, *args, **kwargs):
            pass

    class FakeCfg:
        thrust_scale = 1.0
        moment_scale = 0.01

    robot = FakeRobot()
    bridge = SimpleDynamicsBridge(cfg=FakeCfg(), robot=robot, num_envs=4, device="cpu")
    base_state = bridge.read_base_state()

    for key in COMMON_BASE_STATE_KEYS:
        assert key in base_state, f"Simple bridge must export '{key}' in base_state"


def test_pmd_bridge_dynamics_info():
    """Verify PMD bridge reports correct model_name in dynamics info."""
    from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.pointmass_discrete import PMDDynamicsBridge

    class FakeRobot:
        def __init__(self):
            self.data = type("data", (), {})()

        def permanent_wrench_composer(self):
            return self

        def set_forces_and_torques_index(self, *args, **kwargs):
            pass

    class FakeCfg:
        thrust_scale = 1.0
        moment_scale = 0.01

    bridge = PMDDynamicsBridge(cfg=FakeCfg(), robot=FakeRobot(), num_envs=4, device="cpu")
    dyn_info = bridge.read_dynamics_info()

    assert dyn_info["model_name"] == "pmd"
    assert "state_layout_version" in dyn_info
    assert "quat_convention" in dyn_info


def test_pmc_bridge_dynamics_info():
    """Verify PMC bridge reports correct model_name in dynamics info."""
    from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.pointmass_continuous import PMCDynamicsBridge

    class FakeRobot:
        def __init__(self):
            self.data = type("data", (), {})()

        def permanent_wrench_composer(self):
            return self

        def set_forces_and_torques_index(self, *args, **kwargs):
            pass

    class FakeCfg:
        thrust_scale = 1.0
        moment_scale = 0.01

    bridge = PMCDynamicsBridge(cfg=FakeCfg(), robot=FakeRobot(), num_envs=4, device="cpu")
    dyn_info = bridge.read_dynamics_info()

    assert dyn_info["model_name"] == "pmc"
    assert "state_layout_version" in dyn_info
    assert "quat_convention" in dyn_info


def test_simple_bridge_dynamics_info():
    """Verify Simple bridge reports correct model_name in dynamics info."""
    from diffaero_env.tasks.direct.drone_racing.dynamics_bridge.simplified_quad import SimpleDynamicsBridge

    class FakeRobot:
        def __init__(self):
            self.data = type("data", (), {})()

        def permanent_wrench_composer(self):
            return self

        def set_forces_and_torques_index(self, *args, **kwargs):
            pass

    class FakeCfg:
        thrust_scale = 1.0
        moment_scale = 0.01

    bridge = SimpleDynamicsBridge(cfg=FakeCfg(), robot=FakeRobot(), num_envs=4, device="cpu")
    dyn_info = bridge.read_dynamics_info()

    assert dyn_info["model_name"] == "simple"
    assert "state_layout_version" in dyn_info
    assert "quat_convention" in dyn_info


def test_build_sim_state_pmd_succeeds():
    """Verify build_sim_state supports 'pmd' model_name."""
    from diffaero_common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="pmd")
    assert state is not None
    assert "position_w" in state
    assert "dynamics" in state
    assert state["dynamics"]["model_name"] == "pmd"


def test_build_sim_state_pmc_succeeds():
    """Verify build_sim_state supports 'pmc' model_name."""
    from diffaero_common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="pmc")
    assert state is not None
    assert "position_w" in state
    assert "dynamics" in state
    assert state["dynamics"]["model_name"] == "pmc"


def test_build_sim_state_simple_succeeds():
    """Verify build_sim_state supports 'simple' model_name."""
    from diffaero_common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="simple")
    assert state is not None
    assert "position_w" in state
    assert "dynamics" in state
    assert state["dynamics"]["model_name"] == "simple"


def test_flatten_unflatten_round_trip_pmd():
    """Verify flatten/unflatten round-trip works for pmd state."""
    from diffaero_common.adapters.flatten import flatten_sim_state, unflatten_sim_state

    sim_state = {
        "position_w": torch.randn(2, 3),
        "quaternion_w": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "linear_velocity_w": torch.randn(2, 3),
        "angular_velocity_b": torch.randn(2, 3),
    }
    flat = flatten_sim_state(sim_state, model_name="pmd")
    restored = unflatten_sim_state(flat, model_name="pmd")

    for key in sim_state:
        assert key in restored, f"Key '{key}' missing after unflatten"
        torch.testing.assert_close(restored[key], sim_state[key])


def test_flatten_unflatten_round_trip_pmc():
    """Verify flatten/unflatten round-trip works for pmc state."""
    from diffaero_common.adapters.flatten import flatten_sim_state, unflatten_sim_state

    sim_state = {
        "position_w": torch.randn(2, 3),
        "quaternion_w": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "linear_velocity_w": torch.randn(2, 3),
        "angular_velocity_b": torch.randn(2, 3),
    }
    flat = flatten_sim_state(sim_state, model_name="pmc")
    restored = unflatten_sim_state(flat, model_name="pmc")

    for key in sim_state:
        assert key in restored, f"Key '{key}' missing after unflatten"
        torch.testing.assert_close(restored[key], sim_state[key])


def test_flatten_unflatten_round_trip_simple():
    """Verify flatten/unflatten round-trip works for simple state."""
    from diffaero_common.adapters.flatten import flatten_sim_state, unflatten_sim_state

    sim_state = {
        "position_w": torch.randn(2, 3),
        "quaternion_w": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "linear_velocity_w": torch.randn(2, 3),
        "angular_velocity_b": torch.randn(2, 3),
    }
    flat = flatten_sim_state(sim_state, model_name="simple")
    restored = unflatten_sim_state(flat, model_name="simple")

    for key in sim_state:
        assert key in restored, f"Key '{key}' missing after unflatten"
        torch.testing.assert_close(restored[key], sim_state[key])


def test_pmd_sim_state_common_fields():
    """Verify pmd sim_state has all common fields required by the contract."""
    from diffaero_common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="pmd")

    assert "position_w" in state
    assert "linear_velocity_w" in state
    assert "target_position_w" in state
    assert "last_action" in state
    assert "progress" in state
    assert "step_count" in state


def test_pmc_sim_state_common_fields():
    """Verify pmc sim_state has all common fields required by the contract."""
    from diffaero_common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="pmc")

    assert "position_w" in state
    assert "linear_velocity_w" in state
    assert "target_position_w" in state
    assert "last_action" in state
    assert "progress" in state
    assert "step_count" in state


def test_simple_sim_state_common_fields():
    """Verify simple sim_state has all common fields required by the contract."""
    from diffaero_common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="simple")

    assert "position_w" in state
    assert "linear_velocity_w" in state
    assert "target_position_w" in state
    assert "last_action" in state
    assert "progress" in state
    assert "step_count" in state
