# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for diffaero_common contract constants, capability flags, and adapters."""

import torch


def test_contract_keys_match_design():
    """Verify all observation and extras keys match the approved design."""
    from diffaero_common.keys import (
        OBS_POLICY,
        OBS_CRITIC,
        EXTRA_TASK_TERMS,
        EXTRA_SIM_STATE,
        EXTRA_CAPABILITIES,
        EXTRA_DYNAMICS_INFO,
        EXTRA_RESET_STATE,
        EXTRA_TERMINAL_STATE,
    )

    assert OBS_POLICY == "policy"
    assert OBS_CRITIC == "critic"
    assert EXTRA_TASK_TERMS == "task_terms"
    assert EXTRA_SIM_STATE == "sim_state"
    assert EXTRA_CAPABILITIES == "capabilities"
    assert EXTRA_DYNAMICS_INFO == "dynamics"
    assert EXTRA_RESET_STATE == "state_before_reset"
    assert EXTRA_TERMINAL_STATE == "terminal_state"


def test_capability_flags_match_design():
    """Verify all capability flags match the approved design."""
    from diffaero_common.capabilities import (
        SUPPORTS_CRITIC_STATE,
        SUPPORTS_SIM_STATE,
        SUPPORTS_TASK_TERMS,
        SUPPORTS_TERMINAL_STATE,
        SUPPORTS_DIFFERENTIAL_ROLLOUT,
        SUPPORTS_DYNAMICS_SWITCH,
        SUPPORTS_WARP_BACKEND,
    )

    assert SUPPORTS_CRITIC_STATE == "supports_critic_state"
    assert SUPPORTS_SIM_STATE == "supports_sim_state"
    assert SUPPORTS_TASK_TERMS == "supports_task_terms"
    assert SUPPORTS_TERMINAL_STATE == "supports_terminal_state"
    assert SUPPORTS_DIFFERENTIAL_ROLLOUT == "supports_differential_rollout"
    assert SUPPORTS_DYNAMICS_SWITCH == "supports_dynamics_switch"
    assert SUPPORTS_WARP_BACKEND == "supports_warp_backend"


def test_task_term_names_match_design():
    """Verify standard task term names are defined."""
    from diffaero_common.terms import (
        TERM_PROGRESS,
        TERM_TRACKING_ERROR,
        TERM_GATE_PASS,
        TERM_COLLISION,
        TERM_TERMINAL,
        TERM_CONTROL_EFFORT,
        TERM_CONTROL_SMOOTHNESS,
        TERM_ANGULAR_RATE,
        TERM_TIME_PENALTY,
    )

    assert TERM_PROGRESS == "progress"
    assert TERM_TRACKING_ERROR == "tracking_error"
    assert TERM_GATE_PASS == "gate_pass"
    assert TERM_COLLISION == "collision"
    assert TERM_TERMINAL == "terminal"
    assert TERM_CONTROL_EFFORT == "control_effort"
    assert TERM_CONTROL_SMOOTHNESS == "control_smoothness"
    assert TERM_ANGULAR_RATE == "angular_rate"
    assert TERM_TIME_PENALTY == "time_penalty"


def test_flatten_round_trip_for_quad_state():
    """Verify flatten/unflatten round-trip preserves motor_omega shape for quad model."""
    from diffaero_common.adapters.flatten import flatten_sim_state, unflatten_sim_state

    sim_state = {
        "position_w": torch.zeros(2, 3),
        "quaternion_w": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "linear_velocity_w": torch.zeros(2, 3),
        "angular_velocity_b": torch.zeros(2, 3),
        "motor_omega": torch.zeros(2, 4),
    }
    flat = flatten_sim_state(sim_state, model_name="quad")
    restored = unflatten_sim_state(flat, model_name="quad")
    assert restored["motor_omega"].shape == (2, 4)


def test_flatten_preserves_all_quad_fields():
    """Verify flatten/unflatten round-trip preserves all quad fields."""
    from diffaero_common.adapters.flatten import flatten_sim_state, unflatten_sim_state

    position_w = torch.randn(2, 3)
    quaternion_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    linear_velocity_w = torch.randn(2, 3)
    angular_velocity_b = torch.randn(2, 3)
    motor_omega = torch.randn(2, 4)

    sim_state = {
        "position_w": position_w,
        "quaternion_w": quaternion_w,
        "linear_velocity_w": linear_velocity_w,
        "angular_velocity_b": angular_velocity_b,
        "motor_omega": motor_omega,
    }
    flat = flatten_sim_state(sim_state, model_name="quad")
    restored = unflatten_sim_state(flat, model_name="quad")

    assert restored["position_w"].shape == (2, 3)
    assert restored["quaternion_w"].shape == (2, 4)
    assert restored["linear_velocity_w"].shape == (2, 3)
    assert restored["angular_velocity_b"].shape == (2, 3)
    assert restored["motor_omega"].shape == (2, 4)
    torch.testing.assert_close(restored["position_w"], position_w)
    torch.testing.assert_close(restored["quaternion_w"], quaternion_w)
    torch.testing.assert_close(restored["linear_velocity_w"], linear_velocity_w)
    torch.testing.assert_close(restored["angular_velocity_b"], angular_velocity_b)
    torch.testing.assert_close(restored["motor_omega"], motor_omega)


def test_sim_state_builder_returns_required_fields():
    """Verify sim_state builder returns all required common fields plus quad-specific fields."""
    from diffaero_common.adapters.sim_state import build_sim_state

    batch_size = 4
    state = build_sim_state(batch_size=batch_size, model_name="quad")

    # Common fields
    assert "position_w" in state
    assert "linear_velocity_w" in state
    assert "target_position_w" in state
    assert "last_action" in state
    assert "progress" in state
    assert "step_count" in state

    # Quad-specific fields
    assert "quaternion_w" in state
    assert "angular_velocity_b" in state
    assert "motor_omega" in state

    # Dynamics metadata
    assert "dynamics" in state
    assert state["dynamics"]["model_name"] == "quad"
    assert "state_layout_version" in state["dynamics"]
    assert "quat_convention" in state["dynamics"]
    assert state["dynamics"]["quat_convention"] == "wxyz"

    # Shapes
    assert state["position_w"].shape == (batch_size, 3)
    assert state["linear_velocity_w"].shape == (batch_size, 3)
    assert state["quaternion_w"].shape == (batch_size, 4)
    assert state["angular_velocity_b"].shape == (batch_size, 3)
    assert state["motor_omega"].shape == (batch_size, 4)
