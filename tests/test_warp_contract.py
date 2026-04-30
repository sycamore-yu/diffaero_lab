# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failing tests for Warp/Newton backend-aware contract fields (Phase 2D Task 1).

These tests verify that the backend-aware contract fields (tensor_backend, write_mode,
quat_convention) are properly exposed for the Warp backend. They should FAIL until
Phase 2D Tasks 2 and 3 implement the Newton adapter and backend-aware sim_state.
"""

import torch


def test_newton_adapter_is_implemented():
    """Verify build_newton_adapter does not raise NotImplementedError.

    This test FAILS because build_newton_adapter() currently raises NotImplementedError.
    Expected failure reason: NotImplementedError - Warp/Newton adapter not yet implemented
    """
    from diffaero_lab.uav.adapters.newton import build_newton_adapter

    try:
        adapter = build_newton_adapter(cfg=None, device="cpu", backend="warp")
    except NotImplementedError:
        raise AssertionError(
            "build_newton_adapter raises NotImplementedError. Phase 2D Task 2 must implement the NewtonBackendAdapter."
        )


def test_newton_adapter_exposes_required_interface():
    """Verify NewtonBackendAdapter has the required methods for Warp execution.

    This test FAILS because build_newton_adapter raises NotImplementedError.
    Expected failure reason: NotImplementedError - Warp/Newton adapter not yet implemented
    """
    from diffaero_lab.uav.adapters.newton import build_newton_adapter

    adapter = build_newton_adapter(cfg=None, device="cpu", backend="warp")
    assert adapter is not None, "build_newton_adapter returned None"

    required_methods = [
        "process_action",
        "apply_to_sim",
        "read_base_state",
        "read_motor_state",
        "read_dynamics_info",
        "reset",
    ]
    for method in required_methods:
        assert hasattr(adapter, method), f"NewtonBackendAdapter missing method: {method}"


def test_warp_sim_state_has_tensor_backend_warp():
    """Verify build_sim_state accepts backend parameter and sets tensor_backend='warp'.

    This test FAILS because build_sim_state does not accept a backend parameter.
    Expected failure reason: TypeError - build_sim_state() got unexpected keyword argument 'backend'
    """
    from diffaero_lab.common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="quad", backend="warp")

    assert state["dynamics"]["tensor_backend"] == "warp", (
        f"Expected tensor_backend='warp' for Warp backend, got: {state['dynamics'].get('tensor_backend')}"
    )


def test_warp_sim_state_has_write_mode_field():
    """Verify sim_state dynamics metadata includes write_mode field for Warp backend.

    This test FAILS because build_sim_state does not accept a backend parameter.
    Expected failure reason: TypeError - build_sim_state() got unexpected keyword argument 'backend'
    """
    from diffaero_lab.common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="quad", backend="warp")

    assert "write_mode" in state["dynamics"], "sim_state['dynamics'] must contain 'write_mode' field"
    assert state["dynamics"]["write_mode"] in ("masked", "indexed"), (
        f"write_mode must be 'masked' or 'indexed', got: {state['dynamics']['write_mode']}"
    )


def test_warp_sim_state_preserves_quat_convention():
    """Verify sim_state dynamics metadata includes quat_convention for Warp backend.

    This test FAILS because build_sim_state does not accept a backend parameter.
    Expected failure reason: TypeError - build_sim_state() got unexpected keyword argument 'backend'
    """
    from diffaero_lab.common.adapters.sim_state import build_sim_state

    state = build_sim_state(batch_size=4, model_name="quad", backend="warp")

    assert "quat_convention" in state["dynamics"], "sim_state['dynamics'] must contain 'quat_convention' field"
    assert state["dynamics"]["quat_convention"] in ("wxyz", "xyzw"), (
        f"quat_convention must be 'wxyz' or 'xyzw', got: {state['dynamics']['quat_convention']}"
    )


def test_newton_adapter_read_dynamics_info_returns_warp_metadata():
    """Verify NewtonBackendAdapter.read_dynamics_info() returns correct Warp metadata.

    This test FAILS because build_newton_adapter raises NotImplementedError.
    Expected failure reason: NotImplementedError - Warp/Newton adapter not yet implemented
    """
    from diffaero_lab.uav.adapters.newton import build_newton_adapter

    adapter = build_newton_adapter(cfg=None, device="cpu", backend="warp")
    assert adapter is not None, "build_newton_adapter returned None"
    dyn_info = adapter.read_dynamics_info()

    required_fields = ["model_name", "state_layout_version", "tensor_backend", "write_mode", "quat_convention"]
    for field in required_fields:
        assert field in dyn_info, f"read_dynamics_info() must return '{field}' field"

    assert dyn_info["tensor_backend"] == "warp", (
        f"Expected tensor_backend='warp', got: {dyn_info.get('tensor_backend')}"
    )


def test_warp_dynamics_info_model_name_quad():
    """Verify Warp dynamics info correctly reports model_name='quad'.

    This test FAILS because build_newton_adapter raises NotImplementedError.
    Expected failure reason: NotImplementedError - Warp/Newton adapter not yet implemented
    """
    from diffaero_lab.uav.adapters.newton import build_newton_adapter

    adapter = build_newton_adapter(cfg=None, device="cpu", backend="warp")
    assert adapter is not None, "build_newton_adapter returned None"
    dyn_info = adapter.read_dynamics_info()

    assert dyn_info["model_name"] == "quad", f"Expected model_name='quad', got: {dyn_info.get('model_name')}"
