# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-dynamics backend registration.

These tests verify that the pmd (point-mass discrete), pmc (point-mass continuous),
and simple (simplified quadrotor) backends are properly registered alongside quad.
They fail until the backends are implemented per Phase 2C.
"""

import torch


def test_pmd_is_registered_in_dynamics_registry():
    """Verify pmd (point-mass discrete) backend is registered in DYNAMICS_REGISTRY."""
    from diffaero_lab.uav.dynamics.registry import DYNAMICS_REGISTRY

    assert "pmd" in DYNAMICS_REGISTRY, "pmd backend must be registered in DYNAMICS_REGISTRY"


def test_pmc_is_registered_in_dynamics_registry():
    """Verify pmc (point-mass continuous) backend is registered in DYNAMICS_REGISTRY."""
    from diffaero_lab.uav.dynamics.registry import DYNAMICS_REGISTRY

    assert "pmc" in DYNAMICS_REGISTRY, "pmc backend must be registered in DYNAMICS_REGISTRY"


def test_simple_is_registered_in_dynamics_registry():
    """Verify simple (simplified quadrotor) backend is registered in DYNAMICS_REGISTRY."""
    from diffaero_lab.uav.dynamics.registry import DYNAMICS_REGISTRY

    assert "simple" in DYNAMICS_REGISTRY, "simple backend must be registered in DYNAMICS_REGISTRY"


def test_pmd_dynamics_model_entry_in_env_cfg():
    """Verify DroneRacingEnvCfg supports dynamics_model='pmd' selection."""
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    cfg = DroneRacingEnvCfg()
    # dynamics_model field must exist and be settable
    assert hasattr(cfg, "dynamics_model"), "DroneRacingEnvCfg must have dynamics_model field"
    cfg.dynamics_model = "pmd"
    assert cfg.dynamics_model == "pmd"


def test_pmc_dynamics_model_entry_in_env_cfg():
    """Verify DroneRacingEnvCfg supports dynamics_model='pmc' selection."""
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    cfg = DroneRacingEnvCfg()
    assert hasattr(cfg, "dynamics_model"), "DroneRacingEnvCfg must have dynamics_model field"
    cfg.dynamics_model = "pmc"
    assert cfg.dynamics_model == "pmc"


def test_simple_dynamics_model_entry_in_env_cfg():
    """Verify DroneRacingEnvCfg supports dynamics_model='simple' selection."""
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    cfg = DroneRacingEnvCfg()
    assert hasattr(cfg, "dynamics_model"), "DroneRacingEnvCfg must have dynamics_model field"
    cfg.dynamics_model = "simple"
    assert cfg.dynamics_model == "simple"


def test_build_dynamics_pmd_succeeds():
    """Verify build_dynamics('pmd', ...) returns a dynamics model instance."""
    from diffaero_lab.uav.dynamics.registry import build_dynamics

    # This will raise ValueError if pmd is not registered
    class FakeCfg:
        pass

    model = build_dynamics("pmd", cfg=FakeCfg(), device="cpu")
    assert model is not None, "build_dynamics('pmd') must return a valid model"


def test_build_dynamics_pmc_succeeds():
    """Verify build_dynamics('pmc', ...) returns a dynamics model instance."""
    from diffaero_lab.uav.dynamics.registry import build_dynamics

    class FakeCfg:
        pass

    model = build_dynamics("pmc", cfg=FakeCfg(), device="cpu")
    assert model is not None, "build_dynamics('pmc') must return a valid model"


def test_build_dynamics_simple_succeeds():
    """Verify build_dynamics('simple', ...) returns a dynamics model instance."""
    from diffaero_lab.uav.dynamics.registry import build_dynamics

    class FakeCfg:
        pass

    model = build_dynamics("simple", cfg=FakeCfg(), device="cpu")
    assert model is not None, "build_dynamics('simple') must return a valid model"


def test_quad_still_registered():
    """Verify the original quad backend remains registered after adding new backends."""
    from diffaero_lab.uav.dynamics.registry import DYNAMICS_REGISTRY

    assert "quad" in DYNAMICS_REGISTRY, "quad backend must remain registered"
