# Phase 2C Multi-Dynamics Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current `drone_racing` task from `quad`-only to `quad + pmd + pmc + simple` backends while preserving the current environment contract and Phase 1 training paths.

**Architecture:** Add the missing point-mass and simplified quad dynamics models to `diffaero_lab.uav`, add matching `dynamics_bridge` implementations to `diffaero_lab.env`, and keep `sim_state` contract compatibility through model-specific fields plus common fields. The environment task ID remains the same and switches backend through `DroneRacingEnvCfg.dynamics_model`.

**Tech Stack:** `diffaero_lab.uav` dynamics registry, `diffaero_lab.env` bridge layer, current common contract, pytest, IsaacLab smoke commands.

---

## File map

### Create

- `source/diffaero_lab/diffaero_lab/uav/dynamics/pointmass_discrete.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/pointmass_continuous.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/simplified_quadrotor.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/pointmass_discrete.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/pointmass_continuous.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/simplified_quad.py`
- `tests/test_multi_dynamics_registration.py`
- `tests/test_multi_dynamics_contract.py`

### Modify

- `source/diffaero_lab/diffaero_lab/uav/dynamics/registry.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env_cfg.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/sim_state.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/flatten.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/sim_state.py`
- `source/diffaero_lab/docs/CHANGELOG.rst`

## Scope guard

This plan adds model-switching support under the existing PhysX task.

Out of scope:

- Warp / Newton execution
- changes to SHAC/SHA2C objective logic

### Task 1: Add failing backend-selection tests

- [ ] **Step 1: Write a failing registration/config test for `dynamics_model=pmd|pmc|simple`**
- [ ] **Step 2: Write a failing contract test proving each backend still exports the required common sim_state keys**
- [ ] **Step 3: Run the tests and confirm the new backends are missing**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_multi_dynamics_registration.py tests/test_multi_dynamics_contract.py -q`
Expected: FAIL

### Task 2: Add `pmd` and `pmc` dynamics models

**Files:**
- Create: `pointmass_discrete.py`
- Create: `pointmass_continuous.py`
- Modify: `registry.py`

- [ ] **Step 1: Port `refer/diffaero/dynamics/pointmass.py` into two explicit backend modules**
- [ ] **Step 2: Register `pmd` and `pmc` in `DYNAMICS_REGISTRY`**
- [ ] **Step 3: Add matching bridge implementations**

### Task 3: Add `simple` backend

**Files:**
- Create: `simplified_quadrotor.py`
- Create: `dynamics_bridge/simplified_quad.py`

- [ ] **Step 1: Implement the simplified quad model with the same common state export shape**
- [ ] **Step 2: Add bridge support and selection by cfg**

### Task 4: Extend env cfg and sim_state schema

**Files:**
- Modify: `drone_racing_env_cfg.py`
- Modify: `drone_racing_env.py`
- Modify: `state/sim_state.py`
- Modify: common adapters

- [ ] **Step 1: Make `dynamics_model` runtime-selectable across all four backends**
- [ ] **Step 2: Preserve common sim_state keys and add backend-specific fields cleanly**
- [ ] **Step 3: Ensure flatten/unflatten supports all backends**

### Task 5: Final verification

- [ ] **Step 1: Run the new backend tests**
- [ ] **Step 2: Re-run `tests/test_drone_racing_registration.py` and `tests/test_apg_env_adapter.py`**
- [ ] **Step 3: Run smoke commands for at least `quad`, `pmd`, and `pmc`**
