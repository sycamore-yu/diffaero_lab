# Phase 3 Drone Racing Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Phase 1/2 placeholder task semantics in `drone_racing` with real progress, gate-pass, collision, tracking, target-position, and motor-state behavior that works consistently across PhysX and Warp routes.

**Architecture:** Keep the current `diffaero_lab` structure and algorithm stack unchanged. Phase 3 focuses on the environment semantics layer: `diffaero_lab.env` will compute real racing terms and export them through the existing contract, while `diffaero_lab.uav` will provide the motor/dynamics state needed to make those terms meaningful. The public task IDs, backend registry, and differential algorithm entry points remain the same.

**Tech Stack:** IsaacLab `DirectRLEnv`, current `dynamics_bridge` layer, shared contract in `diffaero_lab.common`, current `diffaero_lab.uav` dynamics adapters, pytest, IsaacLab runner commands.

---

## Scope guard

This plan only hardens the existing `drone_racing` task semantics.

In scope:

- real `progress`
- real `gate_pass`
- real `collision`
- real `tracking_error`
- meaningful `target_position_w`
- non-placeholder `motor_omega` path where backend supports it
- alignment of reward terms, task terms, and `sim_state`

Out of scope:

- new algorithms
- new tasks beyond `drone_racing`
- new dynamics backends
- new Warp/Newton architectural work beyond semantics parity

## File map

### Modify

- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/rewards.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/observations.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/resets.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/terminations.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/sim_state.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/task_terms.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env_cfg.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/quad.py`
- `source/diffaero_lab/diffaero_lab/uav/adapters/newton.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/sim_state.py`
- `tests/test_drone_racing_registration.py`
- `tests/test_apg_env_adapter.py`

### Create

- `tests/test_drone_racing_semantics.py`
- `tests/test_drone_racing_warp_semantics.py`

## Task breakdown

### Task 1: Add failing semantics tests

**Files:**
- Create: `tests/test_drone_racing_semantics.py`
- Create: `tests/test_drone_racing_warp_semantics.py`

- [ ] **Step 1: Write failing tests for real task terms on PhysX**

Add failing tests covering:

- `progress` changes with forward motion toward target / gate
- `tracking_error` becomes non-zero when deviating from target line
- `gate_pass` changes when crossing a gate threshold
- `collision` changes when contact/termination condition is triggered
- `sim_state` includes non-placeholder `target_position_w`

- [ ] **Step 2: Write failing parity tests for Warp/Newton path**

The Warp tests should assert contract-level parity for semantics fields, not identical physics values.

- [ ] **Step 3: Run tests to verify the current placeholder implementation fails**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_semantics.py tests/test_drone_racing_warp_semantics.py -q`
Expected: FAIL because `progress`, `gate_pass`, `collision`, and `target_position_w` are still placeholders or missing.

### Task 2: Implement real target/gate state export

**Files:**
- Modify: `drone_racing_env.py`
- Modify: `state/sim_state.py`
- Modify: `mdp/observations.py`
- Modify: `drone_racing_env_cfg.py`

- [ ] **Step 1: Add explicit target / gate state to env runtime state**

The environment should maintain the current gate or target position in world frame.

- [ ] **Step 2: Export `target_position_w` through the env-level sim_state builder**

Match the current common adapter contract.

- [ ] **Step 3: Update policy/critic observations only if required for semantic correctness**

Keep existing observation dimensions stable unless a test proves a missing field.

- [ ] **Step 4: Run targeted sim_state tests**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_semantics.py -k 'target_position or sim_state' -q`
Expected: PASS

### Task 3: Implement real reward terms and terminations

**Files:**
- Modify: `mdp/rewards.py`
- Modify: `mdp/terminations.py`
- Modify: `state/task_terms.py`

- [ ] **Step 1: Replace placeholder `progress` and `tracking_error` computation with geometry-based logic**

- [ ] **Step 2: Add real `gate_pass` and `collision` terms**

- [ ] **Step 3: Keep reward aggregation and task term exports aligned**

- [ ] **Step 4: Run semantics tests for reward/task term behavior**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_semantics.py -k 'progress or tracking or gate or collision' -q`
Expected: PASS

### Task 4: Implement meaningful motor-state semantics

**Files:**
- Modify: `dynamics_bridge/quad.py`
- Modify: `source/diffaero_lab/diffaero_lab/uav/adapters/newton.py`
- Modify: `state/sim_state.py`

- [ ] **Step 1: Replace zero-placeholder `motor_omega` on PhysX with a consistent approximation derived from control action or bridge state**

- [ ] **Step 2: Give Warp/Newton adapter the same semantic output shape**

- [ ] **Step 3: Keep backend parity at contract level**

- [ ] **Step 4: Run motor-state tests on both PhysX and Warp paths**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_semantics.py tests/test_drone_racing_warp_semantics.py -k 'motor' -q`
Expected: PASS

### Task 5: Reconcile APG / SHAC / SHA2C with real semantics

**Files:**
- Modify: `tests/test_apg_env_adapter.py`
- Modify: `tests/test_shac.py`
- Modify: `tests/test_sha2c.py`

- [ ] **Step 1: Update expectations that previously tolerated placeholder zeros**

- [ ] **Step 2: Verify existing algorithms still consume the richer task terms and sim_state without regressions**

- [ ] **Step 3: Run algorithm regression tests**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_env_adapter.py tests/test_shac.py tests/test_sha2c.py -q`
Expected: PASS

### Task 6: Final verification

**Files:**
- Modify only the files above if verification uncovers Phase 3 defects

- [ ] **Step 1: Run all new semantics tests**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_semantics.py tests/test_drone_racing_warp_semantics.py -q`
Expected: PASS

- [ ] **Step 2: Re-run existing registration and contract tests**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_registration.py tests/test_multi_dynamics_contract.py tests/test_warp_contract.py -q`
Expected: PASS

- [ ] **Step 3: Re-run differential smoke commands on PhysX and Warp**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1 && refer/IsaacLab/env_isaaclab/bin/python -c "print('PHASE3_PHYSX_OK')"`

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg.py --task Isaac-Drone-Racing-Direct-Warp-v0 --max_iterations 1 && refer/IsaacLab/env_isaaclab/bin/python -c "print('PHASE3_WARP_OK')"`

Expected: both commands exit 0 and print the marker.
