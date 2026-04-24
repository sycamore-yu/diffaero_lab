# Phase 2B SHAC and SHA2C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SHAC and SHA2C to the current PhysX `drone_racing` stack using the existing `critic` observation path and differential environment adapter.

**Architecture:** Build a shared actor-critic base over the existing APG path, then add SHAC and SHA2C trainers that consume `observations[OBS_CRITIC]` and the current `task_terms`/`sim_state` contract. Keep the deterministic APG path intact and isolate critic-specific logic into new modules.

**Tech Stack:** PyTorch actor-critic models, existing `DifferentialEnvAdapter`, current `drone_racing` critic observations, pytest, IsaacLab launch scripts.

---

## File map

### Create

- `source/diffaero_lab/diffaero_lab/algo/algorithms/shac.py`
- `source/diffaero_lab/diffaero_lab/algo/algorithms/sha2c.py`
- `source/diffaero_lab/diffaero_lab/algo/algorithms/actor_critic.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/shac_trainer.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/sha2c_trainer.py`
- `source/diffaero_lab/diffaero_lab/algo/configs/shac/drone_racing.yaml`
- `source/diffaero_lab/diffaero_lab/algo/configs/sha2c/drone_racing.yaml`
- `scripts/differential/train_shac.py`
- `scripts/differential/train_sha2c.py`
- `tests/test_shac.py`
- `tests/test_sha2c.py`

### Modify

- `source/diffaero_lab/diffaero_lab/algo/algorithms/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/__init__.py`
- `source/diffaero_lab/docs/CHANGELOG.rst`
- `source/diffaero_lab/config/extension.toml`

## Scope guard

This plan adds only critic-based algorithms on top of the already working PhysX environment and current APG infrastructure.

Out of scope:

- new dynamics backends
- Warp / Newton runtime
- RL baseline changes

### Task 1: Add failing critic-path tests

**Files:**
- Create: `tests/test_shac.py`
- Create: `tests/test_sha2c.py`

- [ ] **Step 1: Write failing SHAC tests for critic-state consumption**
- [ ] **Step 2: Write failing SHA2C tests for asymmetric actor/critic path**
- [ ] **Step 3: Run the tests to confirm the algorithms are missing**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_shac.py tests/test_sha2c.py -q`
Expected: FAIL with import errors

### Task 2: Implement shared actor-critic base

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/algorithms/actor_critic.py`

- [ ] **Step 1: Add a minimal shared actor-critic module with separate policy and value heads**
- [ ] **Step 2: Use `OBS_POLICY` for actor and `OBS_CRITIC` for critic inputs**
- [ ] **Step 3: Add unit coverage for forward outputs**

### Task 3: Implement SHAC

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/algorithms/shac.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/trainers/shac_trainer.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/configs/shac/drone_racing.yaml`
- Create: `scripts/differential/train_shac.py`

- [ ] **Step 1: Implement SHAC config and actor-critic wrapper**
- [ ] **Step 2: Implement rollout with value bootstrap and policy/value losses**
- [ ] **Step 3: Run SHAC smoke test**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_shac.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1`
Expected: PASS

### Task 4: Implement SHA2C

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/algorithms/sha2c.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/trainers/sha2c_trainer.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/configs/sha2c/drone_racing.yaml`
- Create: `scripts/differential/train_sha2c.py`

- [ ] **Step 1: Implement asymmetric actor-critic update path using current `critic` observations**
- [ ] **Step 2: Reuse the shared actor-critic module where possible**
- [ ] **Step 3: Run SHA2C smoke test**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_sha2c.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1`
Expected: PASS

### Task 5: Final verification

- [ ] **Step 1: Run `tests/test_shac.py` and `tests/test_sha2c.py`**
- [ ] **Step 2: Re-run `tests/test_apg_env_adapter.py` to guard regressions**
- [ ] **Step 3: Re-run both smoke scripts on the current `quad` backend**
- [ ] **Step 4: After Phase 2C lands, re-run both smoke scripts once against at least one non-quad backend to confirm critic-path compatibility is preserved**
