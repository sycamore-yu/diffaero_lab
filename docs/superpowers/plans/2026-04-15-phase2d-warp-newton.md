# Phase 2D Warp and Newton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first working Warp/Newton execution route for `drone_racing`, including backend-aware contracts, a Warp task ID, and a smoke-tested APG path on the experimental backend.

**Architecture:** Build on the completed multi-backend Phase 2C state by adding a Warp/Newton task registration target, backend-aware adapter logic, and contract metadata for `tensor_backend`, `write_mode`, and quaternion convention. Keep the PhysX route intact and isolate Warp/Newton differences in adapters and task registration.

**Tech Stack:** IsaacLab experimental Newton integration, Warp arrays, `diffaero_lab.uav.adapters.newton`, current APG path, pytest, IsaacLab runner commands.

---

## File map

### Create

- `tests/test_warp_registration.py`
- `tests/test_warp_contract.py`

### Modify

- `source/diffaero_lab/diffaero_lab/uav/adapters/newton.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env_cfg.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/sim_state.py`
- `source/diffaero_lab/diffaero_lab/common/capabilities.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/sim_state.py`
- `source/diffaero_lab/diffaero_lab/algo/wrappers/env_adapter.py`
- `scripts/differential/train_apg.py`
- `source/diffaero_lab/docs/CHANGELOG.rst`

## Scope guard

This plan only adds the first working Warp/Newton APG route.

Out of scope:

- SHAC/SHA2C on Warp unless they already work with the same adapter changes
- new RL baseline wrappers for Warp

### Task 1: Add failing Warp/Newton contract tests

- [ ] **Step 1: Write a failing registration test for `Isaac-Drone-Racing-Direct-Warp-v0`**
- [ ] **Step 2: Write a failing contract test for `tensor_backend`, `write_mode`, and `quat_convention`**
- [ ] **Step 3: Run the tests and confirm the Warp route is missing**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_warp_registration.py tests/test_warp_contract.py -q`
Expected: FAIL

### Task 2: Implement the Newton adapter and task registration

**Files:**
- Modify: `source/diffaero_lab/diffaero_lab/uav/adapters/newton.py`
- Modify: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/__init__.py`
- Modify: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env_cfg.py`

- [ ] **Step 1: Replace the `NotImplementedError` placeholder with a minimal backend adapter**

The minimal adapter must satisfy this interface:

```python
def build_newton_adapter(cfg, device, backend: str = "warp") -> NewtonBackendAdapter:
    ...


class NewtonBackendAdapter:
    def process_action(self, action: torch.Tensor) -> None: ...
    def apply_to_sim(self) -> None: ...
    def read_base_state(self) -> dict[str, torch.Tensor | object]: ...
    def read_motor_state(self) -> dict[str, torch.Tensor | object]: ...
    def read_dynamics_info(self) -> dict[str, str | int]: ...
    def reset(self, env_ids: torch.Tensor | None) -> None: ...
```

Required `read_dynamics_info()` fields for the Warp route:

- `model_name`
- `state_layout_version`
- `tensor_backend = "warp"`
- `write_mode`
- `quat_convention`
- [ ] **Step 2: Register `Isaac-Drone-Racing-Direct-Warp-v0` with a Warp/Newton cfg path**
- [ ] **Step 3: Ensure `tensor_backend="warp"` and `write_mode="masked"|"indexed"` are exposed through the contract**

### Task 3: Make `sim_state` and env adapter backend-aware

**Files:**
- Modify: `state/sim_state.py`
- Modify: common sim_state adapter
- Modify: `env_adapter.py`

- [ ] **Step 1: Normalize quaternion convention explicitly (`wxyz` or `xyzw`)**
- [ ] **Step 2: Handle Warp arrays without assuming torch-native buffers**
- [ ] **Step 3: Keep the existing PhysX path untouched**

### Task 4: Re-enable APG smoke on Warp/Newton

**Files:**
- Modify: `scripts/differential/train_apg.py`

- [ ] **Step 1: Add a task/backend-compatible APG smoke path for `Isaac-Drone-Racing-Direct-Warp-v0`**
- [ ] **Step 1: Extend the existing `scripts/differential/train_apg.py` to dispatch by task ID / backend metadata rather than creating a second APG entry script**
- [ ] **Step 2: Keep the current PhysX path unchanged and gate Warp/Newton handling behind `Isaac-Drone-Racing-Direct-Warp-v0` or equivalent backend metadata from the env cfg / contract**
- [ ] **Step 3: Run a one-iteration Warp/Newton APG smoke command**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg.py --task Isaac-Drone-Racing-Direct-Warp-v0 --max_iterations 1`
Expected: PASS

### Task 5: Final verification

- [ ] **Step 1: Run Warp-specific tests**
- [ ] **Step 2: Re-run PhysX APG smoke to guard regressions**
- [ ] **Step 3: Record any remaining experimental-backend limitations in changelogs**
