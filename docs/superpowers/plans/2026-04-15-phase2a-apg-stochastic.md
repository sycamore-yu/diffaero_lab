# Phase 2A APG_stochastic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stochastic-policy version of APG on top of the completed PhysX `Isaac-Drone-Racing-Direct-v0` path, with working tests and a one-iteration smoke command.

**Architecture:** Reuse the existing `DifferentialEnvAdapter`, environment contract, and APG trainer structure from Phase 1. Add a stochastic actor head, log-prob/entropy handling, and a separate APG_stochastic trainer/script/config path without changing the existing deterministic APG path.

**Tech Stack:** IsaacLab `DirectRLEnv`, PyTorch distributions, existing `diffaero_lab.algo` APG modules, `refer/IsaacLab/isaaclab.sh -p`, pytest.

---

## File map

### Create

- `source/diffaero_lab/diffaero_lab/algo/algorithms/apg_stochastic.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/apg_stochastic_trainer.py`
- `source/diffaero_lab/diffaero_lab/algo/wrappers/apg_stochastic.py`
- `source/diffaero_lab/diffaero_lab/algo/configs/apg_stochastic/drone_racing.yaml`
- `scripts/differential/train_apg_stochastic.py`
- `tests/test_apg_stochastic.py`

### Modify

- `source/diffaero_lab/diffaero_lab/algo/algorithms/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/wrappers/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/__init__.py`
- `source/diffaero_lab/docs/CHANGELOG.rst`
- `source/diffaero_lab/config/extension.toml`

## Scope guard

This plan only adds `APG_stochastic` to the existing PhysX environment and current APG infrastructure.

Out of scope:

- SHAC
- SHA2C
- point-mass / simplified backends
- Warp / Newton execution
- changes to the RL baseline path

### Task 1: Add failing stochastic-policy tests

**Files:**
- Create: `tests/test_apg_stochastic.py`

- [ ] **Step 1: Write a failing test for stochastic action sampling**

```python
def test_apg_stochastic_actor_outputs_action_and_log_prob():
    from diffaero_lab.algo.algorithms.apg_stochastic import APGStochastic, APGStochasticConfig

    policy = APGStochastic(APGStochasticConfig(), obs_dim=17, action_dim=4, device="cpu")
    obs = torch.zeros(8, 17)
    action, info = policy.act(obs)
    assert action.shape == (8, 4)
    assert "log_prob" in info
```

- [ ] **Step 2: Write a failing integration test for the adapter + stochastic trainer**

```python
def test_apg_stochastic_rollout_and_update_smoke(shared_env):
    from diffaero_lab.algo.algorithms.apg_stochastic import APGStochastic, APGStochasticConfig
    from diffaero_lab.algo.trainers.apg_stochastic_trainer import APGStochasticTrainer
```

- [ ] **Step 3: Run the tests to confirm the path is missing**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_stochastic.py -q`
Expected: FAIL with import errors for `apg_stochastic`

### Task 2: Implement the stochastic actor core

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/algorithms/apg_stochastic.py`
- Modify: `source/diffaero_lab/diffaero_lab/algo/algorithms/__init__.py`

- [ ] **Step 1: Add a dataclass config for stochastic APG**

```python
@dataclass
class APGStochasticConfig:
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    rollout_horizon: int = 32
    init_log_std: float = 0.0
    entropy_coef: float = 0.0
```

- [ ] **Step 2: Implement a Gaussian actor with reparameterized sampling**

```python
dist = torch.distributions.Normal(mean, std)
raw_action = dist.rsample()
action = torch.tanh(raw_action)
log_prob = dist.log_prob(raw_action).sum(dim=-1)
```

- [ ] **Step 3: Re-run the unit tests for stochastic action outputs**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_stochastic.py::test_apg_stochastic_actor_outputs_action_and_log_prob -q`
Expected: PASS

### Task 3: Implement stochastic trainer and wrapper integration

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/trainers/apg_stochastic_trainer.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/wrappers/apg_stochastic.py`
- Modify: `source/diffaero_lab/diffaero_lab/algo/trainers/__init__.py`
- Modify: `source/diffaero_lab/diffaero_lab/algo/wrappers/__init__.py`

- [ ] **Step 1: Reuse the deterministic APG trainer loop shape and add log-prob recording**

```python
action, policy_info = self.policy.act(batch.observations[OBS_POLICY])
loss = -(policy_info["log_prob"] * rewards.detach()).mean()
```

- [ ] **Step 2: Keep the existing `DifferentialEnvAdapter` as the default environment interface**

Use a thin re-export wrapper if needed, but keep one contract path.

- [ ] **Step 3: Run the trainer smoke test**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_stochastic.py::test_apg_stochastic_rollout_and_update_smoke -q`
Expected: PASS

### Task 4: Add configuration and runnable script

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/configs/apg_stochastic/drone_racing.yaml`
- Create: `scripts/differential/train_apg_stochastic.py`
- Modify: `source/diffaero_lab/diffaero_lab/algo/__init__.py`
- Modify: `source/diffaero_lab/config/extension.toml`

- [ ] **Step 1: Add a minimal config file**

```yaml
lr: 3e-4
max_grad_norm: 1.0
rollout_horizon: 32
init_log_std: 0.0
entropy_coef: 0.0
```

- [ ] **Step 2: Add a one-iteration launch script mirroring `train_apg.py`**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg_stochastic.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1`
Expected: exits 0 and prints one iteration of stochastic APG training

### Task 5: Final verification

**Files:**
- Modify: files above only if verification uncovers Task 2A defects

- [ ] **Step 1: Run the full stochastic test file**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_stochastic.py -q`
Expected: PASS

- [ ] **Step 2: Re-run deterministic APG tests to guard regressions**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_env_adapter.py -q`
Expected: PASS

- [ ] **Step 3: Run `train_apg_stochastic.py` smoke**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg_stochastic.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1`
Expected: PASS
