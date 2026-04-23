# PhysX Drone Racing Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable migration slice for DiffAero on IsaacLab: `Isaac-Drone-Racing-Direct-v0` on PhysX, one IsaacLab-integrated RL baseline path, and an `APG` minimal differentiable training path.

**Architecture:** This phase builds the consolidated `diffaero_lab` extension under `source/`, with `diffaero_lab.env`, `diffaero_lab.algo`, `diffaero_lab.common`, and `diffaero_lab.uav` as sub-packages. The environment follows IsaacLab `DirectRLEnv` conventions with `policy` and optional `critic` observations, while differential learning reads `task_terms` and `sim_state` from `extras` through a custom wrapper. The physics backend for this phase is PhysX only; Warp/Newton stays explicitly out of scope except for adapter placeholders and contract flags.

**Tech Stack:** IsaacLab own-project extensions, `DirectRLEnv`, `gym.register`, `isaaclab_rl` integrated libraries, `skrl` PPO baseline, PyTorch, Hydra overrides, PhysX backend, UV environment at `refer/IsaacLab/env_isaaclab`, runner commands via `refer/IsaacLab/isaaclab.sh -p`.

---

## Scope guard

This plan only covers the first executable sub-project from the approved design:

1. `Isaac-Drone-Racing-Direct-v0`
2. PhysX backend only
3. IsaacLab-integrated RL baseline path using `skrl` PPO first
4. `APG` minimal differentiable rollout and backward path

Out of scope for this plan:

- `APG_stochastic`
- `SHAC`
- `SHA2C`
- Warp / Newton execution
- `Isaac-Drone-Racing-Direct-Warp-v0`
- point-mass backends beyond placeholder file scaffolds

## File map

### Create

- `source/diffaero_lab/diffaero_lab/common/config/extension.toml`
- `source/diffaero_lab/diffaero_lab/common/setup.py`
- `source/diffaero_lab/diffaero_lab/common/docs/CHANGELOG.rst`
- `source/diffaero_lab/diffaero_lab/common/__init__.py`
- `source/diffaero_lab/diffaero_lab/common/keys.py`
- `source/diffaero_lab/diffaero_lab/common/capabilities.py`
- `source/diffaero_lab/diffaero_lab/common/terms.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/__init__.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/flatten.py`
- `source/diffaero_lab/diffaero_lab/common/adapters/sim_state.py`
- `source/diffaero_lab/diffaero_lab/uav/config/extension.toml`
- `source/diffaero_lab/diffaero_lab/uav/setup.py`
- `source/diffaero_lab/diffaero_lab/uav/docs/CHANGELOG.rst`
- `source/diffaero_lab/diffaero_lab/uav/__init__.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/__init__.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/base.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/registry.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/allocation.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/motor.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/quadrotor.py`
- `source/diffaero_lab/diffaero_lab/uav/dynamics/controller.py`
- `source/diffaero_lab/diffaero_lab/uav/adapters/__init__.py`
- `source/diffaero_lab/diffaero_lab/uav/adapters/isaaclab.py`
- `source/diffaero_lab/diffaero_lab/uav/adapters/newton.py`
- `source/diffaero_lab/diffaero_lab/env/config/extension.toml`
- `source/diffaero_lab/diffaero_lab/env/setup.py`
- `source/diffaero_lab/diffaero_lab/env/docs/CHANGELOG.rst`
- `source/diffaero_lab/diffaero_lab/env/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env_cfg.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/base.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/quad.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/policy.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/critic.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/sim_state.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/task_terms.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/rewards.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/terminations.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/resets.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/observations.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/agents/__init__.py`
- `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/agents/skrl_ppo_cfg.yaml`
- `source/diffaero_lab/diffaero_lab/algo/config/extension.toml`
- `source/diffaero_lab/diffaero_lab/algo/setup.py`
- `source/diffaero_lab/diffaero_lab/algo/docs/CHANGELOG.rst`
- `source/diffaero_lab/diffaero_lab/algo/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/algorithms/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/algorithms/apg.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/trainers/apg_trainer.py`
- `source/diffaero_lab/diffaero_lab/algo/wrappers/__init__.py`
- `source/diffaero_lab/diffaero_lab/algo/wrappers/env_adapter.py`
- `source/diffaero_lab/diffaero_lab/algo/wrappers/apg.py`
- `source/diffaero_lab/diffaero_lab/algo/configs/apg/drone_racing.yaml`
- `scripts/reinforcement_learning/skrl/train.py`
- `scripts/reinforcement_learning/skrl/play.py`
- `scripts/differential/train_apg.py`
- `tests/test_drone_racing_registration.py`
- `tests/test_diffaero_lab_common_contracts.py`
- `tests/test_apg_env_adapter.py`

### Modify

- `scripts/list_envs.py`
- `pyproject.toml`

### Reuse as donors

- `source/diffaero_lab/setup.py`
- `source/diffaero_lab/config/extension.toml`
- `source/diffaero_lab/diffaero_lab/tasks/direct/diffaero_lab/__init__.py`
- `scripts/skrl/train.py`
- `scripts/skrl/play.py`
- `refer/isaac_drone_racer/dynamics/allocation.py`
- `refer/isaac_drone_racer/dynamics/motor.py`
- `refer/diffaero/dynamics/base_dynamics.py`
- `refer/diffaero/dynamics/quadrotor.py`
- `refer/diffaero/dynamics/controller.py`
- `refer/diffaero/algo/APG.py`

## Execution notes

- All validation commands should run from repository root: `/home/tong/tongworkspace/diffaero_lab`
- Use IsaacLab runner commands through `refer/IsaacLab/isaaclab.sh -p ...`
- The UV environment referenced by the project is `refer/IsaacLab/env_isaaclab`
- The RL wrapper from `isaaclab_rl` must remain the last wrapper in the RL chain
- This plan assumes no commit happens unless explicitly requested during execution

### Task 1: Scaffold four extensions and shared command surface

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/common/**`
- Create: `source/diffaero_lab/diffaero_lab/uav/**`
- Create: `source/diffaero_lab/diffaero_lab/env/**`
- Create: `source/diffaero_lab/diffaero_lab/algo/**`
- Modify: `pyproject.toml`
- Modify: `scripts/list_envs.py`

- [ ] **Step 1: Copy the template extension metadata into four new extension roots**

Use `source/diffaero_lab/setup.py` and `source/diffaero_lab/config/extension.toml` as the structural donor.

```python
# source/diffaero_lab/diffaero_lab/common/setup.py
setup(
    name="diffaero_lab.common",
    packages=["diffaero_lab.common"],
    install_requires=["psutil"],
)
```

- [ ] **Step 2: Give each extension a real package identity**

Set titles, module names, versions, and dependency blocks so each package can be installed independently.

```toml
# source/diffaero_lab/diffaero_lab/env/config/extension.toml
[package]
title = "DiffAero Environment Extension"
version = "0.1.0"

[dependencies]
"isaaclab" = {}
"isaaclab_rl" = {}
"diffaero_lab.common" = {}
"diffaero_lab.uav" = {}

[[python.module]]
name = "diffaero_lab.env"
```

- [ ] **Step 3: Update root static analysis coverage to include new source trees**

`pyproject.toml` already includes `source` and `scripts`, so this step is a verification edit only if package-specific excludes or docs paths need refinement.

Run: `python -c "from pathlib import Path; print(sorted(p.name for p in Path('source').iterdir()))"`
Expected: output includes `diffaero_lab`

- [ ] **Step 4: Make `scripts/list_envs.py` import the new task package and search the new task prefix**

```python
import diffaero_lab.env.tasks  # noqa: F401

if "Isaac-Drone-Racing-" in task_spec.id:
    ...
```

- [ ] **Step 5: Run a registration smoke command before any environment logic exists**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/list_envs.py --keyword Drone-Racing`
Expected: command runs without import errors; task count may still be zero at this point.

### Task 2: Implement the shared contract extension

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/common/keys.py`
- Create: `source/diffaero_lab/diffaero_lab/common/capabilities.py`
- Create: `source/diffaero_lab/diffaero_lab/common/terms.py`
- Create: `source/diffaero_lab/diffaero_lab/common/adapters/flatten.py`
- Create: `source/diffaero_lab/diffaero_lab/common/adapters/sim_state.py`
- Test: `tests/test_diffaero_lab_common_contracts.py`

- [ ] **Step 1: Write the failing contract tests first**

```python
def test_contract_keys_match_design():
    from diffaero_lab.common.keys import OBS_POLICY, OBS_CRITIC, EXTRA_SIM_STATE

    assert OBS_POLICY == "policy"
    assert OBS_CRITIC == "critic"
    assert EXTRA_SIM_STATE == "sim_state"


def test_flatten_round_trip_for_quad_state():
    from diffaero_lab.common.adapters.flatten import flatten_sim_state, unflatten_sim_state

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
```

- [ ] **Step 2: Run the targeted tests to confirm the module is still missing**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_diffaero_lab_common_contracts.py -q`
Expected: FAIL with import or attribute errors

- [ ] **Step 3: Implement constants and schema helpers**

```python
# source/diffaero_lab/diffaero_lab/common/keys.py
OBS_POLICY = "policy"
OBS_CRITIC = "critic"
EXTRA_TASK_TERMS = "task_terms"
EXTRA_SIM_STATE = "sim_state"
EXTRA_CAPABILITIES = "capabilities"
EXTRA_DYNAMICS_INFO = "dynamics"
EXTRA_RESET_STATE = "state_before_reset"
EXTRA_TERMINAL_STATE = "terminal_state"
```

```python
# source/diffaero_lab/diffaero_lab/common/capabilities.py
SUPPORTS_CRITIC_STATE = "supports_critic_state"
SUPPORTS_SIM_STATE = "supports_sim_state"
SUPPORTS_TASK_TERMS = "supports_task_terms"
SUPPORTS_TERMINAL_STATE = "supports_terminal_state"
SUPPORTS_DIFFERENTIAL_ROLLOUT = "supports_differential_rollout"
SUPPORTS_DYNAMICS_SWITCH = "supports_dynamics_switch"
SUPPORTS_WARP_BACKEND = "supports_warp_backend"
```

- [ ] **Step 4: Implement a minimal PhysX-first flatten adapter**

```python
_QUAD_LAYOUT = (
    ("position_w", 3),
    ("quaternion_w", 4),
    ("linear_velocity_w", 3),
    ("angular_velocity_b", 3),
    ("motor_omega", 4),
)
```

- [ ] **Step 5: Re-run the tests and keep them green**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_diffaero_lab_common_contracts.py -q`
Expected: PASS

### Task 3: Build the reusable UAV extension for PhysX quadrotor only

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/uav/dynamics/base.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/dynamics/registry.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/dynamics/allocation.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/dynamics/motor.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/dynamics/quadrotor.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/dynamics/controller.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/adapters/isaaclab.py`
- Create: `source/diffaero_lab/diffaero_lab/uav/adapters/newton.py`

- [ ] **Step 1: Copy donor files with minimal renaming into the new package**

Use direct copies as the first draft from:

- `refer/isaac_drone_racer/dynamics/allocation.py`
- `refer/isaac_drone_racer/dynamics/motor.py`
- `refer/diffaero/dynamics/base_dynamics.py`
- `refer/diffaero/dynamics/quadrotor.py`
- `refer/diffaero/dynamics/controller.py`

- [ ] **Step 2: Normalize imports and public entry points**

```python
# source/diffaero_lab/diffaero_lab/uav/dynamics/registry.py
from diffaero_lab.uav.dynamics.quadrotor import QuadrotorModel

DYNAMICS_REGISTRY = {
    "quad": QuadrotorModel,
}


def build_dynamics(model_name: str, cfg, device):
    return DYNAMICS_REGISTRY[model_name](cfg=cfg, device=device)
```

- [ ] **Step 3: Add explicit placeholders for later backends without implementing them**

```python
# source/diffaero_lab/diffaero_lab/uav/adapters/newton.py
def build_newton_adapter(*args, **kwargs):
    raise NotImplementedError("Warp/Newton execution is intentionally deferred to a follow-up plan")
```

- [ ] **Step 4: Verify the new package imports cleanly**

Run: `refer/IsaacLab/isaaclab.sh -p -c "import diffaero_lab.uav; from diffaero_lab.uav.dynamics.registry import build_dynamics"`
Expected: command exits 0

### Task 4: Register the PhysX direct task and make the environment return IsaacLab-shaped observations

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/env/__init__.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/__init__.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/__init__.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/__init__.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env_cfg.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/drone_racing_env.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/base.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/dynamics_bridge/quad.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/state/*.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/mdp/*.py`
- Create: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/agents/skrl_ppo_cfg.yaml`
- Test: `tests/test_drone_racing_registration.py`

- [ ] **Step 1: Write failing registration and observation-shape tests**

```python
def test_drone_racing_env_is_registered():
    import gymnasium as gym
    import diffaero_lab.env.tasks  # noqa: F401

    assert "Isaac-Drone-Racing-Direct-v0" in gym.registry


def test_direct_env_returns_policy_and_optional_critic_keys():
    import gymnasium as gym

    env = gym.make("Isaac-Drone-Racing-Direct-v0")
    obs, extras = env.reset()
    assert "policy" in obs
    assert "task_terms" in extras
    env.close()
```

- [ ] **Step 2: Run the test to verify the task does not exist yet**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_registration.py -q`
Expected: FAIL with missing module or missing registry entry

- [ ] **Step 3: Register the task in the IsaacLab style with task-local agent config**

```python
# source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/__init__.py
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Drone-Racing-Direct-v0",
    entry_point=f"{__name__}.drone_racing_env:DroneRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racing_env_cfg:DroneRacingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

- [ ] **Step 4: Implement the smallest valid DirectRLEnv slice**

```python
def _get_observations(self) -> dict[str, torch.Tensor]:
    policy_obs = build_policy_obs(...)
    critic_obs = build_critic_obs(...)
    return {
        OBS_POLICY: policy_obs,
        OBS_CRITIC: critic_obs,
    }


def _get_rewards(self) -> torch.Tensor:
    reward, task_terms = compute_reward_and_terms(...)
    self.extras[EXTRA_TASK_TERMS] = task_terms
    self.extras[EXTRA_SIM_STATE] = build_sim_state(...)
    return reward
```

- [ ] **Step 5: Keep the bridge as a subordinate component of DirectRLEnv hooks**

```python
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self._bridge.process_action(actions)


def _apply_action(self) -> None:
    self._bridge.apply_to_sim()
```

- [ ] **Step 6: Re-run the registration tests**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_drone_racing_registration.py -q`
Expected: PASS

### Task 5: Add the first RL baseline path using IsaacLab-integrated skrl PPO

**Files:**
- Create: `scripts/reinforcement_learning/skrl/train.py`
- Create: `scripts/reinforcement_learning/skrl/play.py`
- Modify: `source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/agents/skrl_ppo_cfg.yaml`

- [ ] **Step 1: Copy the current `scripts/skrl` pair into the IsaacLab-standard directory**

Use `scripts/skrl/train.py` and `scripts/skrl/play.py` as the direct donor.

- [ ] **Step 2: Replace `diffaero_lab.tasks` imports with the new env extension**

```python
import diffaero_lab.env.tasks  # noqa: F401
```

- [ ] **Step 3: Point the default task and config chain at `Isaac-Drone-Racing-Direct-v0`**

```yaml
# source/diffaero_lab/diffaero_lab/env/tasks/direct/drone_racing/agents/skrl_ppo_cfg.yaml
seed: 42
trainer:
  timesteps: 32768
agent:
  rollouts: 32
  experiment:
    directory: drone_racing_direct
```

- [ ] **Step 4: Run a one-iteration RL smoke test**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Drone-Racing-Direct-v0 --headless --max_iterations 1 --num_envs 8`
Expected: process starts, creates a log directory, completes one short training run, and exits 0

### Task 6: Implement the APG minimal differentiable adapter and trainer

**Files:**
- Create: `source/diffaero_lab/diffaero_lab/algo/algorithms/apg.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/trainers/apg_trainer.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/wrappers/env_adapter.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/wrappers/apg.py`
- Create: `source/diffaero_lab/diffaero_lab/algo/configs/apg/drone_racing.yaml`
- Create: `scripts/differential/train_apg.py`
- Test: `tests/test_apg_env_adapter.py`

- [ ] **Step 1: Write the failing APG adapter test first**

```python
def test_apg_adapter_reads_policy_critic_and_sim_state():
    from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter

    adapter = DifferentialEnvAdapter.make("Isaac-Drone-Racing-Direct-v0")
    batch = adapter.reset()
    assert "policy" in batch.observations
    assert "critic" in batch.observations
    assert "sim_state" in batch.extras
    adapter.close()
```

- [ ] **Step 2: Verify the APG stack is still missing**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_env_adapter.py -q`
Expected: FAIL with import errors

- [ ] **Step 3: Copy the APG training loop as a donor and reduce it to the smallest working form**

Use `refer/diffaero/algo/APG.py` as the donor for rollout and backward logic, but rewrite imports so the trainer consumes:

- `observations[OBS_POLICY]`
- `observations[OBS_CRITIC]`
- `extras[EXTRA_TASK_TERMS]`
- `extras[EXTRA_SIM_STATE]`

- [ ] **Step 4: Implement the environment adapter with strict contract checks**

```python
class DifferentialEnvAdapter:
    @classmethod
    def make(cls, task_id: str, cfg=None):
        env = gym.make(task_id, cfg=cfg)
        return cls(env)

    def reset(self):
        observations, extras = self.env.reset()
        self._validate(observations, extras)
        return Batch(observations=observations, extras=extras)
```

- [ ] **Step 5: Add a short APG runner script**

```python
# scripts/differential/train_apg.py
env = DifferentialEnvAdapter.make(args.task, cfg=env_cfg)
trainer = APGTrainer(env=env, cfg=algo_cfg)
trainer.train(max_iterations=args.max_iterations)
```

- [ ] **Step 6: Run the adapter tests and a one-iteration APG smoke command**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_apg_env_adapter.py -q`
Expected: PASS

Run: `refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1`
Expected: rollout executes, backward completes, command exits 0

### Task 7: Phase-1 verification and cleanup

**Files:**
- Modify: all files touched in Tasks 1-6 only if verification uncovers defects introduced by those tasks

- [ ] **Step 1: Run focused Python diagnostics on the new source trees**

Run: `python -m py_compile $(git ls-files 'source/diffaero_lab/**/*.py' 'scripts/reinforcement_learning/**/*.py' 'scripts/differential/**/*.py' 'tests/*.py')`
Expected: exit 0

- [ ] **Step 2: Run the new focused test suite**

Run: `refer/IsaacLab/isaaclab.sh -p -m pytest tests/test_diffaero_lab_common_contracts.py tests/test_drone_racing_registration.py tests/test_apg_env_adapter.py -q`
Expected: PASS

- [ ] **Step 3: Run the environment listing command**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/list_envs.py --keyword Drone-Racing`
Expected: table includes `Isaac-Drone-Racing-Direct-v0`

- [ ] **Step 4: Run one RL smoke command and one APG smoke command back-to-back**

Run: `refer/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Drone-Racing-Direct-v0 --headless --max_iterations 1 --num_envs 8 && refer/IsaacLab/isaaclab.sh -p scripts/differential/train_apg.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 1`
Expected: both commands exit 0

- [ ] **Step 5: Run repository formatting and lint checks**

Run: `refer/IsaacLab/isaaclab.sh -f`
Expected: exit 0

## Completion criteria for this plan

The plan is complete when all items below are true:

1. Four new extensions install cleanly beside the legacy `diffaero_lab` template.
2. `Isaac-Drone-Racing-Direct-v0` is registered and discoverable through Gym and `scripts/list_envs.py`.
3. The direct environment returns `observations["policy"]` and optional `observations["critic"]` in IsaacLab style.
4. The RL baseline path runs through an `isaaclab_rl` integrated wrapper.
5. The APG path consumes `task_terms` and `sim_state` through the custom differential adapter.
6. All focused tests and smoke commands in Task 7 pass.

## Follow-up plans after this phase

Write separate implementation plans for:

1. `APG_stochastic`
2. `SHAC`
3. `SHA2C`
4. `pmd` / `pmc` / `simple` backends
5. Warp / Newton execution path and `Isaac-Drone-Racing-Direct-Warp-v0`
