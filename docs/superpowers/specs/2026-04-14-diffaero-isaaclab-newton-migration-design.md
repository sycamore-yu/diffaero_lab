# DiffAero Isaac Lab + Newton Migration Spec

Status: active spec
Last aligned: 2026-04-30

## 1. Purpose

This spec defines the migration of `refer/diffaero/` into the current Isaac Lab workspace at `source/diffaero_lab/`.

The target platform is a unified UAV research stack that supports:

1. Multiple UAV dynamics models: `quad`, `pmd`, `pmc`, and `simple`.
2. Multiple training families: standard RL algorithms and differential algorithms.
3. Multiple task scenes: racing first, then position control, obstacle avoidance, multi-agent position control, and map coverage.
4. Shared assets and training facilities: drone assets, gate/track assets, dynamics parameters, controllers, motor models, logging, wrappers, and runnable training configs.
5. A Newton/Warp route for differentiable physics experiments.

This document replaces the earlier four-extension design. The current project has already moved to a single installable Isaac Lab extension with internal domain packages, so future work must align to that structure.

## 2. External Documentation Confirmed

Isaac Lab and Newton design constraints were checked through Context7 on 2026-04-30:

- Isaac Lab custom direct tasks use a `DirectRLEnvCfg` configclass with `SimulationCfg`, `InteractiveSceneCfg`, robot/scene config, action/observation/state spaces, and task parameters.
- Isaac Lab custom tasks register Gymnasium IDs from the task package `__init__.py`, with agent config entry points under `agents/`.
- The Isaac Lab extension template installs an external project with `python -m pip install -e source/<extension_name>` and keeps `config/extension.toml` plus `setup.py` in the extension root.
- Newton differentiability is exposed through Warp tape execution: finalize the Newton model with `requires_grad=True`, run simulation and loss kernels inside `wp.Tape()`, then call `tape.backward(loss)`.
- Newton solver examples show `SolverSemiImplicit` as the straightforward differentiable path; MJWarp/Newton execution through Isaac Lab is a Warp-backed simulator route that still needs an explicit autograd bridge before PyTorch direct APG can receive simulator gradients.

Primary docs referenced by Context7:

- Isaac Lab custom environment and Gym registration docs: <https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_omniisaacgymenvs.html>
- Isaac Lab extension template: <https://github.com/isaac-sim/IsaacLabExtensionTemplate>
- Newton solver and differentiability docs: <https://newton-physics.github.io/newton/latest/api/newton_solvers.html>

## 3. Current Project Shape

The active implementation is one Isaac Lab extension package:

```text
source/diffaero_lab/
├── config/extension.toml
├── setup.py
├── pyproject.toml
└── diffaero_lab/
    ├── algo/
    ├── common/
    ├── tasks/
    └── uav/
```

The canonical installation unit is:

```bash
python -m pip install -e source/diffaero_lab
```

The internal package boundaries are:

| Package | Responsibility |
|---|---|
| `diffaero_lab.tasks` | Isaac Lab task registration, environment lifecycle, task scenes, MDP functions, observation/reward/done/reset semantics |
| `diffaero_lab.uav` | UAV platform assets, dynamics models, controllers, motor models, allocation logic, Isaac Lab/Newton adapters |
| `diffaero_lab.algo` | Differential algorithms, trainers, wrappers, and algorithm configs |
| `diffaero_lab.common` | Shared observation/extras keys, task terms, capabilities, flattening/state adapters |

The project keeps Isaac Lab extension packaging at the top level and uses internal packages for domain separation.

## 4. Migration Target

The migrated platform should preserve DiffAero's research breadth while adopting Isaac Lab's environment lifecycle and extension conventions.

### 4.1 Source-to-target mapping

| Source area | Target area |
|---|---|
| `refer/diffaero/env/racing.py` | `diffaero_lab.tasks.direct.drone_racing` |
| `refer/diffaero/env/position_control.py` | future `diffaero_lab.tasks.direct.position_control` |
| `refer/diffaero/env/obstacle_avoidance.py` | future `diffaero_lab.tasks.direct.obstacle_avoidance` |
| `refer/diffaero/env/position_control_multi_agent.py` | future `diffaero_lab.tasks.direct.position_control_multi_agent` |
| `refer/diffaero/dynamics/quadrotor.py` | `diffaero_lab.uav.dynamics.quadrotor` |
| `refer/diffaero/dynamics/pointmass.py` | `diffaero_lab.uav.dynamics.pointmass_discrete` and `pointmass_continuous` |
| `refer/diffaero/dynamics/controller.py` | `diffaero_lab.uav.dynamics.controller` |
| `refer/diffaero/algo/APG.py` | `diffaero_lab.algo.algorithms.apg` |
| `refer/diffaero/algo/SHAC.py` | `diffaero_lab.algo.algorithms.shac` |
| `refer/diffaero/algo/MASHAC.py` | future multi-agent trainer family |
| `refer/diffaero/cfg/**` | task and algorithm config files under `diffaero_lab.tasks/**/agents` and `diffaero_lab.algo/configs` |
| `refer/isaac_drone_racer/assets/**` | `diffaero_lab.uav.assets/**` |
| `refer/isaac_drone_racer/tasks/drone_racer/**` | `diffaero_lab.tasks.direct.drone_racing/**` |

### 4.2 First production slice

The first supported vertical slice is:

```text
drone_racing task
+ Crazyflie/5-inch drone asset path
+ gate/track assets
+ quad/pmd/pmc/simple dynamics selection
+ PhysX RL route
+ Newton/Warp experimental route
+ APG/APG stochastic/SHAC/SHA2C training contracts
```

## 5. Task Architecture

`drone_racing` is implemented as a direct Isaac Lab task:

```text
source/diffaero_lab/diffaero_lab/tasks/direct/drone_racing/
├── __init__.py
├── agents/
│   └── skrl_ppo_cfg.yaml
├── drone_racing_env.py
├── drone_racing_env_cfg.py
├── drone_racing_env_warp_cfg.py
├── dynamics_bridge/
├── mdp/
├── state/
└── track_generator.py
```

### 5.1 Registered task IDs

Current task IDs:

| Task ID | Physics route | Config |
|---|---|---|
| `Isaac-Drone-Racing-Direct-v0` | PhysX | `DroneRacingEnvCfg` |
| `Isaac-Drone-Racing-Direct-Warp-v0` | Isaac Lab Newton/MJWarp | `DroneRacingWarpEnvCfg` |

The task IDs share the same `DroneRacingEnv` class and diverge through config and backend adapter selection.

### 5.2 Environment lifecycle

`DroneRacingEnv` owns the Isaac Lab `DirectRLEnv` lifecycle:

1. `_setup_scene()` creates robot, ground, lights, track state, and the dynamics bridge.
2. `_pre_physics_step(actions)` stores actions and calls `bridge.process_action(actions)`.
3. `_apply_action()` calls `bridge.apply_to_sim()`.
4. `_get_observations()` reads bridge state and builds policy/critic observations.
5. `_get_rewards()` computes reward terms, updates gate progress, and writes `extras`.
6. `_get_dones()` evaluates gate failure, out-of-bounds, and timeout conditions.
7. `_reset_idx()` resets Isaac Lab state, task-local gate state, bridge state, and cached tensors.

The environment creates exactly one simulation context through Isaac Lab. Dynamics bridges are adapters inside that environment lifecycle.

### 5.3 Dynamics bridge contract

Every dynamics route exposed by `drone_racing` must implement:

```python
reset(env_ids)
process_action(actions)
apply_to_sim()
read_base_state()
read_motor_state()
read_dynamics_info()
detach()
```

Required base-state fields:

```text
position_w
quaternion_w
linear_velocity_w
angular_velocity_b
```

Required motor-state fields:

```text
motor_omega
```

Required dynamics metadata:

```text
model_name
state_layout_version
tensor_backend
write_mode
quat_convention
```

The bridge hides whether state came from PhysX tensors, Newton/Warp arrays, or a torch-only analytical dynamics model.

## 6. Shared Contract Between Environments and Algorithms

Environment outputs use Isaac Lab/Gymnasium conventions plus shared `extras`.

### 6.1 Observations

Standard observation keys:

| Key | Meaning |
|---|---|
| `policy` | Actor observation |
| `critic` | Critic/state observation when `state_space > 0` |

Current racing policy observation is 13-dimensional:

```text
target_position_relative_w: 3
linear_velocity_w: 3
angular_velocity_b: 3
next_target_position_relative_w: 3
next_target_yaw_relative: 1
```

### 6.2 Extras

Standard extras keys:

| Key | Meaning |
|---|---|
| `task_terms` | Reward/loss terms with stable names |
| `sim_state` | Backend-normalized simulator state and dynamics metadata |
| `capabilities` | Optional feature flags advertised by the environment |
| `dynamics` | Optional shorthand dynamics metadata |
| `state_before_reset` | Optional reset bookkeeping |
| `terminal_state` | Optional terminal-state snapshot |

### 6.3 Task terms

Canonical task terms:

```text
progress
tracking_error
gate_pass
collision
terminal
control_effort
control_smoothness
angular_rate
time_penalty
loss
reward
```

`loss` is the algorithm-facing differential objective. `reward` is the RL/logging objective.

### 6.4 Sim state

Required common `sim_state` fields:

```text
position_w
quaternion_w
linear_velocity_w
angular_velocity_b
motor_omega
step_count
last_action
progress
target_position_w
dynamics
```

The `dynamics` sub-dict owns layout metadata. Quaternion convention must be explicit because current code has both `xyzw` and `wxyz` assumptions in different helpers.

Spec requirement:

```text
Runtime sim_state from live bridges must report the convention used by the source tensor.
Common adapters must normalize only when explicitly requested by the caller.
```

## 7. UAV Platform Asset Spec

`diffaero_lab.uav` is the platform layer for reusable UAV assets and physical models.

### 7.1 Asset classes

Assets are grouped by semantic role:

| Asset class | Examples | Target location |
|---|---|---|
| Drone models | Crazyflie, 5-inch drone USD/URDF/mesh | `diffaero_lab/uav/assets/drones/` |
| Racing gates | `gate.usd`, textures, collision geometry | `diffaero_lab/uav/assets/gates/` |
| Track definitions | generated gate layouts and named courses | task-local `track_generator.py` plus reusable course configs |
| Sensor payloads | camera, lidar, IMU metadata | future `diffaero_lab/uav/assets/sensors/` |
| Scene assets | ground, obstacles, rooms, outdoor maps | future `diffaero_lab/uav/assets/scenes/` |

Current state:

- Gate assets from `refer/isaac_drone_racer/assets/gate` have been copied to `source/diffaero_lab/diffaero_lab/uav/assets/gate`.
- Racing now uses `CRAZYFLIE_CFG` from the local `diffaero_lab.uav.assets` catalog, backed by the mirrored OmniDrones Crazyflie USD.
- The 5-inch drone asset from `refer/isaac_drone_racer/assets/5_in_drone` should be migrated into `uav/assets/drones/five_in/` before it becomes a supported platform option.

### 7.2 Asset metadata

Each reusable drone asset must define:

```text
asset_id
display_name
usd_path
urdf_path optional
mesh_paths optional
body_name
rotor_joint_names
mass
inertia
rotor_layout
motor_model_id
controller_defaults
sensor_mounts
supported_physics_routes
license/source
```

Each reusable scene asset must define:

```text
asset_id
asset_type
usd_path or generator
collision_enabled
visual_enabled
scale
frame_convention
license/source
```

Assets imported from `refer/` must retain source attribution in metadata.

## 8. UAV Dynamics, Controller, and Motor Spec

### 8.1 Dynamics model registry

`diffaero_lab.uav.dynamics.registry` is the canonical lookup point for dynamics models.

Supported model names:

| Name | Meaning | Route |
|---|---|---|
| `quad` | Full quadrotor wrench/body dynamics | PhysX and Newton/Warp adapter route |
| `pmd` | Point-mass discrete dynamics | torch analytical dynamics bridge |
| `pmc` | Point-mass continuous dynamics | torch analytical dynamics bridge |
| `simple` | Simplified quadrotor dynamics | torch analytical dynamics bridge |

Spec requirements:

1. Model names are stable public config values.
2. Each model exports a common base-state layout.
3. Backend-specific state fields live under namespaced metadata or optional model fields.
4. Task code selects by `cfg.dynamics_model`.
5. Training code reads capability and backend metadata from `extras`, then chooses objective mode.

### 8.2 Controller and allocation

Controller code belongs in `diffaero_lab.uav.dynamics.controller`.

Allocation code belongs in `diffaero_lab.uav.dynamics.allocation`.

The controller stack must separate:

```text
policy action
-> action normalization
-> controller target or wrench command
-> allocation to motor thrusts
-> motor dynamics
-> simulator write
```

This separation allows the same policy/action contract to run through PhysX, analytical torch dynamics, and Newton/Warp experiments.

### 8.3 Motor dynamics

Motor dynamics belongs in `diffaero_lab.uav.dynamics.motor`.

Each motor model must specify:

```text
motor_model_id
input_unit
output_unit
time_constant
thrust_coefficient
torque_coefficient
min_omega
max_omega
spin_directions
```

The bridge should expose `motor_omega` in `sim_state` even when the backend approximates it.

## 9. Training Facility Spec

### 9.1 RL algorithms

Standard RL uses Isaac Lab-compatible task IDs and agent configs:

```text
tasks/direct/<task>/agents/*.yaml
```

Current racing config:

```text
source/diffaero_lab/diffaero_lab/tasks/direct/drone_racing/agents/skrl_ppo_cfg.yaml
```

RL algorithms treat simulator state as sampled environment transitions. Their gradient source is reward-weighted policy likelihood or value learning.

### 9.2 Differential algorithms

Differential algorithms live under:

```text
diffaero_lab.algo.algorithms
diffaero_lab.algo.trainers
diffaero_lab.algo.wrappers
diffaero_lab.algo.configs
```

Current algorithm modules:

```text
apg.py
apg_stochastic.py
shac.py
sha2c.py
actor_critic.py
```

Differential trainers must choose one objective route:

| Route | Gradient source | Backend requirement |
|---|---|---|
| Direct APG | `task_terms["loss"] -> simulator state -> action -> actor` | Differentiable simulator path connected to actor parameters |
| Score-function PG | `log_prob * reward/advantage` | Any sampled environment |
| Hybrid | differentiable reward/model terms plus sampled simulator transitions | Explicit trainer config |

The route must be selected by explicit capability metadata. `tensor_backend` describes simulator memory/backend shape.

## 10. Newton/Warp Differentiability Spec

### 10.1 Current finding

The current racing Newton/Warp route lacks a direct PyTorch gradient from actor action to simulator state.

The analysis in:

```text
/home/tong/.gemini/antigravity/brain/7de4b4ed-ed21-447b-8a45-5ac29d390502/gradient_analysis.md
```

identifies three breakpoints:

1. `NewtonBackendAdapter.process_action()` stores `actions.clone().detach()`.
2. `warp.to_torch()` returns torch views outside the PyTorch autograd graph.
3. `APGStochasticTrainer` currently maps `tensor_backend == "warp"` to direct APG loss.

The key mismatch:

```text
Direct APG expects:
actor -> torch action -> differentiable physics -> torch state -> torch loss -> backward

Current MJWarp route provides:
actor -> torch action -> detached adapter/write -> Warp simulation -> torch view outside autograd -> torch loss
```

### 10.2 Immediate route

Warp/Newton racing training should use score-function policy gradient until an explicit autograd bridge exists.

Trainer selection rule:

```text
if extras.capabilities.supports_differential_rollout is true:
    use direct differential objective
else:
    use score-function or RL objective
```

For the current `Isaac-Drone-Racing-Direct-Warp-v0` route:

```text
tensor_backend = "warp"
supports_differential_rollout = false
recommended_objective = "score_function_pg"
```

### 10.3 Correct direct-differentiable route

The direct Newton route requires a Newton/Warp-owned differentiable rollout module.

Required design:

1. Build the Newton model with `requires_grad=True`.
2. Use a solver with confirmed differentiability for the target rigid-body route; start with `SolverSemiImplicit` for minimal differentiable tests.
3. Allocate differentiable state/control arrays in Warp.
4. Run rollout and loss computation inside `wp.Tape()`.
5. Call `tape.backward(loss)`.
6. Expose gradients to the optimizer through one of two explicit integration modes:
   - Warp-owned optimizer updates Warp parameters directly.
   - Custom `torch.autograd.Function` maps torch actions/parameters to Warp arrays in forward and returns Warp tape gradients in backward.

The second mode is the required path for PyTorch APG/SHAC trainers to optimize PyTorch actors through Newton simulation.

### 10.4 Differentiability milestones

| Milestone | Acceptance criteria |
|---|---|
| D0: Current Warp smoke | `Isaac-Drone-Racing-Direct-Warp-v0` runs and trains through score-function gradients |
| D1: Newton particle gradient fixture | Minimal Newton example proves `wp.Tape()` gradients through state/control |
| D2: Single-step quad wrench gradient | One quad state step produces nonzero gradient w.r.t. control |
| D3: Torch autograd bridge | `torch.autograd.grad(loss, action)` is nonzero through a custom bridge |
| D4: Racing short-horizon APG | Direct APG on racing reports nonzero actor gradient through simulator state |
| D5: SHAC/SHA2C rollout | Critic-assisted direct rollout works with controlled truncation/detach policy |

### 10.5 Solver policy

Newton solver selection must be documented per experiment:

| Solver | Role in this project |
|---|---|
| `SolverSemiImplicit` | First differentiability fixture and simple rigid-body gradient path |
| `SolverMuJoCo` / MJWarp | High-performance Isaac Lab Newton route, currently experimental for direct PyTorch gradients |
| `SolverXPBD` | Candidate for non-smooth/contact-heavy scenes after gradient behavior is proven |
| `SolverFeatherstone` | Candidate for articulated dynamics experiments |

The racing spec treats Newton/MJWarp as an experimental simulator backend until the bridge milestones above are complete.

## 11. Scene and Task Asset Spec

### 11.1 Racing scene

Racing owns task-specific track semantics:

```text
gate order
gate crossing plane
gate opening/collision rule
target gate
next target gate
progress and terminal conditions
```

`track_generator.py` should generate Isaac Lab scene objects from reusable gate assets.

The canonical gate pass rule should match the current implementation:

1. A drone passes the gate when its segment crosses the gate plane from the previous position to the current position.
2. The crossing point must lie inside the gate opening under the configured L1 radius.
3. A plane crossing outside the opening is a gate collision.

### 11.2 Future scenes

Future tasks should follow the same layout:

```text
tasks/direct/<task_name>/
├── __init__.py
├── agents/
├── <task_name>_env.py
├── <task_name>_env_cfg.py
├── mdp/
├── state/
└── scene or generator files
```

Candidate migrations:

| DiffAero task | Target task | Notes |
|---|---|---|
| `pc` | `position_control` | minimal dynamics/algorithm benchmark |
| `oa` / `oa_small` | `obstacle_avoidance` | scene assets and collision semantics required |
| `mapc` | `map_coverage` | multi-agent and coverage metrics required |
| `racing` | `drone_racing` | active first slice |

## 12. Project Structure Target

Target structure:

```text
source/diffaero_lab/diffaero_lab/
├── algo/
│   ├── algorithms/
│   ├── configs/
│   ├── trainers/
│   └── wrappers/
├── common/
│   ├── adapters/
│   ├── capabilities.py
│   ├── keys.py
│   └── terms.py
├── tasks/
│   ├── direct/
│   │   ├── drone_racing/
│   │   ├── position_control/
│   │   ├── obstacle_avoidance/
│   │   └── position_control_multi_agent/
│   └── manager_based/
└── uav/
    ├── adapters/
    ├── assets/
    │   ├── drones/
    │   ├── gates/
    │   ├── scenes/
    │   └── sensors/
    └── dynamics/
```

The default project path now exposes `diffaero_lab.tasks`, `diffaero_lab.algo`, `diffaero_lab.common`, and `diffaero_lab.uav` as the supported runtime surface.

## 13. Implementation Rules

1. Keep the single extension package layout.
2. Add new task scenes under `diffaero_lab.tasks.direct`.
3. Put reusable UAV platform pieces under `diffaero_lab.uav`.
4. Put training algorithms under `diffaero_lab.algo`.
5. Put task-algorithm contracts under `diffaero_lab.common`.
6. Register every runnable task with a stable Gymnasium ID in the task package `__init__.py`.
7. Keep RL reward and differential loss as separate terms.
8. Treat `tensor_backend` as memory/backend metadata.
9. Treat `supports_differential_rollout` as the direct-gradient capability flag.
10. Keep quaternion convention explicit in every state contract.
11. Preserve source attribution for assets and reference code imported from `refer/`.
12. Add focused tests before expanding a backend, asset family, or task family.

## 14. Acceptance Criteria

The migration is aligned when the following are true:

1. `Isaac-Drone-Racing-Direct-v0` runs with PhysX and exports the standard observation/extras contract.
2. `Isaac-Drone-Racing-Direct-Warp-v0` runs with Newton/MJWarp and advertises its gradient capability accurately.
3. `quad`, `pmd`, `pmc`, and `simple` dynamics routes expose the common state contract.
4. APG stochastic produces gradients through score-function mode on non-direct-differentiable routes.
5. A Newton differentiability fixture proves `wp.Tape()` gradients before direct APG is enabled on racing.
6. Drone/gate/track assets live under `diffaero_lab.uav.assets` with metadata and source attribution.
7. Future task migrations reuse the same task layout and shared UAV platform layer.

## 15. Immediate Next Work

1. Add explicit capability export in `DroneRacingEnv.extras`, including `supports_differential_rollout`.
2. Change APG stochastic trainer routing to use capability metadata; keep `tensor_backend == "warp"` as backend shape metadata.
3. Add tests for the current Warp route proving score-function gradients are nonzero and direct APG is gated off.
4. Normalize or document quaternion convention at the bridge boundary.
5. Move the 5-inch drone asset into `diffaero_lab.uav.assets.drones` with metadata.
6. Add a minimal Newton `wp.Tape()` differentiability fixture independent of Isaac Lab racing.
