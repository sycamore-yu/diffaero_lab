# DiffAero Lab

DiffAero Lab is a UAV simulation and learning workspace that migrates DiffAero research environments into the Isaac Lab ecosystem while preserving differentiable-control and reinforcement-learning workflows.

## Language

**UAV Platform**:
A reusable drone research platform made of airframe assets, dynamics models, controllers, motor models, adapters, and task-facing state contracts.
_Avoid_: drone code pile, platform glue

**Task Scene**:
A runnable learning problem with scene assets, reset rules, observations, rewards, terminal conditions, and a Gymnasium task ID.
_Avoid_: demo scene, environment asset

**Dynamics Model**:
The mathematical model that advances UAV state from actions or controls.
_Avoid_: backend, simulator

**Physics Route**:
The simulator execution path used by a task scene, such as PhysX or Newton/Warp.
_Avoid_: dynamics model

**Dynamics Bridge**:
The adapter that connects the Isaac Lab environment lifecycle to a selected dynamics model or physics route.
_Avoid_: second environment loop

**Differential Algorithm**:
A training algorithm that can use gradients from a differentiable objective and, when available, a differentiable simulator rollout.
_Avoid_: RL algorithm

**Direct Differential Rollout**:
A rollout whose loss has a gradient path from simulator state back to actor actions or parameters.
_Avoid_: Warp route, Newton route

**Score-Function Route**:
A policy-gradient route that optimizes stochastic policies through log probabilities and sampled rewards.
_Avoid_: fallback gradient

**Asset Metadata**:
Source, physical, frame, and compatibility information required to reuse an imported drone, gate, sensor, or scene asset safely.
_Avoid_: asset notes

## Relationships

- A **Task Scene** uses exactly one Isaac Lab environment lifecycle.
- A **Task Scene** selects one **Physics Route** through configuration.
- A **Task Scene** selects one **Dynamics Model** through configuration.
- A **Dynamics Bridge** exposes simulator state from a **Dynamics Model** or **Physics Route** to the shared contract.
- A **UAV Platform** provides the assets and physical models reused by multiple **Task Scenes**.
- A **Differential Algorithm** uses a **Direct Differential Rollout** when the task advertises that capability.
- A **Differential Algorithm** uses a **Score-Function Route** when the simulator route provides sampled transitions with score-function gradients.
- **Asset Metadata** belongs to assets imported into the **UAV Platform**.

## Example Dialogue

> **Dev:** "The racing **Task Scene** runs on the Newton/Warp **Physics Route**. Can APG use direct gradients?"
> **Domain expert:** "Only after the route advertises **Direct Differential Rollout**. Until then APG uses the **Score-Function Route**."

## Flagged Ambiguities

- "Newton 可微" can mean a Newton/Warp simulator is running, or that a direct gradient path reaches the actor. The canonical distinction is **Physics Route** for execution and **Direct Differential Rollout** for gradient capability.
- "后端" was used for both physics execution and UAV equations of motion. The canonical terms are **Physics Route** and **Dynamics Model**.
