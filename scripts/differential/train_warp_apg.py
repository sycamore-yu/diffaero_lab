#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Brax-style Warp-native differentiable APG training for drone racing.

Architecture:
    IsaacLab shell: task registration, USD assets, scene, visualization, evaluation, reset
    Newton/Warp core: differentiable model, rollout, actor MLP, loss, wp.Tape() backward

Training path (Newton/Warp owns all differentiability):
    wp.Tape() {
        for t in 0..horizon:
            mlp_forward(obs, params) → action
            action_to_wrench(action) → body_f
            SolverSemiImplicit.step(state, body_f) → next_state
            compute_obs(next_state) → next_obs
            compute_loss(state, action) → step_loss
    }
    tape.backward(total_loss)
    warp.optim.SGD.step(params)

IsaacLab path (evaluation only):
    export_policy_torch() → DirectRLEnv.step() → high-fidelity rollout + viz

Usage:
    python scripts/differential/train_warp_apg.py \
        --task Isaac-Drone-Racing-Direct-Warp-v0 \
        --num_envs 256 \
        --rollout_horizon 32 \
        --max_iterations 1000 \
        --viz viser
"""

from __future__ import annotations

import argparse
import sys
import time

import diffaero_lab.tasks  # noqa: F401 — register gym tasks
import torch
from common import add_common_training_args, build_env_adapter, build_env_cfg
from diffaero_lab.algo.differential.warp_apg import WarpAPG
from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter
from diffaero_lab.uav.differential.rollout import RolloutConfig, WarpDroneRollout

from isaaclab_tasks.utils import launch_simulation


class WarpAPGTrainer:
    """Brax-style trainer: IsaacLab shell + Newton/Warp differentiable core.

    IsaacLab responsibilities:
        - Gym task registration & scene construction
        - Environment reset (provides initial state)
        - Visualization (viser)
        - Evaluation loop

    Warp core responsibilities:
        - Newton differentiable model (requires_grad=True)
        - Warp MLP actor (flat params array, requires_grad=True)
        - Full rollout under wp.Tape()
        - warp.optim.SGD updates
    """

    def __init__(self, isaac_env: DifferentialEnvAdapter, cfg: RolloutConfig):
        self.isaac_env = isaac_env
        self.cfg = cfg

        # Build Warp-native differentiable core
        self.rollout = WarpDroneRollout(cfg)
        self.apg = WarpAPG(self.rollout, lr=3e-4)

    def train(self, max_iterations: int = 1000, eval_every: int = 50, viz: bool = False) -> None:  # noqa: FBT001
        """Run the Brax-style training loop.

        Each iteration:
            1. Reset IsaacLab env → extract initial state → set into rollout
            2. WarpAPG.train_step() → forward+backward+optimizer under wp.Tape()
            3. Export policy → evaluate in IsaacLab (every eval_every iters)
        """
        loss_history: list[float] = []
        grad_history: list[float] = []

        for iteration in range(max_iterations):
            t0 = time.perf_counter()

            # 1. Reset IsaacLab env to get initial state & gate targets
            batch = self.isaac_env.reset()
            sim_state = batch.extras.get("sim_state", {})

            # Get drone env for gate targets
            drone_env = self.isaac_env.env.unwrapped

            # Set initial state into Warp rollout (via torch tensors from IsaacLab)
            init_pos = sim_state.get("position_w")
            init_quat = sim_state.get("quaternion_w")
            init_lin_vel = sim_state.get("linear_velocity_w")
            init_ang_vel = sim_state.get("angular_velocity_b")

            if init_pos is not None:
                self.rollout.set_initial_state(
                    {
                        "position_w": init_pos,
                        "quaternion_w": init_quat,
                        "linear_velocity_w": init_lin_vel,
                        "angular_velocity_b": init_ang_vel,
                        "target_position_w": sim_state.get("target_position_w"),
                    }
                )

            # Extract gate targets from drone env
            target_pos, target_yaw, next_pos, next_yaw = drone_env._gate_targets()
            self.rollout.set_gate_targets(target_pos, target_yaw, next_pos, next_yaw)

            # 2. Warp-native training step
            result = self.apg.train_step()
            loss_history.append(result["loss"])
            grad_history.append(result["grad_norm"])

            elapsed = time.perf_counter() - t0

            # 3. Logging
            if iteration % 10 == 0:
                print(
                    f"Iter {iteration}: loss={result['loss']:.4f}, "
                    f"grad_norm={result['grad_norm']:.6f}, "
                    f"it/s={1.0 / elapsed:.1f}"
                )

            # 4. Periodic evaluation in IsaacLab
            if eval_every > 0 and iteration > 0 and iteration % eval_every == 0:
                self._evaluate(viz=viz)

    def _evaluate(self, viz: bool = False) -> None:  # noqa: FBT001
        """Evaluate current policy in IsaacLab env.

        Exports Warp MLP weights → PyTorch tensors → runs IsaacLab rollout.
        """
        weights, biases = self.rollout.export_actor_pytorch()

        # Build a PyTorch MLP matching Warp architecture
        layer_dims = [self.cfg.obs_dim] + list(self.cfg.hidden_dims) + [self.cfg.action_dim]
        layers = []
        for li in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[li], layer_dims[li + 1]))
            layers[-1].weight.data = weights[li]
            layers[-1].bias.data = biases[li]
            if li < len(layer_dims) - 2:
                layers.append(torch.nn.ReLU())
            else:
                layers.append(torch.nn.Tanh())
        torch_actor = torch.nn.Sequential(*layers).to("cuda:0").eval()

        # Run one IsaacLab rollout for evaluation
        batch = self.isaac_env.reset()
        total_reward = 0.0
        steps = 0

        with torch.no_grad():
            for _ in range(self.cfg.horizon):
                obs = batch.observations["policy"]
                action = torch_actor(obs)
                batch.observations, rewards, terminated, truncated, batch.extras = self.isaac_env.step(action)
                total_reward += rewards.mean().item()
                steps += 1

        avg_reward = total_reward / max(steps, 1)
        print(f"  [Eval] avg_reward={avg_reward:.4f}, steps={steps}")


def main():
    parser = argparse.ArgumentParser(description="Brax-style Warp-native APG training")
    add_common_training_args(parser, "Train APG on drone racing with Warp-native differentiable core")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate in IsaacLab every N iterations")
    parser.add_argument("--sim_dt", type=float, default=1.0 / 400.0, help="Simulation timestep")
    parser.add_argument("--sim_substeps", type=int, default=4, help="Physics substeps per control step")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64], help="MLP hidden dimensions")
    parser.add_argument("--thrust_scale", type=float, default=1.9, help="Thrust scale factor")
    parser.add_argument("--moment_scale", type=float, default=0.01, help="Moment scale factor")
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    print(f"[INFO] Starting Brax-style Warp-native APG training on task: {args_cli.task}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")
    print("[INFO] Architecture: IsaacLab shell + Newton/Warp differentiable core")
    print(f"[INFO] Horizon: {args_cli.rollout_horizon}, Envs: {args_cli.num_envs}")

    env_cfg = build_env_cfg(args_cli)

    with launch_simulation(env_cfg, args_cli):
        adapter = build_env_adapter(args_cli, env_cfg)

        cfg = RolloutConfig(
            num_envs=args_cli.num_envs,
            horizon=args_cli.rollout_horizon,
            sim_dt=args_cli.sim_dt,
            sim_substeps=args_cli.sim_substeps,
            hidden_dims=tuple(args_cli.hidden_dims),
            thrust_scale=args_cli.thrust_scale,
            moment_scale=args_cli.moment_scale,
        )

        trainer = WarpAPGTrainer(isaac_env=adapter, cfg=cfg)

        print("[INFO] Starting training loop...")
        trainer.train(
            max_iterations=args_cli.max_iterations,
            eval_every=args_cli.eval_every,
            viz=args_cli.viz is not None,
        )

        print("[INFO] Training complete")
        adapter.close()


if __name__ == "__main__":
    main()
