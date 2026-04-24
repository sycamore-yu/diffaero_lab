#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
APG Stochastic differential training script for drone racing.

Usage:
    python scripts/differential/train_apg_stochastic.py --task Isaac-Drone-Racing-Direct-v0 --max_iterations 100
"""

import argparse
import sys

import diffaero_lab.env.tasks  # noqa: F401
from diffaero_lab.algo.algorithms.apg_stochastic import APGStochasticConfig
from diffaero_lab.algo.trainers.apg_stochastic_trainer import APGStochasticTrainer
from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, parse_env_cfg


def main():
    parser = argparse.ArgumentParser(description="Train APG Stochastic on drone racing task")
    parser.add_argument("--task", type=str, default="Isaac-Drone-Racing-Direct-v0", help="Task ID")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum training iterations")
    parser.add_argument("--rollout_horizon", type=int, default=32, help="Rollout horizon per iteration")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
    add_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + hydra_args

    print(f"[INFO] Starting APG Stochastic training on task: {args_cli.task}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    with launch_simulation(env_cfg, args_cli):
        adapter = DifferentialEnvAdapter.make(args_cli.task, cfg=env_cfg)

        cfg = APGStochasticConfig(
            lr=args_cli.lr,
            max_grad_norm=args_cli.max_grad_norm,
            rollout_horizon=args_cli.rollout_horizon,
        )

        trainer = APGStochasticTrainer(env=adapter, cfg=cfg)

        print("[INFO] Starting training loop...")
        trainer.train(max_iterations=args_cli.max_iterations)

        print("[INFO] Training complete")
        adapter.close()


if __name__ == "__main__":
    main()
