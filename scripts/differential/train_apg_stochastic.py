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

import diffaero_lab.tasks  # noqa: F401
from diffaero_lab.algo.algorithms.apg_stochastic import APGStochasticConfig
from diffaero_lab.algo.trainers.apg_stochastic_trainer import APGStochasticTrainer

from common import add_common_training_args, build_env_adapter, build_env_cfg
from isaaclab_tasks.utils import launch_simulation


def main():
    parser = argparse.ArgumentParser()
    add_common_training_args(parser, "Train APG Stochastic on drone racing task")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    args_cli, hydra_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + hydra_args

    print(f"[INFO] Starting APG Stochastic training on task: {args_cli.task}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")

    env_cfg = build_env_cfg(args_cli)

    with launch_simulation(env_cfg, args_cli):
        adapter = build_env_adapter(args_cli, env_cfg)

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
