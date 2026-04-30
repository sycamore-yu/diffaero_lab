#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

import diffaero_lab.tasks  # noqa: F401
from diffaero_lab.algo.algorithms.sha2c import SHA2CConfig
from diffaero_lab.algo.trainers.sha2c_trainer import SHA2CTrainer

from common import add_common_training_args, build_env_adapter, build_env_cfg
from isaaclab_tasks.utils import launch_simulation


def main():
    parser = argparse.ArgumentParser(description="Train SHA2C on drone racing task")
    add_common_training_args(parser, "Train SHA2C on drone racing task")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--target_entropy", type=float, default=-4.0, help="Target entropy for auto-tuning")
    parser.add_argument("--soft_update_coef", type=float, default=0.005, help="Target critic soft update coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    args_cli, hydra_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + hydra_args

    print(f"[INFO] Starting SHA2C training on task: {args_cli.task}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")

    env_cfg = build_env_cfg(args_cli)

    with launch_simulation(env_cfg, args_cli):
        adapter = build_env_adapter(args_cli, env_cfg)

        cfg = SHA2CConfig(
            actor_lr=args_cli.actor_lr,
            critic_lr=args_cli.critic_lr,
            max_grad_norm=args_cli.max_grad_norm,
            rollout_horizon=args_cli.rollout_horizon,
            target_entropy=args_cli.target_entropy,
            soft_update_coef=args_cli.soft_update_coef,
        )

        trainer = SHA2CTrainer(env=adapter, cfg=cfg)

        print("[INFO] Starting training loop...")
        trainer.train(max_iterations=args_cli.max_iterations)

        print("[INFO] Training complete")
        adapter.close()


if __name__ == "__main__":
    main()
