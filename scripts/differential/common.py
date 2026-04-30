#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared launch helpers for differential training scripts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter

from isaaclab_tasks.utils import add_launcher_args, parse_env_cfg


def add_common_training_args(parser: ArgumentParser, description: str) -> None:
    parser.description = description
    parser.add_argument("--task", type=str, default="Isaac-Drone-Racing-Direct-v0", help="Task ID")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum training iterations")
    parser.add_argument("--rollout_horizon", type=int, default=32, help="Rollout horizon per iteration")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
    add_launcher_args(parser)


def build_env_cfg(args_cli: Namespace):
    return parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)


def build_env_adapter(args_cli: Namespace, env_cfg) -> DifferentialEnvAdapter:
    return DifferentialEnvAdapter.make(args_cli.task, cfg=env_cfg)
