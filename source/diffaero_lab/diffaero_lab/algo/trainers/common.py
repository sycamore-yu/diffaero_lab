# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared trainer helpers for rollout-route and task-loss selection."""

from __future__ import annotations

import torch

from diffaero_lab.common.keys import EXTRA_TASK_TERMS
from diffaero_lab.common.rollout_route import (
    DIRECT_DIFFERENTIAL_ROUTE,
    SCORE_FUNCTION_ROUTE,
    select_rollout_route,
)


def rollout_route(extras: dict) -> str:
    return select_rollout_route(extras)


def direct_differential_rollout(extras: dict) -> bool:
    return rollout_route(extras) == DIRECT_DIFFERENTIAL_ROUTE


def score_function_route(extras: dict) -> bool:
    return rollout_route(extras) == SCORE_FUNCTION_ROUTE


def trainer_loss(rewards: torch.Tensor, extras: dict, *, prefer_direct_loss: bool = True) -> torch.Tensor:
    task_terms = extras.get(EXTRA_TASK_TERMS, {})
    if prefer_direct_loss and direct_differential_rollout(extras) and "loss" in task_terms:
        return task_terms["loss"]
    if "progress" in task_terms:
        return -task_terms["progress"].mean()
    return -rewards.mean()
