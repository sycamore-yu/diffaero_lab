# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared rollout-route selection for Differential Algorithm modules."""

from __future__ import annotations

from typing import Any

from diffaero_lab.common.capabilities import SUPPORTS_DIFFERENTIAL_ROLLOUT
from diffaero_lab.common.keys import EXTRA_CAPABILITIES

DIRECT_DIFFERENTIAL_ROUTE = "direct_differential_rollout"
SCORE_FUNCTION_ROUTE = "score_function_route"


def extract_capabilities(extras: dict[str, Any]) -> dict[str, Any]:
    capabilities = extras.get(EXTRA_CAPABILITIES, {})
    return capabilities if isinstance(capabilities, dict) else {}


def supports_direct_differential_rollout(extras: dict[str, Any]) -> bool:
    capabilities = extract_capabilities(extras)
    return bool(capabilities.get(SUPPORTS_DIFFERENTIAL_ROLLOUT, False))


def select_rollout_route(extras: dict[str, Any]) -> str:
    if supports_direct_differential_rollout(extras):
        return DIRECT_DIFFERENTIAL_ROUTE
    return SCORE_FUNCTION_ROUTE
