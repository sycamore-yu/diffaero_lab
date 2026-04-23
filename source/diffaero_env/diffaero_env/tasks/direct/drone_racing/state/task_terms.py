# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_common.terms import (
    TERM_ANGULAR_RATE,
    TERM_COLLISION,
    TERM_CONTROL_EFFORT,
    TERM_CONTROL_SMOOTHNESS,
    TERM_GATE_PASS,
    TERM_PROGRESS,
    TERM_TERMINAL,
    TERM_TIME_PENALTY,
    TERM_TRACKING_ERROR,
)


def build_task_terms(
    progress: Tensor,
    tracking_error: Tensor,
    gate_pass: Tensor,
    collision: Tensor,
    terminal: Tensor,
    control_effort: Tensor,
    control_smoothness: Tensor,
    angular_rate: Tensor,
    time_penalty: Tensor,
) -> dict[str, Tensor]:
    return {
        TERM_PROGRESS: progress,
        TERM_TRACKING_ERROR: tracking_error,
        TERM_GATE_PASS: gate_pass,
        TERM_COLLISION: collision,
        TERM_TERMINAL: terminal,
        TERM_CONTROL_EFFORT: control_effort,
        TERM_CONTROL_SMOOTHNESS: control_smoothness,
        TERM_ANGULAR_RATE: angular_rate,
        TERM_TIME_PENALTY: time_penalty,
    }
