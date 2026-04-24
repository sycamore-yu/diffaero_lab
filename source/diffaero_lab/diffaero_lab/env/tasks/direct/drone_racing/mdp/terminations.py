# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor


def compute_dones(
    episode_length_buf: Tensor,
    max_episode_length: int,
    reset_terminated: Tensor,
) -> tuple[Tensor, Tensor]:
    time_out = episode_length_buf >= max_episode_length - 1
    return reset_terminated, time_out
