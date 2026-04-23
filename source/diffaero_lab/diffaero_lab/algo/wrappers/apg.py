# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""APG-specific gym wrapper for additional wrapping if needed.

This module provides wrapper classes specific to APG training if additional
processing is needed beyond what DifferentialEnvAdapter provides.
"""

from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter

__all__ = ["DifferentialEnvAdapter"]
