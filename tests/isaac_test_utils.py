# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager

from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter


def make_cpu_test_adapter(task_id: str = "Isaac-Drone-Racing-Direct-v0", *, num_envs: int = 8) -> DifferentialEnvAdapter:
    from diffaero_lab.tasks.direct.drone_racing.drone_racing_env_cfg import DroneRacingEnvCfg

    warp_cache_dir = Path("/tmp/warp-cache")
    warp_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(warp_cache_dir))
    sim_utils.create_new_stage()
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)
    cfg = DroneRacingEnvCfg()
    cfg.scene = cfg.scene.replace(num_envs=num_envs, replicate_physics=False)
    cfg.sim.device = "cpu"
    return DifferentialEnvAdapter.make(task_id, cfg=cfg)
