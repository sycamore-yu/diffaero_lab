# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from diffaero_lab.env.tasks.direct.drone_racing import agents


gym.register(
    id="Isaac-Drone-Racing-Direct-v0",
    entry_point=f"{__name__}.drone_racing_env:DroneRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racing_env_cfg:DroneRacingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racing-Direct-Warp-v0",
    entry_point=f"{__name__}.drone_racing_env:DroneRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racing_env_warp_cfg:DroneRacingWarpEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
