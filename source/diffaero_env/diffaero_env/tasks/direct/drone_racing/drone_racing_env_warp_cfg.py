# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class DroneRacingWarpSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 2.5
    replicate_physics: bool = True

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class DroneRacingWarpEnvCfg(DirectRLEnvCfg):
    scene: DroneRacingWarpSceneCfg = DroneRacingWarpSceneCfg()

    decimation: int = 4
    episode_length_s: float = 20.0

    action_space: int = 4
    observation_space: int = 17
    state_space: int = 13

    solver_cfg = MJWarpSolverCfg(
        njmax=1000,
        nconmax=2000,
        cone="pyramidal",
        integrator="implicitfast",
        impratio=1,
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
    )

    sim: SimulationCfg = SimulationCfg(dt=1 / 400, render_interval=decimation, physics=newton_cfg)

    dynamics_model: str = "quad"

    thrust_scale: float = 1.9
    moment_scale: float = 0.01

    rew_scale_progress: float = 20.0
    rew_scale_tracking: float = -1.0
    rew_scale_control_effort: float = -0.0001
    rew_scale_ang_vel: float = -0.0001
    rew_scale_terminal: float = -500.0
    rew_scale_gate: float = 400.0

    max_speed: float = 10.0
    max_ang_vel: float = 15.7
