# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from diffaero_lab.uav.assets import CRAZYFLIE_CFG

from .track_generator import generate_track


@configclass
class DroneRacingWarpSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 20.0
    replicate_physics: bool = True

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

    track: RigidObjectCollectionCfg = generate_track()


@configclass
class DroneRacingWarpEnvCfg(DirectRLEnvCfg):
    scene: DroneRacingWarpSceneCfg = DroneRacingWarpSceneCfg()

    decimation: int = 4
    episode_length_s: float = 40.0

    action_space: int = 4
    observation_space: int = 13
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

    # Maps normalized thrust action to [0, thrust_scale * robot_weight].
    # action[3] ~= 0 starts near hover, matching Isaac Lab's quadcopter DirectRLEnv.
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
    gate_l1_radius: float = 1.5
    oob_xy_limit: float = 5.0
    oob_z_max: float = 7.0

    track_visuals_enabled: bool = True
    track_num_gates: int = 20
    track_gate_spacing: float = 2.0
    track_gate_z_start: float = 2.0
    track_gate_half_width: float = 1.0
    track_gate_bar_thickness: float = 0.05
    track_ground_size: tuple[float, float, float] = (4.0, 4.0, 0.02)
