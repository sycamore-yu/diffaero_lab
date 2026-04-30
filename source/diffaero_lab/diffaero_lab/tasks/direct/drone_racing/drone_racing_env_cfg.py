# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import ArticulationCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
import torch

from isaaclab_assets import CRAZYFLIE_CFG

from .track_generator import generate_track

# Import visualizer configs
try:
    from isaaclab_visualizers.rerun import RerunVisualizerCfg
except ImportError:
    RerunVisualizerCfg = None

try:
    from isaaclab_visualizers.viser import ViserVisualizerCfg
except ImportError:
    ViserVisualizerCfg = None


@configclass
class DroneRacingSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 20.0
    replicate_physics: bool = True

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={
                "m1_joint": 200.0,
                "m2_joint": -200.0,
                "m3_joint": 200.0,
                "m4_joint": -200.0,
            },
        ),
    )

    track: RigidObjectCollectionCfg = generate_track(
        track_config={
            "1": {"pos": (0.0, 0.0, 1.0), "yaw": 0.0},
            "2": {"pos": (10.0, 5.0, 0.0), "yaw": 0.0},
            "3": {"pos": (10.0, -5.0, 0.0), "yaw": (5 / 4) * torch.pi},
            "4": {"pos": (-5.0, -5.0, 2.5), "yaw": torch.pi},
            "5": {"pos": (-5.0, -5.0, 0.0), "yaw": 0.0},
            "6": {"pos": (5.0, 0.0, 0.0), "yaw": (1 / 2) * torch.pi},
            "7": {"pos": (0.0, 5.0, 0.0), "yaw": torch.pi},
        }
    )


@configclass
class DroneRacingEnvCfg(DirectRLEnvCfg):
    scene: DroneRacingSceneCfg = DroneRacingSceneCfg()

    decimation: int = 4
    episode_length_s: float = 40.0

    action_space: int = 4
    observation_space: int = 13
    state_space: int = 13

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=decimation,
        physics=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**24,
        ),
        visualizer_cfgs=[
            ViserVisualizerCfg(
                port=8080,
                open_browser=False,
            ) if ViserVisualizerCfg is not None else None,
            RerunVisualizerCfg(
                web_port=9091,
                grpc_port=9877,
                open_browser=False,
            ) if RerunVisualizerCfg is not None else None,
        ],
    )

    # Dynamics model selection: "quad" (default, full quadrotor), "pmd" (point-mass discrete),
    # "pmc" (point-mass continuous), "simple" (simplified quadrotor)
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
