# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


@dataclass(frozen=True, slots=True)
class AssetMetadata:
    asset_id: str
    display_name: str
    asset_type: str
    usd_path: str
    source: str
    frame_convention: str = "xyzw"
    supported_physics_routes: tuple[str, ...] = ("physx",)


@dataclass(frozen=True, slots=True)
class GatePose:
    pos: tuple[float, float, float]
    yaw: float


ASSETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASSETS_DIR.parents[4]


def _build_quad_cfg(usd_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


GATE_ASSET = AssetMetadata(
    asset_id="gate",
    display_name="Racing Gate",
    asset_type="gate",
    usd_path=str(ASSETS_DIR / "gate" / "gate.usd"),
    source="Migrated from refer/isaac_drone_racer/assets/gate/gate.usd",
    supported_physics_routes=("physx", "newton"),
)

CRAZYFLIE_ASSET = AssetMetadata(
    asset_id="crazyflie",
    display_name="Crazyflie",
    asset_type="drone",
    usd_path=str(REPO_ROOT / "refer" / "OmniDrones" / "omni_drones" / "robots" / "assets" / "usd" / "cf2x_isaac.usd"),
    source="Local Crazyflie asset mirrored from refer/OmniDrones/omni_drones/robots/assets/usd/cf2x_isaac.usd",
    supported_physics_routes=("physx", "newton"),
)
CRAZYFLIE_CFG = _build_quad_cfg(CRAZYFLIE_ASSET.usd_path)

FIVE_IN_DRONE_ASSET = AssetMetadata(
    asset_id="five_in_drone",
    display_name="Five-Inch Drone",
    asset_type="drone",
    usd_path=str(REPO_ROOT / "refer" / "isaac_drone_racer" / "assets" / "5_in_drone" / "5_in_drone.usd"),
    source="Migrated from refer/isaac_drone_racer/assets/5_in_drone/5_in_drone.usd",
    supported_physics_routes=("physx", "newton"),
)
FIVE_IN_DRONE_CFG = _build_quad_cfg(FIVE_IN_DRONE_ASSET.usd_path)

DEFAULT_DRONE_RACING_TRACK: dict[str, GatePose] = {
    "1": GatePose(pos=(0.0, 0.0, 1.0), yaw=0.0),
    "2": GatePose(pos=(10.0, 5.0, 0.0), yaw=0.0),
    "3": GatePose(pos=(10.0, -5.0, 0.0), yaw=(5.0 / 4.0) * pi),
    "4": GatePose(pos=(-5.0, -5.0, 2.5), yaw=pi),
    "5": GatePose(pos=(-5.0, -5.0, 0.0), yaw=0.0),
    "6": GatePose(pos=(5.0, 0.0, 0.0), yaw=(1.0 / 2.0) * pi),
    "7": GatePose(pos=(0.0, 5.0, 0.0), yaw=pi),
}
