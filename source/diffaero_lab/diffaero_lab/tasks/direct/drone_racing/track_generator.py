import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

from diffaero_lab.uav.assets import DEFAULT_DRONE_RACING_TRACK, GATE_ASSET, GatePose


def generate_track(track_config: dict[str, GatePose] | None = None) -> RigidObjectCollectionCfg:
    track_config = DEFAULT_DRONE_RACING_TRACK if track_config is None else track_config
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"gate_{gate_id}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Gate_{gate_id}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=GATE_ASSET.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=gate_config.pos,
                    rot=math_utils.quat_from_euler_xyz(
                        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(gate_config.yaw)
                    ).tolist(),
                ),
            )
            for gate_id, gate_config in track_config.items()
        }
    )
