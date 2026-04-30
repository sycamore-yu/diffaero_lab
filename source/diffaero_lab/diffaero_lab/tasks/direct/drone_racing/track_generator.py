import os
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

def generate_track(track_config: dict | None) -> RigidObjectCollectionCfg:
    gate_usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../uav/assets/gate/gate.usd"))
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"gate_{gate_id}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Gate_{gate_id}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=gate_usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=gate_config["pos"],
                    rot=math_utils.quat_from_euler_xyz(
                        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(gate_config["yaw"])
                    ).tolist(),
                ),
            )
            for gate_id, gate_config in track_config.items()
        }
    )
