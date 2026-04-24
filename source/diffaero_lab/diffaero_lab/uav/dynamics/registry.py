# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from diffaero_lab.uav.dynamics.quadrotor import QuadrotorModel
from diffaero_lab.uav.dynamics.pointmass_discrete import DiscretePointMassModel
from diffaero_lab.uav.dynamics.pointmass_continuous import ContinuousPointMassModel
from diffaero_lab.uav.dynamics.simplified_quadrotor import SimplifiedQuadrotorModel

DYNAMICS_REGISTRY = {
    "quad": QuadrotorModel,
    "pmd": DiscretePointMassModel,
    "pmc": ContinuousPointMassModel,
    "simple": SimplifiedQuadrotorModel,
}


def build_dynamics(model_name: str, cfg, device):
    if model_name not in DYNAMICS_REGISTRY:
        raise ValueError(f"Unknown dynamics model '{model_name}'. Available models: {list(DYNAMICS_REGISTRY.keys())}")
    return DYNAMICS_REGISTRY[model_name](cfg=cfg, device=device)
