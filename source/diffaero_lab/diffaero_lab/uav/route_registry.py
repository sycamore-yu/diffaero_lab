# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Route registry for Task Scene runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from diffaero_lab.common.sim_contract import build_capabilities
from diffaero_lab.tasks.direct.drone_racing.dynamics_bridge.pointmass_continuous import PMCDynamicsBridge
from diffaero_lab.tasks.direct.drone_racing.dynamics_bridge.pointmass_discrete import PMDDynamicsBridge
from diffaero_lab.tasks.direct.drone_racing.dynamics_bridge.quad import QuadDynamicsBridge
from diffaero_lab.tasks.direct.drone_racing.dynamics_bridge.simplified_quad import SimpleDynamicsBridge
from diffaero_lab.uav.adapters.newton import build_newton_adapter


@dataclass(frozen=True)
class RouteSpec:
    physics_route: str
    dynamics_model: str
    tensor_backend: str
    write_mode: str
    quat_convention: str
    supports_differential_rollout: bool
    adapter_factory: Callable[..., Any]

    def build_capabilities(self, *, supports_critic_state: bool) -> dict[str, bool]:
        return build_capabilities(
            supports_critic_state=supports_critic_state,
            supports_differential_rollout=self.supports_differential_rollout,
            supports_warp_backend=self.tensor_backend == "warp",
        )


class RouteRegistry:
    _ROUTES: dict[tuple[str, str], RouteSpec] = {
        ("physx", "quad"): RouteSpec(
            physics_route="physx",
            dynamics_model="quad",
            tensor_backend="physx",
            write_mode="indexed",
            quat_convention="xyzw",
            supports_differential_rollout=False,
            adapter_factory=QuadDynamicsBridge,
        ),
        ("physx", "pmd"): RouteSpec(
            physics_route="physx",
            dynamics_model="pmd",
            tensor_backend="physx",
            write_mode="indexed",
            quat_convention="xyzw",
            supports_differential_rollout=False,
            adapter_factory=PMDDynamicsBridge,
        ),
        ("physx", "pmc"): RouteSpec(
            physics_route="physx",
            dynamics_model="pmc",
            tensor_backend="physx",
            write_mode="indexed",
            quat_convention="xyzw",
            supports_differential_rollout=False,
            adapter_factory=PMCDynamicsBridge,
        ),
        ("physx", "simple"): RouteSpec(
            physics_route="physx",
            dynamics_model="simple",
            tensor_backend="physx",
            write_mode="indexed",
            quat_convention="xyzw",
            supports_differential_rollout=False,
            adapter_factory=SimpleDynamicsBridge,
        ),
        ("newton", "quad"): RouteSpec(
            physics_route="newton",
            dynamics_model="quad",
            tensor_backend="warp",
            write_mode="indexed",
            quat_convention="xyzw",
            supports_differential_rollout=False,
            adapter_factory=build_newton_adapter,
        ),
    }

    @classmethod
    def detect_physics_route(cls, cfg: Any) -> str:
        physics = getattr(getattr(cfg, "sim", None), "physics", None)
        type_name = type(physics).__name__
        module_name = type(physics).__module__
        if "Newton" in type_name or "MJWarp" in type_name or "newton" in module_name.lower():
            return "newton"
        return "physx"

    @classmethod
    def resolve(cls, cfg: Any) -> RouteSpec:
        physics_route = cls.detect_physics_route(cfg)
        dynamics_model = getattr(cfg, "dynamics_model", "quad")
        key = (physics_route, dynamics_model)
        if key not in cls._ROUTES:
            available = ", ".join(f"{route}:{model}" for route, model in sorted(cls._ROUTES))
            raise ValueError(f"Unsupported route '{physics_route}' with dynamics model '{dynamics_model}'. {available}")
        return cls._ROUTES[key]

    @classmethod
    def build_adapter(cls, *, cfg: Any, robot: Any, num_envs: int, device: str) -> tuple[Any, RouteSpec]:
        route_spec = cls.resolve(cfg)
        adapter = route_spec.adapter_factory(
            cfg=cfg,
            robot=robot,
            num_envs=num_envs,
            device=device,
        )
        return adapter, route_spec
