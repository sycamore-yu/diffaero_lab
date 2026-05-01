# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Newton differentiable drone model for Warp-native training.

Independent of IsaacLab's simulation context. This model owns:
- Newton ModelBuilder → finalized Model with requires_grad=True
- SolverSemiImplicit for differentiable stepping
- CollisionPipeline for ground contact
"""

from __future__ import annotations

import newton
import warp as wp

# ---------------------------------------------------------------------------
# Propeller parameters (matching Newton diffsim drone example)
# ---------------------------------------------------------------------------


@wp.struct
class Propeller:
    """Propeller parameters computed from physical quantities.

    See Newton diffsim drone example for the full derivation.
    """

    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


def define_propeller(
    drone: int,
    pos: wp.vec3,
    fps: float,
    thrust: float = 0.109919,
    power: float = 0.040164,
    diameter: float = 0.2286,
    height: float = 0.01,
    max_rpm: float = 6396.667,
    turning_direction: float = 1.0,
) -> Propeller:
    """Create a Propeller struct with thrust/torque computed from physical params."""
    # Air density at sea level
    air_density = 1.225  # kg/m³

    rps = max_rpm / fps
    max_speed = rps * wp.TAU  # rad/s
    rps_square = rps**2

    prop = Propeller()
    prop.body = drone
    prop.pos = pos
    prop.dir = wp.vec3(0.0, 0.0, 1.0)
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / wp.TAU
    prop.turning_direction = turning_direction
    prop.max_speed_square = max_speed**2

    return prop


# ---------------------------------------------------------------------------
# WarpDroneModel
# ---------------------------------------------------------------------------


class WarpDroneModel:
    """Standalone differentiable Newton drone model.

    Creates a Newton model with one body per environment,
    two crossbar box shapes per drone, plus ground plane.
    Stores propeller params for external wrench computation.

    Usage:
        model = WarpDroneModel(num_envs=256, dt=1/400, requires_grad=True)
        state = model.state()
        control = model.control()
        contacts = model.contacts()
        # Apply forces, call solver.step(), read state
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        requires_grad: bool = True,
        device: str = "cuda:0",
        drone_size: float = 0.1,
    ):
        self._num_envs = num_envs
        self._dt = dt
        self._drone_size = drone_size

        # Build physics scene
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.05

        props = []
        crossbar_length = drone_size
        crossbar_height = drone_size * 0.05
        crossbar_width = drone_size * 0.05
        carbon_fiber_density = 1750.0  # kg/m³

        for i in range(num_envs):
            body = builder.add_body(label=f"drone_{i}")

            # Two crossbars forming the X-frame
            builder.add_shape_box(
                body,
                hx=crossbar_width,
                hy=crossbar_length,
                hz=crossbar_height,
                cfg=newton.ModelBuilder.ShapeConfig(density=carbon_fiber_density, collision_group=i),
            )
            builder.add_shape_box(
                body,
                hx=crossbar_length,
                hy=crossbar_width,
                hz=crossbar_height,
                cfg=newton.ModelBuilder.ShapeConfig(density=carbon_fiber_density, collision_group=i),
            )

            # Propellers for thrust/torque computation (applied externally via body_f)
            props.extend(
                (
                    define_propeller(
                        body,
                        wp.vec3(0.0, crossbar_length, 0.0),
                        1.0 / dt,
                        turning_direction=-1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, -crossbar_length, 0.0),
                        1.0 / dt,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(crossbar_length, 0.0, 0.0),
                        1.0 / dt,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(-crossbar_length, 0.0, 0.0),
                        1.0 / dt,
                        turning_direction=-1.0,
                    ),
                ),
            )

        # Ground plane with soft contact
        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        self._props = wp.array(props, dtype=Propeller)

        # Finalize model
        self._model = builder.finalize(requires_grad=requires_grad)

        # Configure soft contact on model
        self._model.soft_contact_ke = ke
        self._model.soft_contact_kf = kf
        self._model.soft_contact_kd = kd
        self._model.soft_contact_mu = mu
        self._model.soft_contact_restitution = 1.0

        # Solver — uses SolverSemiImplicit for differentiability
        self._solver = newton.solvers.SolverSemiImplicit(self._model)

        # Collision pipeline
        self._collision_pipeline = newton.CollisionPipeline(
            self._model,
            broad_phase="explicit",
            soft_contact_margin=10.0,
            requires_grad=requires_grad,
        )

        # Body count
        self._body_count = len(builder.body_q)

        # Robot weight (total mass * gravity)
        self._robot_weight = None

    def state(self) -> newton.State:
        """Return a fresh state for the model."""
        return self._model.state()

    def control(self) -> newton.Control:
        """Return a fresh control for the model."""
        return self._model.control()

    def contacts(self) -> newton.Contacts:
        """Return a fresh contacts container."""
        return self._collision_pipeline.contacts()

    @property
    def model(self) -> newton.Model:
        return self._model

    @property
    def solver(self):
        return self._solver

    @property
    def collision_pipeline(self):
        return self._collision_pipeline

    @property
    def props(self):
        return self._props

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def num_props(self) -> int:
        return len(self._props)

    @property
    def robot_weight(self) -> float:
        """Total weight per drone (mass × |g|). Computed once from model data."""
        if self._robot_weight is None:
            masses = wp.to_torch(self._model.body_mass)
            gravity = wp.to_torch(self._model.gravity)
            g_norm = float(gravity[0].norm())
            self._robot_weight = float(masses[0].sum() * g_norm)
        return self._robot_weight
