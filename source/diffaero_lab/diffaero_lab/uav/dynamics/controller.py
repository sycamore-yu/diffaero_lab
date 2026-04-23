# Copyright (c) 2025, DiffAero Authors
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import torch
import torch.nn.functional as F


class RateController:
    def __init__(
        self,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        gravity: torch.Tensor,
        cfg: "DictConfig",
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.mass = mass
        self.inertia = inertia
        self.gravity = gravity
        self.thrust_ratio: float = cfg.thrust_ratio
        self.torque_ratio: float = cfg.torque_ratio

        self.K_angvel = torch.tensor(cfg.K_angvel, device=device)

        self.min_action = torch.tensor(
            [
                cfg.min_normed_thrust,
                cfg.min_roll_rate,
                cfg.min_pitch_rate,
                cfg.min_yaw_rate,
            ],
            device=device,
        )

        self.max_action = torch.tensor(
            [
                cfg.max_normed_thrust,
                cfg.max_roll_rate,
                cfg.max_pitch_rate,
                cfg.max_yaw_rate,
            ],
            device=device,
        )

    def __call__(self, q_xyzw, w, action) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_dims = action.shape[:-1]
        action_flat = action.reshape(-1, 4)
        q_flat = q_xyzw.reshape(-1, 4)
        w_flat = w.reshape(-1, 3)

        R_b2i = self._quaternion_to_matrix(q_flat)
        R_b2i.clamp_(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        R_i2b = R_b2i.transpose(-1, -2)

        desired_angvel_b = action_flat[:, 1:]
        actual_angvel_b = torch.bmm(R_i2b, w_flat.unsqueeze(-1)).squeeze(-1)
        angvel_err = desired_angvel_b - actual_angvel_b

        cross = torch.cross(
            actual_angvel_b,
            torch.bmm(
                self.inertia.unsqueeze(1)
                .expand(-1, actual_angvel_b.shape[0] // self.inertia.shape[0], -1)
                .reshape(-1, 3, 3),
                actual_angvel_b.unsqueeze(-1),
            ).squeeze(-1),
            dim=-1,
        )
        cross.div_(
            torch.max(
                cross.norm(dim=-1, keepdim=True) / 100,
                torch.tensor(1.0, device=cross.device),
            ).detach()
        )
        angacc = self.torque_ratio * self.K_angvel * angvel_err
        torque = (
            torch.bmm(
                self.inertia.unsqueeze(1)
                .expand(-1, actual_angvel_b.shape[0] // self.inertia.shape[0], -1)
                .reshape(-1, 3, 3),
                angacc.unsqueeze(-1),
            ).squeeze(-1)
            + cross
        )
        thrust = action_flat[:, 0] * self.thrust_ratio * self.gravity * self.mass

        torque = torque.reshape(*batch_dims, 3)
        thrust = thrust.reshape(*batch_dims)
        return thrust, torque

    def _quaternion_to_matrix(self, q):
        w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
        norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
        R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
        R[..., 0, 1] = 2 * (x * y - w * z)
        R[..., 0, 2] = 2 * (x * z + w * y)
        R[..., 1, 0] = 2 * (x * y + w * z)
        R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
        R[..., 1, 2] = 2 * (y * z - w * x)
        R[..., 2, 0] = 2 * (x * z - w * y)
        R[..., 2, 1] = 2 * (y * z + w * x)
        R[..., 2, 2] = 1 - 2 * (x**2 + y**2)
        return R
