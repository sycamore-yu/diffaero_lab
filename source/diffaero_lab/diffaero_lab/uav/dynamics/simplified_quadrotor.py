# Copyright (c) 2025, DiffAero Authors
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified quadrotor dynamics model (simple).

A stripped-down quadrotor model that uses the same 4D action space as the full
QuadrotorModel ([roll, pitch, yaw, thrust]) but with simplified physics.
"""

import torch
from torch import Tensor

from diffaero_lab.uav.dynamics.base import BaseDynamics


class SimplifiedQuadrotorModel(BaseDynamics):
    """Simplified quadrotor dynamics with 4D action space.

    State layout (13-dim): [p(3), q(4), v(3), w(3)]
    Action layout (4-dim): [roll, pitch, yaw, thrust] scaled by thrust_scale/moment_scale.
    """

    def __init__(self, cfg: "DictConfig", device: torch.device):
        super().__init__(cfg, device)
        self.n_envs = getattr(cfg, "n_envs", 1)
        self.n_agents = getattr(cfg, "n_agents", 1)
        self.dt = getattr(cfg, "dt", 0.01)

        self._state = torch.zeros(self.n_envs, self.n_agents, self.state_dim, device=device)
        self._acc = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        if self.n_agents == 1:
            self._state = self._state.squeeze(1)
            self._acc = self._acc.squeeze(1)

        self.thrust_scale = getattr(cfg, "thrust_scale", 1.0)
        self.moment_scale = getattr(cfg, "moment_scale", 0.01)

        self._G_vec = torch.tensor([0.0, 0.0, -getattr(cfg, "g", 9.81)], device=device, dtype=torch.float32)

    @property
    def model_name(self) -> str:
        return "simple"

    @property
    def state_dim(self) -> int:
        return 13

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def min_action(self) -> Tensor:
        return torch.tensor([-1.0, -1.0, -1.0, 0.0], device=self.device)

    @property
    def max_action(self) -> Tensor:
        return torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)

    @property
    def _p(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, :3]
        return self._state[:, :, :3]

    @property
    def _q(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, 3:7]
        return self._state[:, :, 3:7]

    @property
    def _v(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, 7:10]
        return self._state[:, :, 7:10]

    @property
    def _w(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, 10:13]
        return self._state[:, :, 10:13]

    def reset(self, env_ids: Tensor) -> None:
        mask = torch.zeros_like(self._acc, dtype=torch.bool)
        mask[env_ids] = True
        self._acc = torch.where(mask, 0.0, self._acc)

    def step(self, U: Tensor) -> None:
        """Step dynamics using simplified flat-plate model.

        U: [roll, pitch, yaw, thrust] in [-1, 1] range, scaled by thrust_scale/moment_scale
        """
        p = self._p
        q = self._q
        v = self._v
        w = self._w

        thrust = U[:, 3] * self.thrust_scale
        roll = U[:, 0] * self.moment_scale
        pitch = U[:, 1] * self.moment_scale
        yaw = U[:, 2] * self.moment_scale

        thrust_acc = torch.zeros_like(v)
        thrust_acc[:, 2] = thrust

        q_dot = 0.5 * self._quat_mul(q, torch.cat([w, torch.zeros((q.size(0), 1), device=self.device)], dim=-1))

        w_dot = torch.zeros_like(w)
        w_dot[:, 0] = roll
        w_dot[:, 1] = pitch
        w_dot[:, 2] = yaw

        v_dot = thrust_acc + self._G_vec
        self._acc = v_dot

        if self.n_agents == 1:
            self._state[:, :3] = p + self.dt * v
            self._state[:, 3:7] = q + self.dt * q_dot
            self._state[:, 7:10] = v + self.dt * v_dot
            self._state[:, 10:13] = w + self.dt * w_dot
        else:
            self._state[:, :, :3] = p + self.dt * v
            self._state[:, :, 3:7] = q + self.dt * q_dot
            self._state[:, :, 7:10] = v + self.dt * v_dot
            self._state[:, :, 10:13] = w + self.dt * w_dot

        q_norm = self._quat_norm(self._state[:, 3:7] if self.n_agents == 1 else self._state[:, :, 3:7])
        q_norm = q_norm.unsqueeze(-1)
        if self.n_agents == 1:
            self._state[:, 3:7] = self._state[:, 3:7] / q_norm
        else:
            self._state[:, :, 3:7] = self._state[:, :, 3:7] / q_norm

    def compute(self, *args, **kwargs) -> Tensor:
        return self._acc

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 3], q1[..., 0], q1[..., 1], q1[..., 2]
        w2, x2, y2, z2 = q2[..., 3], q2[..., 0], q2[..., 1], q2[..., 2]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _quat_norm(self, q):
        return torch.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2 + q[..., 3] ** 2)
