# Copyright (c) 2025, DiffAero Authors
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_uav.dynamics.base import BaseDynamics
from diffaero_uav.dynamics.controller import RateController


class QuadrotorModel(BaseDynamics):
    def __init__(self, cfg: "DictConfig", device: torch.device):
        super().__init__(cfg, device)
        self.n_envs = cfg.n_envs
        self.n_agents = cfg.get("n_agents", 1)
        self.dt = cfg.dt

        self._state = torch.zeros(self.n_envs, self.n_agents, self.state_dim, device=device)
        self._acc = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        if self.n_agents == 1:
            self._state = self._state.squeeze(1)
            self._acc = self._acc.squeeze(1)

        self.n_substeps: int = cfg.get("n_substeps", 1)

        self._m = cfg.m * torch.ones(self.n_envs, device=device)
        self._arm_l = cfg.arm_l * torch.ones(self.n_envs, device=device)
        self._c_tau = cfg.c_tau * torch.ones(self.n_envs, device=device)
        self._thrust_coeff = cfg.get("thrust_coeff", 1.0) * torch.ones(self.n_envs, device=device)

        self.J_xy = cfg.J.xy * torch.ones(self.n_envs, device=device)
        self.J_z = cfg.J.z * torch.ones(self.n_envs, device=device)
        self.D_xy = cfg.D.xy * torch.ones(self.n_envs, device=device)
        self.D_z = cfg.D.z * torch.ones(self.n_envs, device=device)

        self._v_xy_max = torch.tensor(float("inf"), device=device)
        self._v_z_max = torch.tensor(float("inf"), device=device)
        self._omega_xy_max = torch.tensor(cfg.max_w_xy, device=device)
        self._omega_z_max = torch.tensor(cfg.max_w_z, device=device)
        self._T_max = torch.tensor(cfg.max_T, device=device)
        self._T_min = torch.tensor(cfg.min_T, device=device)

        self._X_lb = torch.tensor(
            [
                -float("inf"),
                -float("inf"),
                -float("inf"),
                -self._v_xy_max,
                -self._v_xy_max,
                -self._v_z_max,
                -1,
                -1,
                -1,
                -1,
                -self._omega_xy_max,
                -self._omega_xy_max,
                -self._omega_z_max,
            ],
            device=device,
        )

        self._X_ub = torch.tensor(
            [
                float("inf"),
                float("inf"),
                float("inf"),
                self._v_xy_max,
                self._v_xy_max,
                self._v_z_max,
                1,
                1,
                1,
                1,
                self._omega_xy_max,
                self._omega_xy_max,
                self._omega_z_max,
            ],
            device=device,
        )

        self._U_lb = torch.tensor(
            [self._T_min, self._T_min, self._T_min, self._T_min],
            device=device,
        )
        self._U_ub = torch.tensor(
            [self._T_max, self._T_max, self._T_max, self._T_max],
            device=device,
        )

        self.controller = RateController(self._m, self._J, self._G, cfg.controller, self.device)

    @property
    def model_name(self) -> str:
        return "quadrotor"

    @property
    def state_dim(self) -> int:
        return 13

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def min_action(self) -> Tensor:
        return self.controller.min_action

    @property
    def max_action(self) -> Tensor:
        return self.controller.max_action

    @property
    def _tau_thrust_matrix(self) -> Tensor:
        c = self._c_tau
        d = self._arm_l / (2**0.5)
        ones = torch.ones(self.n_envs, 4, device=c.device, dtype=c.dtype)
        return torch.stack(
            [
                torch.stack([d, -d, -d, d], dim=-1),
                torch.stack([-d, d, -d, d], dim=-1),
                torch.stack([c, c, -c, -c], dim=-1),
                ones,
            ],
            dim=-2,
        )

    @property
    def _J(self) -> Tensor:
        return self._J_matrix

    @property
    def _J_matrix(self) -> Tensor:
        J = torch.zeros(self.n_envs, 3, 3, device=self.device)
        J[:, 0, 0] = self.J_xy
        J[:, 1, 1] = self.J_xy
        J[:, 2, 2] = self.J_z
        return J

    @property
    def _J_inv(self) -> Tensor:
        J_inv = torch.zeros(self.n_envs, 3, 3, device=self.device)
        J_inv[:, 0, 0] = 1.0 / self.J_xy
        J_inv[:, 1, 1] = 1.0 / self.J_xy
        J_inv[:, 2, 2] = 1.0 / self.J_z
        return J_inv

    @property
    def _D_matrix(self) -> Tensor:
        D = torch.zeros(self.n_envs, 3, 3, device=self.device)
        D[:, 0, 0] = self.D_xy
        D[:, 1, 1] = self.D_xy
        D[:, 2, 2] = self.D_z
        return D

    @property
    def _G(self) -> torch.Tensor:
        return torch.tensor(self.cfg.g, device=self.device, dtype=torch.float32)

    @property
    def _G_vec(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, -self.cfg.g], device=self.device, dtype=torch.float32)

    def _quat_rotate(self, q, v):
        w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
        return torch.stack(
            [
                2 * (w * v[..., 1] + x * v[..., 2] - y * v[..., 0]),
                2 * (-w * v[..., 0] + y * v[..., 2] + z * v[..., 0]),
                2 * (-x * v[..., 0] - y * v[..., 1] + z * v[..., 2]),
            ],
            dim=-1,
        )

    def _quat_rotate_inverse(self, q, v):
        w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
        return torch.stack(
            [
                2 * (-w * v[..., 1] - x * v[..., 2] + y * v[..., 0]),
                2 * (w * v[..., 0] - y * v[..., 2] + z * v[..., 0]),
                2 * (x * v[..., 0] + y * v[..., 1] - z * v[..., 2]),
            ],
            dim=-1,
        )

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

    def _quat_inv(self, q):
        return torch.stack([q[..., 0], q[..., 1], q[..., 2], -q[..., 3]], dim=-1)

    def _quat_axis(self, q, axis):
        return self._quat_rotate(q, torch.tensor([0, 0, 1], device=q.device, dtype=q.dtype).expand_as(axis))

    def _quat_norm(self, q):
        return torch.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2 + q[..., 3] ** 2)

    def _normalize_quat(self, q):
        norm = self._quat_norm(q).unsqueeze(-1)
        return q / norm

    def reset(self, env_ids: Tensor) -> None:
        mask = torch.zeros_like(self._acc, dtype=torch.bool)
        mask[env_ids] = True
        self._acc = torch.where(mask, 0.0, self._acc)

    def dynamics(self, X, U):
        p = X[..., :3]
        q = X[..., 3:7]
        v = X[..., 7:10]
        w = X[..., 10:13]

        thrust, torque = self.controller(q, w, U)

        M = torque - torch.cross(w.unsqueeze(-1), torch.matmul(self._J_matrix, w.unsqueeze(-1)).squeeze(-1), dim=-1)
        w_dot = torch.matmul(self._J_inv, M.unsqueeze(-1)).squeeze(-1)

        fdrag = self._quat_rotate(
            q, torch.matmul(self._D_matrix, self._quat_rotate_inverse(q, v).unsqueeze(-1)).squeeze(-1)
        )

        thrust_acc = self._quat_axis(q, 2) * (thrust / self._m.unsqueeze(-1)).unsqueeze(-1)

        acc = thrust_acc + self._G_vec - fdrag / self._m.unsqueeze(-1)
        self._acc = acc

        q_dot = 0.5 * self._quat_mul(q, torch.cat((w, torch.zeros((q.size(0), 1), device=self.device)), dim=-1))

        X_dot = torch.cat([v, q_dot, acc, w_dot], dim=-1)

        return X_dot

    def step(self, U: Tensor) -> None:
        dt_step = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            X_dot = self.dynamics(self._state, U)
            self._state = self._state + dt_step * X_dot

        q_l = self._quat_norm(self._state[..., 3:7]).detach().unsqueeze(-1)
        self._state[..., 3:7] = self._state[..., 3:7] / q_l

    def compute(self, omega):
        thrusts_ref = self._thrust_coeff * omega**2
        thrust_torque = torch.matmul(self._tau_thrust_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque

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

    @property
    def _a(self) -> Tensor:
        return self._acc
