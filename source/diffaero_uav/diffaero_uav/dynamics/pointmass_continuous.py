# Copyright (c) 2025, DiffAero Authors
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from diffaero_uav.dynamics.base import BaseDynamics


class ContinuousPointMassModel(BaseDynamics):
    """Point-mass dynamics with continuous-time first-order thrust filtering.

    State layout (9-dim): [p(3), v(3), a_thrust(3)]
    Action layout (3-dim): a_thrust_cmd in m/s^2 (world or local frame).
    """

    def __init__(self, cfg: "DictConfig", device: torch.device):
        super().__init__(cfg, device)
        self.n_envs = getattr(cfg, "n_envs", 1)
        self.n_agents = getattr(cfg, "n_agents", 1)
        self.dt = getattr(cfg, "dt", 0.01)
        self.action_frame = getattr(cfg, "action_frame", "world")

        self._state = torch.zeros(self.n_envs, self.n_agents, self.state_dim, device=device)
        self._vel_ema = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        self._acc = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        xyz = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        w = torch.ones(self.n_envs, self.n_agents, 1, device=device)
        self.quat_xyzw = torch.cat([xyz, w], dim=-1)
        self.quat_xyzw_init = self.quat_xyzw.clone()
        if self.n_agents == 1:
            self._state.squeeze_(1)
            self._vel_ema.squeeze_(1)
            self._acc.squeeze_(1)
            self.quat_xyzw.squeeze_(1)
            self.quat_xyzw_init.squeeze_(1)

        self._D = getattr(cfg, "D", 0.1) * torch.ones(self.n_envs, self.n_agents, 1, device=device)
        self._lmbda = getattr(cfg, "lmbda", 10.0) * torch.ones(self.n_envs, self.n_agents, 1, device=device)
        if self.n_agents == 1:
            self._D = self._D.squeeze_(1)
            self._lmbda = self._lmbda.squeeze_(1)

        _max_acc_cfg = getattr(cfg, "max_acc", None)
        if _max_acc_cfg is not None:
            max_xy_val = _max_acc_cfg.xy
            max_z_val = _max_acc_cfg.z
        else:
            max_xy_val = 10.0
            max_z_val = 10.0
        self.max_acc_xy = max_xy_val * torch.ones(self.n_envs, self.n_agents, device=device)
        self.max_acc_z = max_z_val * torch.ones(self.n_envs, self.n_agents, device=device)

        self.vel_ema_factor = getattr(cfg, "vel_ema_factor", 0.0)
        self.align_yaw_with_vel_ema = getattr(cfg, "align_yaw_with_vel_ema", True)
        self.align_yaw_with_target_direction = getattr(cfg, "align_yaw_with_target_direction", False)

        self._G_vec = torch.tensor([0.0, 0.0, -getattr(cfg, "g", 9.81)], device=device, dtype=torch.float32)
        self._X_lb = torch.tensor(
            [
                -float("inf"),
                -float("inf"),
                -float("inf"),
                -float("inf"),
                -float("inf"),
                -float("inf"),
                -max_xy_val,
                -max_xy_val,
                -max_z_val,
            ],
            device=device,
        )
        self._X_ub = torch.tensor(
            [
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                max_xy_val,
                max_xy_val,
                max_z_val,
            ],
            device=device,
        )

        self.n_substeps = getattr(cfg, "n_substeps", 1)
        self.Rz_temp: Tensor

    @property
    def model_name(self) -> str:
        return "pmc"

    @property
    def state_dim(self) -> int:
        return 9

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def min_action(self) -> Tensor:
        zero = torch.zeros_like(self.max_acc_xy)
        min_action = torch.stack([-self.max_acc_xy, -self.max_acc_xy, zero], dim=-1)
        if self.n_agents == 1:
            min_action = min_action.squeeze_(0)
        return min_action

    @property
    def max_action(self) -> Tensor:
        max_action = torch.stack([self.max_acc_xy, self.max_acc_xy, self.max_acc_z], dim=-1)
        if self.n_agents == 1:
            max_action = max_action.squeeze_(0)
        return max_action

    @property
    def Rz(self) -> Tensor:
        q = self.quat_xyzw
        w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
        yaw = torch.atan2(2 * (w * 0 + x * 0 - y * 0 + z * 1), w**2 + x**2 - y**2 - z**2 + 1e-7)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        Rz = torch.zeros(*yaw.shape, 3, 3, device=yaw.device, dtype=yaw.dtype)
        Rz[..., 0, 0] = cos_yaw
        Rz[..., 0, 1] = -sin_yaw
        Rz[..., 1, 0] = sin_yaw
        Rz[..., 1, 1] = cos_yaw
        Rz[..., 2, 2] = 1.0
        return Rz

    @property
    def _p(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, :3]
        return self._state[:, :, :3]

    @property
    def _v(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, 3:6]
        return self._state[:, :, 3:6]

    @property
    def _a(self) -> Tensor:
        return self._acc

    @property
    def _a_thrust(self) -> Tensor:
        if self.n_agents == 1:
            return self._state[:, 6:9]
        return self._state[:, :, 6:9]

    @property
    def q(self) -> Tensor:
        return self.quat_xyzw

    def _compute_Rz_local(self) -> Tensor:
        q = self.quat_xyzw
        if self.n_agents == 1:
            w, x, y, z = q[3], q[0], q[1], q[2]
        else:
            w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
        yaw = torch.atan2(2 * (w * 0 + x * 0 - y * 0 + z * 1), w**2 + x**2 - y**2 - z**2 + 1e-7)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        shape = yaw.shape
        Rz = torch.zeros(*shape, 3, 3, device=yaw.device, dtype=yaw.dtype)
        Rz[..., 0, 0] = cos_yaw
        Rz[..., 0, 1] = -sin_yaw
        Rz[..., 1, 0] = sin_yaw
        Rz[..., 1, 1] = cos_yaw
        Rz[..., 2, 2] = 1.0
        return Rz

    def reset(self, env_ids: Tensor) -> None:
        mask = torch.zeros(*self._vel_ema.shape[:-1], dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        mask3 = mask.unsqueeze(-1).expand_as(self._vel_ema)
        self._vel_ema = torch.where(mask3, 0.0, self._vel_ema)
        self._acc = torch.where(mask3, 0.0, self._acc)
        mask4 = mask.unsqueeze(-1).expand_as(self.quat_xyzw)
        self.quat_xyzw = torch.where(mask4, self.quat_xyzw_init, self.quat_xyzw)

    def _update_state(self, next_state: Tensor) -> None:
        self._state = next_state
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor)
        self._acc = self._a_thrust + self._G_vec - self._D * self._v
        orientation = self._vel_ema if self.align_yaw_with_vel_ema else self._v
        self.quat_xyzw = self._point_mass_quat(self._a_thrust, orientation)

    def _point_mass_quat(self, a: Tensor, orientation: Tensor) -> Tensor:
        a_norm = torch.norm(a, dim=-1, keepdim=True).clamp(min=1e-7)
        up = a / a_norm
        yaw = torch.atan2(orientation[..., 1], orientation[..., 0])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        mat_yaw = torch.zeros(*yaw.shape, 3, 3, device=yaw.device, dtype=yaw.dtype)
        mat_yaw[..., 0, 0] = cos_yaw
        mat_yaw[..., 0, 1] = -sin_yaw
        mat_yaw[..., 1, 0] = sin_yaw
        mat_yaw[..., 1, 1] = cos_yaw
        mat_yaw[..., 2, 2] = 1.0
        new_up = torch.matmul(mat_yaw.transpose(-2, -1), up.unsqueeze(-1)).squeeze(-1)
        z = torch.zeros_like(new_up)
        z[..., 2] = 1.0
        quat_axis = torch.linalg.cross(z, new_up, dim=-1)
        quat_axis_norm = torch.norm(quat_axis, dim=-1, keepdim=True).clamp(min=1e-7)
        quat_axis = quat_axis / quat_axis_norm
        cos = torch.sum(new_up * z, dim=-1, keepdim=True)
        sin = torch.norm(new_up[..., :2], dim=-1, keepdim=True) / (torch.norm(new_up, dim=-1, keepdim=True) + 1e-7)
        quat_angle = torch.atan2(sin.squeeze(-1), cos.squeeze(-1)).unsqueeze(-1)
        quat_pitch_roll_xyz = quat_axis * torch.sin(0.5 * quat_angle)
        quat_pitch_roll_w = torch.cos(0.5 * quat_angle)
        quat_pitch_roll = torch.cat([quat_pitch_roll_xyz, quat_pitch_roll_w], dim=-1)
        quat_pitch_roll = quat_pitch_roll / (torch.norm(quat_pitch_roll, dim=-1, keepdim=True) + 1e-7)
        yaw_half = yaw.unsqueeze(-1) / 2
        quat_yaw_xyz = torch.zeros_like(yaw_half)
        quat_yaw_xyz[..., 2] = torch.sin(yaw_half[..., 0])
        quat_yaw_w = torch.cos(yaw_half[..., 0])
        quat_yaw = torch.cat([quat_yaw_xyz, quat_yaw_w], dim=-1)
        w1, x1, y1, z1 = quat_yaw[..., 3], quat_yaw[..., 0], quat_yaw[..., 1], quat_yaw[..., 2]
        w2, x2, y2, z2 = (
            quat_pitch_roll[..., 3],
            quat_pitch_roll[..., 0],
            quat_pitch_roll[..., 1],
            quat_pitch_roll[..., 2],
        )
        quat_xyzw = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )
        return quat_xyzw / (torch.norm(quat_xyzw, dim=-1, keepdim=True) + 1e-7)

    def _dynamics(self, X: Tensor, U: Tensor) -> Tensor:
        p, v, a_thrust = X[..., :3], X[..., 3:6], X[..., 6:9]
        p_dot = v
        fdrag = -self._D * v
        v_dot = a_thrust + self._G_vec + fdrag
        control_delay_factor = (1 - torch.exp(-self._lmbda * self.dt)) / self.dt
        a_thrust_cmd = U
        a_dot = control_delay_factor.unsqueeze(-1) * (a_thrust_cmd - a_thrust)
        X_dot = torch.cat([p_dot, v_dot, a_dot], dim=-1)
        return X_dot

    def _euler_integrate(self, X: Tensor, U: Tensor, dt: float, n_substeps: int) -> Tensor:
        dt_step = dt / n_substeps
        state = X
        for _ in range(n_substeps):
            X_dot = self._dynamics(state, U)
            state = state + dt_step * X_dot
        return state

    def step(self, U: Tensor) -> None:
        dt_step = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            X_dot = self._dynamics(self._state, U)
            self._state = self._state + dt_step * X_dot
        self._update_state(self._state)

    def compute(self, *args, **kwargs) -> Tensor:
        return self._a_thrust

    @property
    def _tau_thrust_matrix(self) -> Tensor:
        return torch.zeros(self.n_envs, 4, 4, device=self.device)
