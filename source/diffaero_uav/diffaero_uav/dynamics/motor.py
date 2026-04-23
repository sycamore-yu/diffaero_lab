# Copyright (c) 2025, Kousheek Chakraborty
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class Motor:
    def __init__(self, num_envs, taus, init, max_rate, min_rate, dt, use, device="cpu", dtype=torch.float32):
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        self.omega = torch.tensor(init, device=device).expand(num_envs, -1).clone()

        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)
        self.max_rate = torch.tensor(max_rate, device=device).expand(num_envs, -1)
        self.min_rate = torch.tensor(min_rate, device=device).expand(num_envs, -1)

    def compute(self, omega_ref):
        if not self.use:
            self.omega = omega_ref
            return self.omega

        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids):
        if isinstance(env_ids, slice):
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(num_envs, -1)
