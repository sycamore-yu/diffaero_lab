# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: F722, F821

"""Warp kernels for differentiable drone racing.

All kernels run on GPU via Warp and participate in wp.Tape() differentiation.
Ports the PyTorch reward/observation/gate logic from mdp/ to Warp.
"""

from __future__ import annotations

import math

import warp as wp

# ---------------------------------------------------------------------------
# MLP forward pass
# ---------------------------------------------------------------------------


@wp.kernel
def mlp_forward_kernel(
    obs: wp.array(dtype=float, ndim=2),
    params: wp.array(dtype=float),
    layer_dims: wp.array(dtype=int),
    num_layers: int,
    hidden: wp.array(dtype=float, ndim=2),
    action: wp.array(dtype=float, ndim=2),
):
    """Multi-layer perceptron forward pass with ReLU hidden and Tanh output.

    params layout (flat, concatenated):
        [W0, b0, W1, b1, W2, b2, W3, b3]
      where Wk shape is (out_dim_k, in_dim_k) stored row-major,
      bk shape is (out_dim_k,).

    layer_dims = [obs_dim, h0, h1, h2, action_dim]  length = num_layers + 1

    One Warp thread per environment.
    """
    tid = wp.tid()

    in_dim = layer_dims[0]
    # copy obs → hidden scratch row
    for i in range(in_dim):
        hidden[tid, i] = obs[tid, i]

    param_offset = int(0)
    for layer in range(num_layers):
        in_d = layer_dims[layer]
        out_d = layer_dims[layer + 1]
        w_size = out_d * in_d

        # Compute output = activation(W @ x + b)
        for o in range(out_d):
            acc = params[param_offset + w_size + o]  # bias
            for i in range(in_d):
                acc += params[param_offset + o * in_d + i] * hidden[tid, i]

            if layer < num_layers - 1:
                hidden[tid, o] = wp.max(acc, 0.0)  # ReLU
            else:
                hidden[tid, o] = wp.tanh(acc)

        param_offset += w_size + out_d

    # Copy final layer output to action
    out_d = layer_dims[num_layers]
    for i in range(out_d):
        action[tid, i] = hidden[tid, i]


# ---------------------------------------------------------------------------
# Action → wrench mapping
# ---------------------------------------------------------------------------


@wp.kernel
def action_to_wrench_kernel(
    action: wp.array(dtype=float, ndim=2),
    thrust_scale: float,
    moment_scale: float,
    robot_weight: float,
    num_envs: int,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Map [roll, pitch, yaw, thrust] action to body spatial forces.

    action[:, 0:3] are multiplied by moment_scale to get roll/pitch/yaw torque.
    action[:, 3] is mapped from [-1, 1] to [0, thrust_scale * robot_weight].

    One Warp thread per environment.
    """
    tid = wp.tid()
    if tid >= num_envs:
        return

    roll = action[tid, 0] * moment_scale
    pitch = action[tid, 1] * moment_scale
    yaw = action[tid, 2] * moment_scale
    thrust = thrust_scale * robot_weight * (action[tid, 3] + 1.0) * 0.5

    force = wp.vec3(0.0, 0.0, thrust)
    torque = wp.vec3(roll, pitch, yaw)
    body_f[tid] = wp.spatial_vector(torque, force)


# ---------------------------------------------------------------------------
# Gate-frame observation (ported from tasks/direct/drone_racing/mdp/gates.py)
# ---------------------------------------------------------------------------


@wp.func
def _wrap_pi(angle: float) -> float:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def _rot_z(gate_yaw: float) -> wp.mat33:
    """Rotation matrix about Z axis by gate_yaw."""
    s = wp.sin(gate_yaw)
    c = wp.cos(gate_yaw)
    return wp.mat33(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)


@wp.func
def _rot_mul_vec(r: wp.mat33, v: wp.vec3) -> wp.vec3:
    """Multiply mat33 by vec3."""
    return wp.vec3(
        r[0, 0] * v[0] + r[0, 1] * v[1] + r[0, 2] * v[2],
        r[1, 0] * v[0] + r[1, 1] * v[1] + r[1, 2] * v[2],
        r[2, 0] * v[0] + r[2, 1] * v[1] + r[2, 2] * v[2],
    )


@wp.kernel
def compute_obs_kernel(
    position_w: wp.array(dtype=wp.vec3),
    quaternion_xyzw: wp.array(dtype=wp.vec4),
    linear_velocity_w: wp.array(dtype=wp.vec3),
    target_position_w: wp.array(dtype=wp.vec3),
    target_yaw: wp.array(dtype=float),
    next_target_position_w: wp.array(dtype=wp.vec3),
    next_target_yaw: wp.array(dtype=float),
    obs: wp.array(dtype=float, ndim=2),
):
    """Compute gate-relative observation (13-dim).

    Observation layout:
        pos_g (3), vel_g (3), rpy_g (3), next_gate_rel_pos (3), next_gate_rel_yaw (1)

    One Warp thread per environment.
    """
    tid = wp.tid()

    pos = position_w[tid]
    vel = linear_velocity_w[tid]
    q = quaternion_xyzw[tid]  # xyzw convention
    gate_pos = target_position_w[tid]
    gate_y = target_yaw[tid]
    next_gate_pos = next_target_position_w[tid]
    next_gate_y = next_target_yaw[tid]

    # Build gate→world rotation and transform
    rot = _rot_z(gate_y)

    # Position in gate frame: gate_pos - drone_pos, rotated to gate frame
    rel_w = wp.vec3(gate_pos[0] - pos[0], gate_pos[1] - pos[1], gate_pos[2] - pos[2])
    pos_g = _rot_mul_vec(rot, rel_w)

    # Velocity in gate frame
    vel_g = _rot_mul_vec(rot, vel)

    # Quaternion to RPY (from gates.py gate_frame_state)
    # q = (x, y, z, w) xyzw
    q_x, q_y, q_z, q_w = q[0], q[1], q[2], q[3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = wp.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = wp.asin(wp.clamp(sin_pitch, -1.0, 1.0))

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = wp.atan2(sin_yaw, cos_yaw)

    rpy_gate_yaw = _wrap_pi(yaw - gate_y)

    # Next gate relative position/yaw (in current gate frame)
    next_rel_w = wp.vec3(next_gate_pos[0] - gate_pos[0], next_gate_pos[1] - gate_pos[1], next_gate_pos[2] - gate_pos[2])
    next_rel_pos = _rot_mul_vec(rot, next_rel_w)
    next_rel_yaw = _wrap_pi(next_gate_y - gate_y)

    # Write 13-dim observation
    obs[tid, 0] = pos_g[0]
    obs[tid, 1] = pos_g[1]
    obs[tid, 2] = pos_g[2]
    obs[tid, 3] = vel_g[0]
    obs[tid, 4] = vel_g[1]
    obs[tid, 5] = vel_g[2]
    obs[tid, 6] = roll
    obs[tid, 7] = pitch
    obs[tid, 8] = rpy_gate_yaw
    obs[tid, 9] = next_rel_pos[0]
    obs[tid, 10] = next_rel_pos[1]
    obs[tid, 11] = next_rel_pos[2]
    obs[tid, 12] = next_rel_yaw


# ---------------------------------------------------------------------------
# Gate crossing detection (ported from gates.py)
# ---------------------------------------------------------------------------


@wp.kernel
def gate_crossing_kernel(
    prev_position_w: wp.array(dtype=wp.vec3),
    position_w: wp.array(dtype=wp.vec3),
    gate_position_w: wp.array(dtype=wp.vec3),
    gate_yaw: wp.array(dtype=float),
    gate_l1_radius: float,
    gate_passed: wp.array(dtype=int),
    gate_collision: wp.array(dtype=int),
):
    """Detect gate crossing: passed or collided.

    One Warp thread per environment.
    """
    tid = wp.tid()

    rot = _rot_z(gate_yaw[tid])

    prev_rel = _rot_mul_vec(rot, prev_position_w[tid] - gate_position_w[tid])
    curr_rel = _rot_mul_vec(rot, position_w[tid] - gate_position_w[tid])

    pass_through = (prev_rel[0] < 0.0) and (curr_rel[0] > 0.0)
    inside_gate = abs(curr_rel[1]) + abs(curr_rel[2]) < gate_l1_radius

    gate_passed[tid] = 1 if (pass_through and inside_gate) else 0
    gate_collision[tid] = 1 if (pass_through and not inside_gate) else 0


# ---------------------------------------------------------------------------
# Differentiable loss (ported from rewards.py)
# ---------------------------------------------------------------------------


@wp.kernel
def compute_loss_kernel(
    position_w: wp.array(dtype=wp.vec3),
    prev_position_w: wp.array(dtype=wp.vec3),
    target_position_w: wp.array(dtype=wp.vec3),
    angular_velocity_b: wp.array(dtype=wp.vec3),
    action: wp.array(dtype=float, ndim=2),
    prev_action: wp.array(dtype=float, ndim=2),
    rew_scale_progress: float,
    rew_scale_tracking: float,
    rew_scale_ang_vel: float,
    rew_scale_control_effort: float,
    step_loss: wp.array(dtype=float),
):
    """Compute per-step differentiable loss matching rewards.py semantics.

    Loss = rew_scale_progress * progress_loss
         + abs(rew_scale_tracking) * pos_loss
         + abs(rew_scale_ang_vel) * ang_rate_loss
         + abs(rew_scale_control_effort) * jerk_loss

    All components retain gradients through position_w and action.
    In wp.Tape(), this means full gradient signal through trajectory.

    One Warp thread per environment.
    """
    tid = wp.tid()

    pos = position_w[tid]
    prev_pos = prev_position_w[tid]
    target = target_position_w[tid]

    # Distance to gate
    dx = pos[0] - target[0]
    dy = pos[1] - target[1]
    dz = pos[2] - target[2]
    d2g = wp.sqrt(dx * dx + dy * dy + dz * dz)

    pdx = prev_pos[0] - target[0]
    pdy = prev_pos[1] - target[1]
    pdz = prev_pos[2] - target[2]
    d2g_prev = wp.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)

    # Progress loss: positive = moved away from gate
    progress_loss = d2g - d2g_prev

    # Position loss: saturates far from gate
    pos_loss = 1.0 - wp.exp(-d2g)

    # Angular rate penalty
    av = angular_velocity_b[tid]
    ang_rate_loss = wp.sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2])

    # Action smoothness (jerk)
    jerk = 0.0
    for d in range(4):
        diff = action[tid, d] - prev_action[tid, d]
        jerk += diff * diff

    # Combined loss (all components retain gradients in wp.Tape)
    loss = 0.0
    loss += rew_scale_progress * progress_loss
    loss += abs(rew_scale_tracking) * pos_loss
    loss += abs(rew_scale_ang_vel) * ang_rate_loss
    loss += abs(rew_scale_control_effort) * jerk

    step_loss[tid] = loss


# ---------------------------------------------------------------------------
# Utility: init MLP params
# ---------------------------------------------------------------------------


def init_mlp_params(
    layer_dims: list[int],
    device: str | None = None,
    requires_grad: bool = True,
) -> wp.array:
    """Initialize MLP parameters with Xavier uniform initialization.

    Args:
        layer_dims: [obs_dim, h0, h1, ..., action_dim] e.g. [13, 256, 128, 64, 4]
        device: CUDA device or None for default.
        requires_grad: Whether params require gradients.

    Returns:
        Flat wp.array of all weights and biases, requires_grad=True.
    """
    from typing import List, Tuple  # noqa: F811

    import numpy as np

    num_layers = len(layer_dims) - 1

    # Compute total param count
    total = 0
    offsets: List[int] = []
    layer_shape_list: List[Tuple[int, int]] = []
    for li in range(num_layers):
        in_d = layer_dims[li]
        out_d = layer_dims[li + 1]
        offsets.append(total)
        total += out_d * in_d + out_d  # weights + bias
        layer_shape_list.append((out_d, in_d))

    params_np = np.zeros(total, dtype=np.float32)
    rng = np.random.default_rng(42)

    for li in range(num_layers):
        out_d, in_d = layer_shape_list[li]
        off = offsets[li]
        w_size = out_d * in_d
        # Xavier uniform
        limit = math.sqrt(6.0 / (in_d + out_d))
        params_np[off : off + w_size] = rng.uniform(-limit, limit, size=w_size)
        # biases zero
        params_np[off + w_size : off + w_size + out_d] = 0.0

    params = wp.array(params_np, dtype=float, device=device, requires_grad=requires_grad)
    return params


def param_count(layer_dims: list[int]) -> int:
    """Return total number of scalar parameters for given layer layout."""
    num_layers = len(layer_dims) - 1
    total = 0
    for li in range(num_layers):
        in_d = layer_dims[li]
        out_d = layer_dims[li + 1]
        total += out_d * in_d + out_d
    return total
