import torch
from torch import Tensor


def wrap_pi(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def gate_rotmat_w2g(gate_yaw: Tensor) -> Tensor:
    zero = torch.zeros_like(gate_yaw)
    one = torch.ones_like(gate_yaw)
    sin_yaw = torch.sin(gate_yaw)
    cos_yaw = torch.cos(gate_yaw)
    return torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw, zero], dim=-1),
            torch.stack([-sin_yaw, cos_yaw, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )


def transform_w_to_gate(rotmat_w2g: Tensor, vector_w: Tensor) -> Tensor:
    return torch.einsum("nij,nj->ni", rotmat_w2g, vector_w)


def gate_frame_state(
    position_w: Tensor,
    velocity_w: Tensor,
    quaternion_xyzw: Tensor,
    gate_position_w: Tensor,
    gate_yaw: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    rotmat_w2g = gate_rotmat_w2g(gate_yaw)
    pos_g = transform_w_to_gate(rotmat_w2g, gate_position_w - position_w)
    vel_g = transform_w_to_gate(rotmat_w2g, velocity_w)

    q_x, q_y, q_z, q_w = quaternion_xyzw.unbind(dim=-1)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    rpy_g = torch.stack([roll, pitch, wrap_pi(yaw - gate_yaw)], dim=-1)
    return pos_g, vel_g, rpy_g


def gate_crossing(
    prev_position_w: Tensor,
    position_w: Tensor,
    gate_position_w: Tensor,
    gate_yaw: Tensor,
    gate_l1_radius: float,
) -> tuple[Tensor, Tensor]:
    rotmat_w2g = gate_rotmat_w2g(gate_yaw)
    prev_rel = transform_w_to_gate(rotmat_w2g, prev_position_w - gate_position_w)
    curr_rel = transform_w_to_gate(rotmat_w2g, position_w - gate_position_w)
    pass_through = (prev_rel[:, 0] < 0.0) & (curr_rel[:, 0] > 0.0)
    inside_gate = torch.linalg.vector_norm(curr_rel[:, 1:], ord=1, dim=-1) < gate_l1_radius
    gate_passed = pass_through & inside_gate
    gate_collision = pass_through & ~inside_gate
    return gate_passed, gate_collision


def out_of_bounds(position_w: Tensor, env_origins: Tensor, xy_limit: float, z_max: float) -> Tensor:
    local_position = position_w - env_origins
    return torch.any(torch.abs(local_position[:, :2]) > xy_limit, dim=-1) | (local_position[:, 2] > z_max)
