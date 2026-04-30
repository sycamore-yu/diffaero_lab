import numpy as np
import torch
import warp as wp

from diffaero_lab.algo.algorithms.apg_stochastic import APGStochastic, APGStochasticConfig
from diffaero_lab.algo.trainers.apg_stochastic_trainer import APGStochasticTrainer
from diffaero_lab.algo.wrappers.env_adapter import DifferentialEnvAdapter
from diffaero_lab.common.keys import EXTRA_SIM_STATE, EXTRA_TASK_TERMS, OBS_CRITIC, OBS_POLICY
from diffaero_lab.tasks.direct.drone_racing.drone_racing_env import DroneRacingEnv
from diffaero_lab.tasks.direct.drone_racing.mdp.gates import gate_crossing
from diffaero_lab.tasks.direct.drone_racing.mdp.resets import reset_body_state
from diffaero_lab.tasks.direct.drone_racing.state.policy import build_policy_obs
from diffaero_lab.tasks.direct.drone_racing.dynamics_bridge.quad import QuadDynamicsBridge


class _ActionSpace:
    shape = (4,)
    high = np.ones(4, dtype=np.float32)
    low = -np.ones(4, dtype=np.float32)


class _GradCapturingEnv:
    action_space = _ActionSpace()

    def __init__(self):
        self.last_action = None
        self.unwrapped = self
        self.num_envs = 2
        self.device = "cpu"

    def reset(self):
        observations = {
            OBS_POLICY: torch.zeros(self.num_envs, 3),
            OBS_CRITIC: torch.zeros(self.num_envs, 3),
        }
        extras = {
            EXTRA_TASK_TERMS: {"loss": torch.zeros(self.num_envs)},
            EXTRA_SIM_STATE: {"dynamics": {"tensor_backend": "warp"}},
        }
        return observations, extras

    def step(self, action):
        self.last_action = action
        observations, extras = self.reset()
        rewards = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return observations, rewards, terminated, truncated, extras


class _GraphCarryingEnv:
    action_space = _ActionSpace()

    def __init__(self):
        self.unwrapped = self
        self.num_envs = 2
        self.device = "cpu"
        self.actions = torch.zeros(self.num_envs, 4)
        self.prev_action = torch.zeros_like(self.actions)
        self.detach_calls = 0

    def reset(self):
        observations = {
            OBS_POLICY: torch.ones(self.num_envs, 3),
            OBS_CRITIC: torch.ones(self.num_envs, 3),
        }
        extras = {
            EXTRA_TASK_TERMS: {"loss": torch.zeros(self.num_envs)},
            EXTRA_SIM_STATE: {},
        }
        return observations, extras

    def step(self, action):
        self.prev_action = self.actions.clone()
        self.actions = action.clone()
        observations, extras = self.reset()
        extras[EXTRA_TASK_TERMS]["loss"] = self.actions.sum(dim=-1) + 0.1 * self.prev_action.sum(dim=-1)
        rewards = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return observations, rewards, terminated, truncated, extras

    def detach(self):
        self.detach_calls += 1
        self.actions = self.actions.detach()
        self.prev_action = self.prev_action.detach()


class _Cfg:
    thrust_scale = 1.9
    moment_scale = 0.01

    class sim:
        gravity = (0.0, 0.0, -9.81)


class _RobotData:
    body_mass = torch.tensor([[0.027]])


class _WrenchComposer:
    def __init__(self):
        self.forces = None
        self.torques = None
        self.body_ids = None

    def set_forces_and_torques_index(self, body_ids, forces, torques):
        self.body_ids = body_ids
        self.forces = forces.clone()
        self.torques = torques.clone()


class _Robot:
    def __init__(self):
        self.data = _RobotData()
        self.permanent_wrench_composer = _WrenchComposer()

    def find_bodies(self, pattern):
        assert pattern == "body"
        return torch.tensor([0], dtype=torch.int32), ["body"]


class _IndexTrackingTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.last_index_dtype = None

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            self.last_index_dtype = idx.dtype
        return self.tensor[idx]


class _ResetRobotData:
    def __init__(self):
        self.default_joint_pos = _IndexTrackingTensor(torch.randn(4, 2))
        self.default_joint_vel = _IndexTrackingTensor(torch.randn(4, 2))
        self.default_root_state = _IndexTrackingTensor(torch.randn(4, 13))


class _ResetRobot:
    def __init__(self):
        self.data = _ResetRobotData()
        self.root_pose_env_ids = None
        self.root_vel_env_ids = None
        self.joint_env_ids = None

    def write_root_pose_to_sim(self, root_pose, env_ids):
        self.root_pose_env_ids = env_ids

    def write_root_velocity_to_sim(self, root_velocity, env_ids):
        self.root_vel_env_ids = env_ids

    def write_joint_state_to_sim(self, joint_pos, joint_vel, joint_ids, env_ids):
        self.joint_env_ids = env_ids


def test_env_adapter_preserves_action_gradient_for_differentiable_losses():
    env = _GradCapturingEnv()
    adapter = DifferentialEnvAdapter(env)

    action = torch.full((2, 4), 0.25, requires_grad=True)
    adapter.step(action)

    assert env.last_action.requires_grad
    loss = env.last_action.square().sum()
    loss.backward()
    assert action.grad is not None


def test_apg_stochastic_record_loss_uses_direct_loss_gradient():
    cfg = APGStochasticConfig(rollout_horizon=1, hidden_dims=(8,), entropy_coef=0.0)
    apg = APGStochastic(cfg=cfg, obs_dim=3, action_dim=2, device="cpu")

    obs = torch.ones(4, 3)
    action, policy_info = apg.act(obs, test=True)
    policy_info["log_prob"] = torch.full_like(policy_info["log_prob"], float("nan"))
    direct_loss = action.sum(dim=-1)

    apg.record_loss(direct_loss, policy_info, {})
    losses, grad_norms = apg.update_actor()

    assert np.isfinite(losses["actor_loss"])
    assert grad_norms["actor_grad_norm"] > 0.0


def test_apg_stochastic_policy_gradient_loss_updates_actor_for_physx_rewards():
    cfg = APGStochasticConfig(rollout_horizon=1, hidden_dims=(8,), entropy_coef=0.0)
    apg = APGStochastic(cfg=cfg, obs_dim=3, action_dim=2, device="cpu")

    obs = torch.arange(12, dtype=torch.float32).view(4, 3) / 12.0
    _, policy_info = apg.act(obs)
    reward = torch.tensor([1.0, -1.0, 0.5, -0.5])

    apg.record_policy_gradient_loss(reward, policy_info)
    losses, grad_norms = apg.update_actor()

    assert np.isfinite(losses["actor_loss"])
    assert grad_norms["actor_grad_norm"] > 0.0


def test_quad_bridge_maps_zero_thrust_action_near_hover_weight():
    robot = _Robot()
    bridge = QuadDynamicsBridge(_Cfg(), robot, num_envs=2, device="cpu")

    bridge.process_action(torch.zeros(2, 4))
    bridge.apply_to_sim()

    expected = 1.9 * 0.027 * 9.81 / 2.0
    assert torch.allclose(robot.permanent_wrench_composer.forces[:, 0, 2], torch.full((2,), expected))
    assert torch.equal(robot.permanent_wrench_composer.body_ids, torch.zeros(1, dtype=torch.int32))


def test_gate_crossing_matches_diffaero_plane_and_l1_opening():
    gate_position = torch.zeros(3, 3)
    gate_yaw = torch.zeros(3)
    prev_position = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 2.0, 0.0], [-1.0, 0.0, 2.0]])
    curr_position = torch.tensor([[1.0, 0.2, 0.2], [1.0, 2.0, 0.0], [1.0, 0.0, 2.0]])

    passed, collision = gate_crossing(prev_position, curr_position, gate_position, gate_yaw, gate_l1_radius=1.5)

    assert torch.equal(passed, torch.tensor([True, False, False]))
    assert torch.equal(collision, torch.tensor([False, True, True]))


def test_policy_observation_uses_gate_relative_state():
    obs = build_policy_obs(
        position_w=torch.tensor([[-1.0, 0.0, 1.0]]),
        quaternion_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        linear_velocity_w=torch.tensor([[2.0, 0.0, 0.0]]),
        angular_velocity_b=torch.zeros(1, 3),
        last_action=torch.zeros(1, 4),
        target_position_w=torch.tensor([[0.0, 0.0, 1.0]]),
        target_yaw=torch.zeros(1),
        next_target_position_w=torch.tensor([[1.0, 2.0, 1.0]]),
        next_target_yaw=torch.tensor([torch.pi / 2]),
    )

    assert obs.shape == (1, 13)
    assert torch.allclose(obs[0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(obs[0, 3:6], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(obs[0, 6:9], torch.zeros(3))
    assert torch.allclose(obs[0, 9:12], torch.tensor([1.0, 2.0, 0.0]))
    assert torch.allclose(obs[0, 12], torch.tensor(torch.pi / 2))


def test_split_reset_env_ids_separates_sim_and_tensor_index_dtypes():
    env_ids_sim, env_ids_index = DroneRacingEnv._split_reset_env_ids(torch.tensor([0, 3], dtype=torch.int32), "cpu")

    assert env_ids_sim.dtype == torch.int32
    assert env_ids_index.dtype == torch.long
    assert torch.equal(env_ids_sim.to(torch.long), env_ids_index)


def test_reset_body_state_uses_long_tensor_indices_and_keeps_sim_ids(monkeypatch):
    monkeypatch.setattr(wp, "to_torch", lambda value: value)
    robot = _ResetRobot()
    env_ids_sim = torch.tensor([1, 3], dtype=torch.int32)
    env_origins = torch.zeros(4, 3)

    reset_body_state(robot, env_ids_sim, env_origins, "cpu", root_position_w=torch.ones(2, 3))

    assert robot.data.default_joint_pos.last_index_dtype == torch.long
    assert robot.data.default_joint_vel.last_index_dtype == torch.long
    assert robot.data.default_root_state.last_index_dtype == torch.long
    assert robot.root_pose_env_ids.dtype == torch.int32
    assert robot.root_vel_env_ids.dtype == torch.int32
    assert robot.joint_env_ids.dtype == torch.int32


def test_apg_stochastic_trainer_detaches_env_graph_between_iterations():
    env = _GraphCarryingEnv()
    adapter = DifferentialEnvAdapter(env)
    cfg = APGStochasticConfig(rollout_horizon=1, hidden_dims=(8,), entropy_coef=0.0)
    trainer = APGStochasticTrainer(adapter, cfg)

    trainer.train(max_iterations=2)

    assert env.detach_calls == 2
