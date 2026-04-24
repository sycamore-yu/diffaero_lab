# DiffAero 到 IsaacLab + Newton 迁移架构设计

## 1. 目标

本设计用于将 `refer/diffaero/` 迁移到 IsaacLab + Newton 环境，并保持如下目标：

1. 采用 IsaacLab 官方推荐的 own-project 结构，在 `source/` 下组织多个 extension。
2. 以 `drone_racing` 作为首个最小闭环任务。
3. 同时支持 IsaacLab 现成 RL baseline 与自定义 differential learning 算法。
4. 将环境、算法、共享契约、无人机平台能力解耦，支持后续扩展到更多无人机任务与多种动力学模型。

## 2. 总体结论

建议采用四个轻量 extension 的结构：

```text
source/
├── diffaero_lab/
```

其中：

- `diffaero_lab.env`：IsaacLab 任务 extension，负责 `DirectRLEnv`、任务配置、Gym 注册、任务语义与状态导出。
- `diffaero_lab.algo`：differential learning extension，负责 `APG`、`APG_stochastic`、`SHAC`、`SHA2C`、trainer、buffer、wrapper。
- `diffaero_lab.common`：共享 schema extension，负责 keys、capabilities、task terms schema、flatten adapter。
- `diffaero_lab.uav`：无人机平台能力 extension，负责共享资产、airframe 参数、控制分配、电机模型、多动力学模型与平台适配。

该组织方式符合 IsaacLab 文档中“一个 project 在 `source/` 下拥有多个 extensions”的推荐方向，也符合“自定义学习库通过 wrapper 接入 IsaacLab，必要时可以创建新的 wrapper module”的官方说明。

### 2.1 与当前 `diffaero_lab` 模板的关系

当前仓库仍然是单 extension 模板：`source/diffaero_lab/`。本设计采用“新建四包并逐步替换模板包”的迁移方式，而不是在 `diffaero_lab` 内继续堆叠所有逻辑。

迁移约定如下：

1. `source/diffaero_lab/` 作为模板遗留包，首阶段保留不动，用于避免迁移早期破坏当前工作区。
2. 新的四个包作为并列的 net-new package 建立在 `source/` 下。
3. 当 `drone_racing + RL baseline + APG` 跑通后，再决定是否删除、归档或保留 `diffaero_lab` 作为兼容壳层。
4. 实施阶段默认按“四个独立可编辑安装 extension package”处理，四者都保留 `config/extension.toml + setup.py`，以贴近 IsaacLab own-project 在 `source/` 下组织多个 extension 的标准形态。

## 3. 首发范围

首发任务固定为 `drone_racing`，采用以下边界：

- 环境主实现：`DirectRLEnv`
- RL baseline：优先复用 IsaacLab 现成 RL 算法与训练脚本模式
- differential learning：首版完成 `APG`、`APG_stochastic`、`SHAC`、`SHA2C`
- 动力学后端：首版按 `quad -> pmd -> pmc -> simple` 顺序实现
- 配置系统：任务配置树与算法配置树并列，运行时用 Hydra 组合

## 4. 包职责与依赖方向

### 4.1 依赖方向

```text
diffaero_lab.env   -> diffaero_lab.uav
diffaero_lab.env   -> diffaero_lab.common
diffaero_lab.algo  -> diffaero_lab.common
diffaero_lab.algo  -> diffaero_lab.uav   # 尽量少，仅在需要平台元信息时使用
```

`diffaero_lab.algo` 通过 `gym.make(task_id)`、wrapper 与 `diffaero_lab.common` 接入环境，避免直接依赖具体任务实现文件。

### 4.2 `diffaero_lab.env`

建议结构：

```text
source/diffaero_lab/
├── config/extension.toml
├── setup.py
└── diffaero_lab/
    ├── __init__.py
    ├── env/
    │   └── tasks/
    │       └── direct/
    │           └── drone_racing/
    │               ├── __init__.py
    │               ├── drone_racing_env.py
    │               ├── drone_racing_env_cfg.py
    │               ├── dynamics_bridge/
    │               ├── scene/
    │               ├── mdp/
    │               ├── state/
    │               └── agents/
    └── utils/
```

职责：

- 实现 `DirectRLEnv`
- 注册 Gym task ID
- 组织 scene、reset、action、done、reward
- 导出 `observations[OBS_POLICY]`、`observations[OBS_CRITIC]`、`extras[EXTRA_TASK_TERMS]`、`extras[EXTRA_SIM_STATE]`

### 4.3 `diffaero_lab.algo`

建议结构：

```text
source/diffaero_lab/
├── config/extension.toml
├── setup.py
└── diffaero_lab/
    ├── __init__.py
    ├── algo/
    │   ├── algorithms/
    │   ├── trainers/
    │   ├── buffers/
    │   ├── wrappers/
    │   ├── models/
    │   └── configs/
```

职责：

- 实现 `APG`、`APG_stochastic`、`SHAC`、`SHA2C`
- 实现 rollout / unroll、loss 组装、detach 策略、梯度裁剪
- 提供 differential wrapper 与训练入口

### 4.4 `diffaero_lab.common`

建议结构：

```text
source/diffaero_lab/
├── config/extension.toml
├── setup.py
└── diffaero_lab/
    ├── __init__.py
    └── common/
        ├── __init__.py
        ├── keys.py
        ├── capabilities.py
        ├── terms.py
        └── adapters/
            ├── flatten.py
            └── sim_state.py
```

职责：

- 统一 key 名、capability 名、task terms schema
- 提供 `sim_state` flatten / unflatten adapter
- 保持任务无关、算法无关、平台无关

安装属性：

- `diffaero_lab.common` 仍然保持"极小共享层"的职责。
- 为了与 IsaacLab own-project 结构保持一致，它作为轻量 extension 存在，但不承担 Gym 注册、环境生命周期或训练脚本职责。
- 该 extension 只暴露共享 schema、adapter 和常量定义。

### 4.5 `diffaero_lab.uav`

建议结构：

```text
source/diffaero_lab/
├── config/extension.toml
├── setup.py
└── diffaero_lab/
    ├── __init__.py
    └── uav/
        ├── __init__.py
        ├── assets/
    │   ├── drone_assets.py
    │   ├── airframes.py
    │   └── sensors.py
    ├── dynamics/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── registry.py
    │   ├── allocation.py
    │   ├── motor.py
    │   ├── quadrotor.py
    │   ├── pointmass_discrete.py
    │   ├── pointmass_continuous.py
    │   ├── simplified_quadrotor.py
    │   └── controller.py
    ├── control/
    ├── params/
    └── adapters/
        ├── isaaclab.py
        └── newton.py
```

职责：

- 放置跨任务复用的无人机资产与 airframe 参数
- 放置共享控制分配、电机模型与多动力学模型
- 为 IsaacLab 与 Newton 提供平台级适配能力

其中：

- `isaaclab.py`：负责把 `diffaero_lab.uav` 的平台参数、动力学能力、控制分配接到 IsaacLab 环境与 bridge。
- `newton.py`：负责把同一批平台参数与动力学元信息接到 Newton 可微物理相关的 differential trainer 或 solver adapter。
- 这两个文件都属于平台适配层，因此放在 `diffaero_lab.uav`，而不是 `diffaero_lab.common`。

## 5. `drone_racing` 任务设计

### 5.1 单任务语义 + 多动力学后端

`drone_racing` 保持单一任务入口，不为每种动力学复制一套环境。通过 `dynamics_bridge/` 实现动力学切换。

建议结构：

```text
tasks/direct/drone_racing/
├── drone_racing_env.py
├── drone_racing_env_cfg.py
├── dynamics_bridge/
│   ├── base.py
│   ├── quad.py
│   ├── pointmass_discrete.py
│   ├── pointmass_continuous.py
│   └── simplified_quad.py
├── scene/
├── mdp/
├── state/
└── agents/
```

### 5.2 `dynamics_bridge` 统一接口

建议桥接接口包含：

- `reset(env_ids)`
- `process_action(action)`
- `apply_to_sim()`
- `read_base_state()`
- `read_motor_state()`
- `read_dynamics_info()`

环境主类通过 `DirectRLEnv` 生命周期驱动 bridge：

- `_pre_physics_step()` 中调用 `process_action(action)`
- `_apply_action()` 中调用 `apply_to_sim()`
- `_get_observations()`、`_get_rewards()`、`_get_dones()` 中读取 `read_base_state()` 等缓存结果

环境主类只负责任务语义，不直接持有第二套环境主循环。

桥接接口到 contract 字段的映射固定如下：

| bridge 方法 | contract 输出字段 |
|---|---|
| `read_base_state()` | `position_w`、`quaternion_w`、`linear_velocity_w`、`angular_velocity_b` |
| `read_motor_state()` | `motor_omega` |
| `read_dynamics_info()` | `dynamics.model_name`、`dynamics.state_layout_version`、`tensor_backend`、`write_mode` |

`sim_state.py` 负责把这些 bridge 输出与任务语义字段一起组装成最终 `extras[EXTRA_SIM_STATE]`。

## 6. 环境与算法之间的 contract

### 6.1 标准观测与扩展输出

环境标准输出：

```python
observations = {"policy": ...}
observations = {"policy": ..., "critic": ...}  # optional
reward
terminated
truncated
```

环境扩展输出：

```python
extras = {
    EXTRA_TASK_TERMS: {...},
    EXTRA_SIM_STATE: {...},
    EXTRA_CAPABILITIES: {...},
    EXTRA_DYNAMICS_INFO: {...},
}
```

### 6.2 `keys.py`

建议定义：

```python
OBS_POLICY = "policy"
OBS_CRITIC = "critic"

EXTRA_TASK_TERMS = "task_terms"
EXTRA_SIM_STATE = "sim_state"
EXTRA_CAPABILITIES = "capabilities"
EXTRA_DYNAMICS_INFO = "dynamics"
EXTRA_RESET_STATE = "state_before_reset"
EXTRA_TERMINAL_STATE = "terminal_state"
```

### 6.3 `capabilities.py`

建议定义：

```python
SUPPORTS_CRITIC_STATE = "supports_critic_state"
SUPPORTS_SIM_STATE = "supports_sim_state"
SUPPORTS_TASK_TERMS = "supports_task_terms"
SUPPORTS_TERMINAL_STATE = "supports_terminal_state"
SUPPORTS_DIFFERENTIAL_ROLLOUT = "supports_differential_rollout"
SUPPORTS_DYNAMICS_SWITCH = "supports_dynamics_switch"
SUPPORTS_WARP_BACKEND = "supports_warp_backend"
```

### 6.4 `terms.py`

`drone_racing` 首版建议统一以下 task terms：

- `progress`
- `tracking_error`
- `gate_pass`
- `collision`
- `terminal`
- `control_effort`
- `control_smoothness`
- `angular_rate`
- `time_penalty`

环境负责产出这些标准化项，算法层负责按训练目标组装 loss。

### 6.5 `sim_state` schema

采用“公共字段 + 模型特有字段 + 动力学元信息”的结构。

公共字段建议至少包含：

- `position_w`
- `linear_velocity_w`
- `target_position_w`
- `last_action`
- `progress`
- `step_count`

完整四旋翼模型可额外提供：

- `quaternion_w`
- `angular_velocity_b`
- `motor_omega`
- `thrust_body`
- `torque_body`

点质量模型可额外提供：

- `heading`
- `acceleration_w`

动力学元信息建议至少包含：

- `model_name`
- `state_layout_version`
- `quat_convention`
- `tensor_backend`
- `write_mode`

约定如下：

- `quat_convention` 明确记录 `wxyz` 或 `xyzw`
- `tensor_backend` 明确记录 `torch` 或 `warp`
- `write_mode` 明确记录 `indexed` 或 `masked`

在 PhysX 路线中，首版默认输出 `torch` tensor；在 Newton / Warp 路线中，adapter 层负责处理 `wp.to_torch()` 或保持 warp-native 数据流。

## 7. 训练入口组织

### 7.1 RL baseline

RL baseline 在本设计中专指 IsaacLab 已集成的学习库，即 `refer/IsaacLab/source/isaaclab_rl/` 下已经支持的框架，例如 `rsl_rl`、`skrl`、`rl_games`、`sb3`。

脚本目录：

```text
scripts/reinforcement_learning/
├── rsl_rl/
├── skrl/
├── rl_games/
└── sb3/
```

运行链路：

```text
task id
-> hydra_task_config(...)
-> gym.make(task_id, cfg=env_cfg)
-> isaaclab_rl wrapper
-> integrated RL library
```

RL baseline 只消费：

- `observations[OBS_POLICY]`
- `observations[OBS_CRITIC]`，当对应集成算法支持 asymmetric actor-critic 时
- `reward`
- `terminated / truncated`

RL baseline 的 agent 配置保持 task-local，放在 `diffaero_lab.env/tasks/direct/drone_racing/agents/`，并通过 `gym.register(..., kwargs={"rsl_rl_cfg_entry_point": ..., "skrl_cfg_entry_point": ...})` 暴露给 IsaacLab 默认训练脚本。

### 7.2 Differential learning

脚本目录：

```text
scripts/differential/
├── train_apg.py
├── train_apg_stochastic.py
├── train_shac.py
├── train_sha2c.py
└── eval.py
```

运行链路：

```text
task id
-> hydra(task_cfg, algo_cfg)
-> gym.make(task_id, cfg=env_cfg)
-> differential env adapter
-> rollout / horizon unroll
-> task terms + sim_state + critic_state
-> loss assembly
-> backward
```

`diffaero_lab.algo/wrappers/` 负责：

- 检查 capability
- 读取 `observations[OBS_POLICY]` / `observations[OBS_CRITIC]` / `extras[EXTRA_TASK_TERMS]` / `extras[EXTRA_SIM_STATE]`
- 根据 `model_name` 做 flatten / unflatten
- 处理 `state_before_reset` 与 `terminal_state`

RL wrapper 约定如下：

- 对 IsaacLab 已支持的 RL 库，优先直接使用 `isaaclab_rl` 中现成 wrapper。
- 只有当某个 RL baseline 需要额外字段时，才在 `diffaero_lab.env/tasks/.../agents/` 或 `scripts/reinforcement_learning/` 中添加薄包装层。
- 因此 `task id -> gym.make -> isaaclab_rl wrapper -> RL library` 是首版默认路径。
- 按 IsaacLab wrapper 约定，learning-framework wrapper 必须放在 wrapper 链最后一层；differential wrapper 也沿用这一规则。

## 8. 配置组织

采用“任务配置树 + 算法配置树并列 + Hydra 运行时组合”。

环境侧：

- `drone_racing_env_cfg.py`
- `agents/` 下放 RL baseline 配置入口

建议最小配置树如下：

```text
env:
  task: drone_racing
  dynamics_model: quad
  sensor_mode: state
  num_envs: 4096
algo:
  name: apg
  rollout_horizon: 32
  optimizer:
    lr: 3e-4
runtime:
  headless: true
  device: cuda:0
```

算法侧：

```text
diffaero_lab.algo/configs/
├── apg/
├── apg_stochastic/
├── shac/
└── sha2c/
```

环境配置中建议显式加入：

- `dynamics_model: quad | pmd | pmc | simple`
- `sensor_mode: state | lidar | camera`

职责分工约定如下：

- Gym registry + task-local `agents/`：负责 IsaacLab 已集成 RL baseline 的发现与默认配置入口
- Hydra：负责 `env.*`、`agent.*`、`algo.*` 等运行时覆盖
- `diffaero_lab.algo/configs/`：主要服务 APG / APG_stochastic / SHAC / SHA2C 这类自定义 differential learning 配置树

## 9. 参考仓库迁移映射

### 9.0 与当前脚本目录的过渡

当前仓库已有 `scripts/skrl/`。迁移阶段采用以下处理：

1. 保留现有 `scripts/skrl/` 不动，作为历史脚本参考。
2. 新增 `scripts/reinforcement_learning/` 与 `scripts/differential/` 作为目标结构。
3. 当新的 RL 入口稳定后，再决定是否将 `scripts/skrl/` 合并、重定向或删除。

### 9.1 可直接复制的部分

| 来源 | 去向 |
|---|---|
| `refer/isaac_drone_racer/dynamics/allocation.py` | `diffaero_lab.uav/dynamics/allocation.py` |
| `refer/isaac_drone_racer/dynamics/motor.py` | `diffaero_lab.uav/dynamics/motor.py` |
| `refer/diffaero/dynamics/base_dynamics.py` | `diffaero_lab.uav/dynamics/base.py` |
| `refer/diffaero/dynamics/quadrotor.py` | `diffaero_lab.uav/dynamics/quadrotor.py` |
| `refer/diffaero/dynamics/pointmass.py` | 拆到 `pointmass_discrete.py` 与 `pointmass_continuous.py` |
| `refer/diffaero/algo/` 中四个 differential algorithms 的核心逻辑 | `diffaero_lab.algo/algorithms/` 与 `trainers/` |

### 9.2 复制后重构的部分

| 来源 | 处理方式 |
|---|---|
| `refer/isaac_drone_racer/tasks/drone_racer/` | 保留任务骨架，按 `DirectRLEnv` 与 `dynamics_bridge` 重组 |
| `refer/diffaero/env/` | 抽取 racing 任务语义、loss term、state 语义，拆到 `mdp/` 与 `state/` |
| `refer/diffaero/cfg/` | 转换成 Hydra 配置树 |
| `refer/diffaero/network/` | 重组到 `diffaero_lab.algo/models/` |
| `AgileFlight_MultiAgent` 的 wrapper 与 trainer 组织 | 吸收 contract 分层方式，不直接继承实现 |

### 9.3 优先评估 IsaacLab 现成多旋翼组件

在实现 `diffaero_lab.uav` 前，先评估 `isaaclab_contrib` 中已有的多旋翼能力是否可复用，包括：

- `Multirotor` / `MultirotorCfg`
- `ThrustAction` / `ThrustActionCfg`
- thruster 相关 actuator 与 data container

评估原则如下：

1. 能满足 `drone_racing` 首版环境与 RL baseline 需求的部分优先复用。
2. 无法覆盖 `diffaero` 所需 allocation、motor delay、多动力学切换或 differential learning 契约的部分，再在 `diffaero_lab.uav` 中补齐。
3. `diffaero_lab.uav` 作为增强层，优先建立在 IsaacLab 已有能力之上，减少纯重复实现。

## 10. 首版实现顺序

1. 建立四包骨架与脚本入口。
2. 迁入 `drone_racing` 任务骨架并完成 Gym 注册。
3. 迁入 `diffaero_lab.uav` 平台能力与多动力学模型。
4. 打通 RL baseline。
5. 打通 `APG` 最小可微闭环。
6. 扩展到 `APG_stochastic`、`SHAC`、`SHA2C`。
7. 扩展到 `pmd`、`pmc`、`simple` 等动力学后端。

建议的实现顺序为：

```text
PhysX + RL baseline -> PhysX + APG -> PhysX + APG_stochastic / SHAC / SHA2C -> Warp/Newton + APG
quad -> pmd -> pmc -> simple
```

## 11. 验收标准

当首版达到以下状态时，说明架构骨架已经成立：

1. `drone_racing` 能通过 Gym ID 创建。
2. RL baseline 能训练并 rollout。
3. `APG` 能完成可微 rollout 与 backward。
4. `sim_state` 能在 `quad` 与至少一种 point-mass 模型下工作。
5. `SHAC / SHA2C` 能读取 `critic_state` 并运行。
6. Hydra 能组合 `task + dynamics_model + algo`。

建议的任务 ID 规划为：

- `Isaac-Drone-Racing-Direct-v0`：PhysX / 默认路线
- `Isaac-Drone-Racing-Direct-Warp-v0`：Warp / Newton 实验路线

## 12. 风险与约束

### 12.1 需要控制的风险

1. IsaacLab 环境主循环与可微 dynamics 后端的职责边界混乱。
2. reward 与 differential loss 混在环境主类里。
3. `sim_state` 在多动力学模型下缺乏稳定 schema。
4. RL wrapper 与 differential wrapper 反向污染环境实现。
5. Newton / Warp 仍处于实验特性阶段，后端 API、四元数约定与数据类型转换存在持续变化。

### 12.2 设计约束

- 环境层只负责任务语义与标准化导出。
- 算法层负责 loss、unroll、buffer、critic 更新。
- `diffaero_lab.common` 保持极小。
- `diffaero_lab.uav` 承接多动力学无人机平台复用能力。
- Newton / Warp 路线按实验后端处理，首版优先完成 PhysX 路线，再接 Warp/Newton 注册项与 adapter。

## 13. 当前推荐的最终摘要

这是一个以 IsaacLab own-project 结构为基础、以 `drone_racing` 为首发任务、同时面向 RL baseline 与 Newton differential learning 的四层架构。环境、算法、契约、平台能力各自分层，多动力学模型通过 bridge 切换，训练入口通过 Hydra 组合，首版优先建立 `quad + APG + RL baseline` 的最小闭环，再向 `APG_stochastic / SHAC / SHA2C` 与 `pmd / pmc / simple` 扩展。
