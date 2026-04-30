# diffaero_lab Extension Source

**Part of:** `../AGENTS.md` (root)

## OVERVIEW

Omniverse extension package. Double-nested structure: `source/diffaero_lab/diffaero_lab/`. Installed as `diffaero_lab` via `pip install -e source/diffaero_lab`.

## STRUCTURE

```
source/diffaero_lab/
├── config/extension.toml     # Omniverse extension metadata
├── setup.py                  # Package entry point
├── docs/CHANGELOG.rst       # Per-extension changelog
└── diffaero_lab/            # Main package (NESTED)
    ├── __init__.py           # Gym env + UI extension registration
    ├── ui_extension_example.py
    └── tasks/
        ├── __init__.py
        ├── direct/           # Direct RL variant
        │   ├── diffaero_lab/           # Single-agent
        │   └── diffaero_lab_marl/     # Multi-agent (MARL)
        └── manager_based/   # Manager-based variant
            └── diffaero_lab/
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add task (single-agent) | `tasks/direct/diffaero_lab/` |
| Add task (multi-agent) | `tasks/direct/diffaero_lab_marl/` |
| Add manager-based task | `tasks/manager_based/diffaero_lab/` |
| Gym registration | `diffaero_lab/__init__.py` |
| UI extension | `ui_extension_example.py` |
| MDP rewards (manager-based) | `tasks/manager_based/diffaero_lab/mdp/rewards.py` |

## CONVENTIONS

- Follow Isaac Lab task structure: `*_env.py` + `*_env_cfg.py` pairs
- RL configs in `agents/` subdir (skrl yaml files)
- Manager-based MDP rewards go in `mdp/rewards.py`
- Task names must match search pattern in `scripts/list_envs.py` (`Template-*`)

## ANTI-PATTERNS

- DO NOT set `agent_spec` directly — use proper Isaac Lab patterns
- DO NOT change extension loading order
- Environment must always create one simulation context
- DO NOT close env when async data generator tasks may still be running
