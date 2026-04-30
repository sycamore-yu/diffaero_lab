# diffaero_lab Extension Source

**Part of:** `../AGENTS.md` (root)

## OVERVIEW

Omniverse extension package. Double-nested structure: `source/diffaero_lab/diffaero_lab/`. Installed as `diffaero_lab` via `pip install -e source/diffaero_lab`.
Current registered Task Scene modules are explicit and production-only.

## STRUCTURE

```
source/diffaero_lab/
├── config/extension.toml     # Omniverse extension metadata
├── setup.py                  # Package entry point
├── docs/CHANGELOG.rst       # Per-extension changelog
└── diffaero_lab/            # Main package (NESTED)
    ├── __init__.py           # Task registration
    ├── algo/
    ├── common/
    └── tasks/
        ├── __init__.py
        └── direct/
            └── drone_racing/
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add direct Task Scene | `tasks/direct/<task_name>/` |
| Route and capability contract | `common/sim_contract.py`, `common/rollout_route.py` |
| UAV Platform assets and route registry | `uav/assets/`, `uav/route_registry.py` |
| Gym registration | `diffaero_lab/__init__.py` |
| Differential Algorithm wrappers and trainers | `algo/wrappers/`, `algo/trainers/` |

## CONVENTIONS

- Follow Isaac Lab task structure: `*_env.py` + `*_env_cfg.py` pairs
- RL configs in `agents/` subdir (skrl yaml files)
- Runtime route selection goes through `uav/route_registry.py`
- Shared simulator metadata and rollout capability live in `common/`

## ANTI-PATTERNS

- DO NOT set `agent_spec` directly — use proper Isaac Lab patterns
- DO NOT change extension loading order
- Environment must always create one simulation context
- DO NOT close env when async data generator tasks may still be running
