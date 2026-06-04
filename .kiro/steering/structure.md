# Project Structure

```
qwen3_rl_pipeline/        # Main package
├── __main__.py           # Entry point: runs run_cli() via asyncio
├── cli.py                # Argument parsing + async subcommand handlers (train, export)
├── config.py             # Frozen dataclasses: TrainingConfig, ExportConfig, ModelfileConfig
├── clients.py            # Tinker client factory (initialize_clients_async)
├── exceptions.py         # Custom exception hierarchy rooted at PipelineError
├── data/
│   ├── loader.py         # JSONL loading and validation → List[DatasetRecord]
│   └── models.py         # DatasetRecord frozen dataclass
├── training/
│   ├── loop.py           # Async GRPO/CISPO RL training loop
│   ├── rewards.py        # RewardFunction protocol + ExactMatchReward, FormatReward, sanitize_reward
│   └── checkpointing.py  # Checkpointer: read/write checkpoint state
├── export/
│   ├── adapter.py        # download_adapter + merge_adapter (LoRA → HF merged model)
│   └── converter.py      # convert_to_gguf + quantize_gguf (llama.cpp subprocesses)
└── deployment/
    └── ollama.py         # generate_modelfile, write_modelfile, register_model

tests/                    # Mirrors source structure exactly
├── conftest.py           # Root fixtures + integration marker auto-skip logic
├── data/
├── training/
├── export/
├── deployment/
├── test_cli.py
├── test_clients.py
└── test_integration.py   # End-to-end tests marked @pytest.mark.integration
```

## Architecture Layers

The codebase is split into three decoupled layers:

1. **I/O** — `cli.py`, `data/loader.py`, `export/adapter.py`, `export/converter.py`, `deployment/ollama.py`
2. **Business logic** — `training/loop.py`, `training/rewards.py`, `training/checkpointing.py`
3. **Data models** — `config.py`, `data/models.py`, `exceptions.py`

## Key Conventions

- **Frozen dataclasses** for all config and data models — immutable after construction.
- **Protocol-based interfaces** (e.g. `RewardFunction`) — no inheritance required, duck-typed.
- **Custom exceptions** all inherit from `PipelineError` — catch the base to handle any pipeline error.
- **Dependency injection** — clients, reward functions, and checkpointers are passed explicitly; no global state.
- **Google-style docstrings** on all public modules, classes, and functions.
- **Type hints** on every function signature including return types; `mypy --strict` must pass.
- **`from __future__ import annotations`** at the top of every module.
- Tests mirror the source tree; use `hypothesis` for property-based tests on parsing/transformation logic.
- Integration tests are marked `@pytest.mark.integration` and auto-skipped when `TINKER_API_KEY` is absent.
