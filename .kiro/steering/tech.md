# Tech Stack

## Language & Runtime
- Python 3.11+ (tested on 3.11 and 3.14)
- Async-first: training loop and CLI handlers are `async`; `pytest-asyncio` with `asyncio_mode = "auto"`

## Key Libraries
| Library | Purpose |
|---|---|
| `tinker` / `tinker-cookbook` | Tinker SDK for remote RL training (GRPO/CISPO) |
| `hypothesis` | Property-based testing for data-transformation and parsing logic |
| `pytest` + `pytest-asyncio` | Test runner; async test support |
| `black` | Code formatter (88-char line length) |
| `ruff` | Linter (E, F, W, I, UP rule sets) |
| `mypy` | Static type checker (strict mode) |

## Build System
- **setuptools** with `pyproject.toml`; package name `qwen3-rl-pipeline`
- Entry point script: `qwen3-rl-pipeline` → `qwen3_rl_pipeline.__main__:main`

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt       # runtime deps
pip install -r requirements-dev.txt   # black, ruff, mypy
export TINKER_API_KEY="your-key"
```

## Common Commands

### Run the pipeline
```bash
python -m qwen3_rl_pipeline train --dataset data.jsonl --output-dir ./output --model-name my-model
python -m qwen3_rl_pipeline export --adapter-path tinker://... --output-dir ./output --model-name my-model
```

### Testing
```bash
# Unit tests only (no API key needed)
pytest tests/ -m "not integration"

# Integration tests (requires TINKER_API_KEY + running Ollama)
TINKER_API_KEY=your-key pytest tests/ -m integration

# All tests
pytest tests/
```

### Linting & formatting
```bash
black qwen3_rl_pipeline/ tests/
ruff check qwen3_rl_pipeline/ tests/
mypy qwen3_rl_pipeline/
```

## Dependency Pinning
All dependencies use exact version pins. Update `requirements.txt` or `requirements-dev.txt` whenever deps change — never use open ranges.
