# Design Document: Qwen3 RL Fine-Tuning Pipeline

## Overview

This document describes the technical design for an end-to-end Reinforcement Learning fine-tuning
pipeline for `Qwen/Qwen3-8B`. The pipeline uses the Tinker SDK and `tinker-cookbook` from
Thinking Machines Lab to run GRPO/CISPO RL training, then exports the resulting LoRA adapter,
merges it into a full HuggingFace model, converts it to a quantized GGUF file, and registers it
with Ollama for local inference.

The pipeline is invoked via a CLI (`python -m qwen3_rl_pipeline`) and is designed to be run
either end-to-end (`train` subcommand) or in stages (`export` subcommand). All state is
persisted to disk via JSON checkpoints so that interrupted runs can be resumed.

### Key Design Decisions

- **Three-layer architecture**: I/O (data loading, subprocess calls, file I/O), business logic
  (training loop, reward computation, advantage calculation), and data models (dataclasses) are
  kept strictly separate to enable unit testing of business logic without network or disk I/O.
- **Async-first training loop**: The Tinker API is async-native. The training loop uses
  `asyncio` throughout and is driven by a top-level `asyncio.run()` call from the CLI.
- **Dependency injection**: All Tinker clients are constructed once at startup and injected into
  downstream components. No component calls `ServiceClient()` independently.
- **Subprocess isolation**: GGUF conversion and Ollama registration are performed via
  `subprocess.run()` with captured stdout/stderr, keeping external tool failures clearly
  distinguishable from Python logic errors.

---

## Architecture

The pipeline is structured as a Python package `qwen3_rl_pipeline` with the following top-level
modules:

```
qwen3_rl_pipeline/
├── __main__.py          # CLI entry point (argparse, asyncio.run)
├── cli.py               # Subcommand handlers: train, export
├── config.py            # Dataclasses for all configuration
├── clients.py           # Tinker client factory and initialization
├── data/
│   ├── loader.py        # JSONL loading and validation
│   └── models.py        # DatasetRecord dataclass
├── training/
│   ├── loop.py          # Async RL training loop
│   ├── rewards.py       # RewardFunction protocol + built-ins
│   └── checkpointing.py # Checkpoint read/write
├── export/
│   ├── adapter.py       # LoRA download + HF merge
│   └── converter.py     # GGUF conversion + quantization
├── deployment/
│   └── ollama.py        # Modelfile generation + ollama create
└── exceptions.py        # Custom exception hierarchy
```

### Data Flow

```mermaid
flowchart TD
    A[JSONL Dataset] -->|loader.py| B[list[DatasetRecord]]
    B -->|training/loop.py| C[RLDataset / EnvGroupBuilders]
    C -->|Tinker sample_async| D[TrajectoryGroups]
    D -->|reward_fn| E[Scored Trajectories]
    E -->|compute_advantages| F[Advantages]
    F -->|forward_backward_async| G[Gradients]
    G -->|optim_step_async| H[Updated LoRA Weights]
    H -->|checkpointing.py| I[checkpoint_step_XXXXXX.json]
    H -->|adapter.py| J[LoRA .safetensors]
    J -->|weights.build_hf_model| K[Merged HF Model]
    K -->|converter.py| L[GGUF File]
    L -->|llama-quantize| M[Quantized GGUF Q4_K_M]
    M -->|ollama.py| N[Modelfile]
    N -->|ollama create| O[Ollama Model]
```

### Training Loop Sequence

```mermaid
sequenceDiagram
    participant CLI
    participant Loop as training/loop.py
    participant TC as TrainingClient
    participant SC as SamplingClient
    participant RF as RewardFunction
    participant CK as Checkpointer

    CLI->>Loop: run_training_loop(config, dataset, reward_fn, clients)
    loop for each epoch
        loop for each batch in RLDataset
            Loop->>TC: save_weights_and_get_sampling_client()
            TC-->>Loop: on_policy_sampling_client
            Loop->>SC: sample_async(prompt, num_samples, params)
            SC-->>Loop: TrajectoryGroups
            Loop->>RF: reward_fn(prompt, completion)
            RF-->>Loop: float rewards
            Loop->>Loop: compute_advantages(trajectory_groups, "cispo")
            Loop->>TC: forward_backward_async(datums, "cispo")
            Loop->>TC: optim_step_async(AdamParams)
            Loop->>CK: maybe_save(step, epoch, weights_path)
            Loop->>Logger: step, mean_reward, mean_advantage
        end
    end
    Loop-->>CLI: final weights path
```

---

## Components and Interfaces

### `config.py` — Configuration Dataclasses

All configuration is expressed as frozen dataclasses to make it easy to serialize, log, and
pass through the call stack without mutation.

```python
@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: str
    output_dir: str
    model_name: str
    epochs: int = 3
    steps_per_checkpoint: int = 100
    num_samples_per_group: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 32
    max_tokens: int = 512
    temperature: float = 1.0
    log_level: str = "INFO"

@dataclass(frozen=True)
class ExportConfig:
    adapter_path: str
    output_dir: str
    model_name: str
    gguf_output_path: str
    quantized_output_path: str
    quantization_type: str = "Q4_K_M"
    llama_cpp_dir: str = "/usr/local/bin"

@dataclass(frozen=True)
class ModelfileConfig:
    gguf_path: str
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096
    system_prompt: str | None = None
```

### `exceptions.py` — Custom Exception Hierarchy

```python
class PipelineError(Exception): ...
class DatasetParseError(PipelineError): ...
class DatasetValidationError(PipelineError): ...
class ExportError(PipelineError): ...
class ConversionError(PipelineError): ...
class DeploymentError(PipelineError): ...
```

### `data/loader.py` — Dataset Loader

**Interface:**

```python
def load_dataset(path: str) -> list[DatasetRecord]:
    """Load and validate a JSONL file into DatasetRecord objects.

    Args:
        path: Absolute or relative path to the JSONL file.

    Returns:
        List of validated DatasetRecord objects, one per line.

    Raises:
        DatasetParseError: If any line is not valid JSON.
        DatasetValidationError: If any record is missing 'prompt' or has an empty prompt.
    """
```

Parsing is line-by-line; each line is parsed with `json.loads`. Errors carry the 1-based line
number and the raw line content. The loader logs the total record count at INFO level after a
successful load.

### `clients.py` — Tinker Client Factory

**Interface:**

```python
@dataclass
class TinkerClients:
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient

def initialize_clients(base_model: str = "Qwen/Qwen3-8B", rank: int = 32) -> TinkerClients:
    """Construct and return all Tinker API clients.

    Reads TINKER_API_KEY from the environment. Raises EnvironmentError if absent.
    Raises and logs any SDK exception at ERROR level.
    """
```

The factory validates `TINKER_API_KEY` before making any network call. All three clients are
constructed here and nowhere else in the codebase.

### `training/rewards.py` — Reward Function Protocol

```python
from typing import Protocol

class RewardFunction(Protocol):
    def __call__(self, prompt: str, completion: str) -> float: ...

class ExactMatchReward:
    def __init__(self, reference_answer: str) -> None: ...
    def __call__(self, prompt: str, completion: str) -> float: ...

class FormatReward:
    def __init__(self, pattern: str) -> None: ...
    def __call__(self, prompt: str, completion: str) -> float: ...
```

`ExactMatchReward` returns `1.0` on exact string equality, `0.0` otherwise.
`FormatReward` compiles the pattern once at construction and returns `1.0` if
`re.search(pattern, completion)` is truthy, `0.0` otherwise.

Non-finite reward values are sanitized to `0.0` in the training loop (not in the reward
functions themselves), keeping reward functions pure.

### `training/loop.py` — RL Training Loop

**Interface:**

```python
async def run_training_loop(
    config: TrainingConfig,
    dataset: list[DatasetRecord],
    reward_fn: RewardFunction,
    clients: TinkerClients,
    checkpointer: Checkpointer,
) -> str:
    """Run the full GRPO/CISPO RL training loop.

    Returns:
        Path to the final adapter weights as returned by the Tinker API.
    """
```

The loop builds an `RLDataset` from the loaded records, iterates over epochs and batches,
and follows this per-step sequence:

1. `training_client.save_weights_and_get_sampling_client()` → on-policy sampler
2. `sampling_client.sample_async(prompt, num_samples=N, sampling_params=...)` per group
3. Score completions with `reward_fn`; replace non-finite values with `0.0`
4. `compute_advantages(trajectory_groups, loss_fn="cispo")`
5. `assemble_training_data(trajectory_groups, advantages)`
6. `training_client.forward_backward_async(datums, "cispo")`
7. `training_client.optim_step_async(AdamParams(learning_rate=...))`
8. Log `step`, `epoch`, `mean_reward`, `mean_advantage`
9. `checkpointer.maybe_save(step, epoch, weights_path)`

Exceptions from any async call are caught, logged at ERROR, and the step is skipped.
A progress log is emitted every 10 steps with elapsed time and ETA.

### `training/checkpointing.py` — Checkpointer

**Interface:**

```python
@dataclass
class Checkpoint:
    step: int
    epoch: int
    adapter_weights_path: str

class Checkpointer:
    def __init__(self, output_dir: str, save_every: int = 100) -> None: ...

    def maybe_save(self, step: int, epoch: int, weights_path: str) -> None:
        """Write checkpoint if step % save_every == 0. Logs WARNING on failure."""

    def load_latest(self) -> Checkpoint | None:
        """Scan output_dir for checkpoint_step_*.json, return highest-step one."""
```

Checkpoints are written as `checkpoint_step_{step:06d}.json` in `output_dir`. The file
contains the JSON-serialized `Checkpoint` dataclass. Write failures are caught and logged at
WARNING; they never halt training.

### `export/adapter.py` — LoRA Adapter Exporter

**Interface:**

```python
def download_adapter(tinker_path: str, output_dir: str) -> str:
    """Download LoRA adapter weights to output_dir.

    Returns:
        Path to the adapter directory.

    Raises:
        ExportError: If no .safetensors or .bin file is found after download.
    """

def merge_adapter(
    adapter_path: str,
    output_path: str,
    base_model: str = "Qwen/Qwen3-8B",
) -> str:
    """Merge LoRA adapter into base model weights.

    Returns:
        Path to the merged HF model directory.

    Raises:
        ExportError: If config.json is absent from the output directory.
    """
```

Both functions log at ERROR and re-raise on any exception.

### `export/converter.py` — GGUF Converter

**Interface:**

```python
def convert_to_gguf(
    hf_model_dir: str,
    output_path: str,
    llama_cpp_dir: str,
) -> str:
    """Run convert_hf_to_gguf.py as a subprocess.

    Raises:
        ConversionError: On non-zero exit code (includes captured stderr).
    """

def quantize_gguf(
    input_path: str,
    output_path: str,
    quantization_type: str = "Q4_K_M",
    llama_cpp_dir: str = "/usr/local/bin",
) -> str:
    """Run llama-quantize as a subprocess.

    Raises:
        ConversionError: On non-zero exit code or if output file is missing/empty.
    """
```

Both functions use `subprocess.run(..., capture_output=True, text=True)`. The quantized file
size is logged at INFO on success.

### `deployment/ollama.py` — Modelfile Generator and Deployer

**Interface:**

```python
def generate_modelfile(config: ModelfileConfig) -> str:
    """Render a Modelfile string from config.

    The FROM directive uses the absolute path to the GGUF file.
    PARAMETER directives are written for temperature, top_p, and num_ctx.
    SYSTEM directive is written only when system_prompt is not None.
    """

def write_modelfile(content: str, path: str) -> None:
    """Write the Modelfile string to disk."""

def register_model(model_name: str, modelfile_path: str) -> None:
    """Run `ollama create <model_name> -f <modelfile_path>` as a subprocess.

    Raises:
        DeploymentError: On non-zero exit code (includes captured stderr).
    """
```

`generate_modelfile` is a pure function (no I/O), which makes it straightforward to test the
round-trip property: `parse(generate(config)) == generate(config)`.

---

## Data Models

### `DatasetRecord`

```python
@dataclass(frozen=True)
class DatasetRecord:
    prompt: str
    reference_answer: str | None = None
```

Serialization: `{"prompt": "...", "reference_answer": "..."}` (reference_answer omitted if
`None`). The round-trip property holds: `deserialize(serialize(record)) == record`.

### `Checkpoint`

```python
@dataclass
class Checkpoint:
    step: int
    epoch: int
    adapter_weights_path: str
```

Serialized as JSON: `{"step": 42, "epoch": 1, "adapter_weights_path": "/path/to/adapter"}`.

### `ModelfileConfig`

```python
@dataclass(frozen=True)
class ModelfileConfig:
    gguf_path: str          # absolute path
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096
    system_prompt: str | None = None
```

The Modelfile format is deterministic given a `ModelfileConfig`: the same config always
produces the same Modelfile string, enabling the round-trip property.

### Modelfile Format

```
FROM /absolute/path/to/model.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM """<system_prompt>"""
```

The `SYSTEM` line is omitted when `system_prompt is None`. Parameters are written in a fixed
order (`temperature`, `top_p`, `num_ctx`) to ensure deterministic output.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions
of a system — essentially, a formal statement about what the system should do. Properties serve
as the bridge between human-readable specifications and machine-verifiable correctness
guarantees.*

### Property 1: JSONL parse-serialize-parse round-trip

*For any* non-empty list of `DatasetRecord` objects (with any non-empty prompt strings and any
optional reference_answer values), serializing each record to a JSON line, writing them to a
JSONL file, and parsing that file back SHALL produce a list of `DatasetRecord` objects equal to
the original — preserving every field, including optional `reference_answer`, without dropping
any records.

**Validates: Requirements 2.1, 2.2, 2.5, 2.6**

### Property 2: Whitespace-only prompts are rejected

*For any* string composed entirely of whitespace characters (spaces, tabs, newlines, or any
Unicode whitespace), attempting to load it as a `prompt` field SHALL raise a
`DatasetValidationError` and no partial dataset SHALL be returned.

**Validates: Requirements 2.4**

### Property 3: Non-finite rewards are sanitized to zero

*For any* non-finite float value (NaN, +∞, or −∞) produced by a reward function for any
(prompt, completion) pair, the `sanitize_reward` function SHALL return exactly `0.0`, and the
result SHALL be finite.

**Validates: Requirements 5.4**

### Property 4: ExactMatchReward is binary and correct

*For any* reference answer string and any completion string, `ExactMatchReward` SHALL return
exactly `1.0` when `completion == reference_answer` and exactly `0.0` otherwise — never any
other value, and never a non-finite value.

**Validates: Requirements 5.1, 5.2**

### Property 5: FormatReward is binary and correct

*For any* valid regex pattern string and any completion string, `FormatReward` SHALL return
exactly `1.0` when `re.search(pattern, completion)` is truthy and exactly `0.0` when it is
falsy — never any other value, and never a non-finite value.

**Validates: Requirements 5.1, 5.3**

### Property 6: Modelfile generation round-trip

*For any* valid `ModelfileConfig` (with any finite float parameters, any positive integer
`num_ctx`, any non-empty path strings, and any optional system prompt string), generating a
Modelfile string, parsing it back into a `ModelfileConfig`, and generating again SHALL produce
an identical Modelfile string — preserving all `PARAMETER` directives, the `FROM` path, and
the `SYSTEM` directive when present.

**Validates: Requirements 9.1, 9.2, 9.3, 9.7**

### Property 7: Checkpoint serialization round-trip

*For any* `Checkpoint` (with any non-negative step number, any non-negative epoch number, and
any non-empty weights path string), serializing it to a JSON file and deserializing it back
SHALL produce a `Checkpoint` equal to the original, and the filename SHALL match the pattern
`checkpoint_step_{step:06d}.json`.

**Validates: Requirements 6.2, 6.5**

---

## Error Handling

### Exception Hierarchy

All pipeline-specific exceptions inherit from `PipelineError` so callers can catch the entire
family with a single `except PipelineError` clause.

| Exception | Raised by | Condition |
|---|---|---|
| `EnvironmentError` | `clients.py` | `TINKER_API_KEY` absent or empty |
| `DatasetParseError` | `data/loader.py` | Line is not valid JSON |
| `DatasetValidationError` | `data/loader.py` | Missing or empty `prompt` field |
| `ExportError` | `export/adapter.py` | Missing adapter files or `config.json` |
| `ConversionError` | `export/converter.py` | Non-zero subprocess exit or missing output |
| `DeploymentError` | `deployment/ollama.py` | Non-zero `ollama create` exit |

### Error Propagation Strategy

- **Training step failures**: Caught inside the loop, logged at ERROR, step skipped. Training
  continues. This prevents a single bad batch from aborting a long run.
- **Checkpoint write failures**: Caught, logged at WARNING, training continues. The checkpoint
  directory is best-effort.
- **Export / conversion / deployment failures**: Re-raised after logging at ERROR. These are
  fatal — there is no meaningful way to continue without a valid artifact.
- **Client initialization failures**: Re-raised after logging at ERROR. The pipeline cannot
  proceed without valid clients.
- **CLI top-level**: Any unhandled exception is caught, logged at CRITICAL, and the process
  exits with code 1.

### Secret Safety

`TINKER_API_KEY` is read once in `clients.py` via `os.environ["TINKER_API_KEY"]`. It is never
stored in a dataclass, never passed as a function argument, and never appears in any log
message. The validation check logs only the variable name, not its value.

---

## Testing Strategy

### Test Layout

```
tests/
├── data/
│   ├── test_loader.py          # unit + property tests for JSONL loading
│   └── test_models.py          # property tests for DatasetRecord round-trip
├── training/
│   ├── test_rewards.py         # property tests for ExactMatchReward, FormatReward
│   ├── test_loop.py            # unit tests with mocked Tinker clients
│   └── test_checkpointing.py   # property tests for Checkpoint round-trip
├── export/
│   ├── test_adapter.py         # unit tests with mocked weights API
│   └── test_converter.py       # unit tests with mocked subprocess
├── deployment/
│   └── test_ollama.py          # property tests for Modelfile round-trip + unit tests
└── test_cli.py                 # CLI argument parsing and exit code tests
```

### Unit Tests

Unit tests cover:
- Specific error conditions: missing `prompt`, invalid JSON, non-zero subprocess exit codes
- Integration points: client injection, checkpoint resume logic
- CLI argument parsing: missing required args exit with code 2, `--help` exits with code 0
- Reward sanitization: NaN and infinity inputs produce `0.0` output

Tinker API calls are mocked using `unittest.mock.AsyncMock`. Subprocess calls are mocked using
`unittest.mock.patch("subprocess.run")`.

### Property-Based Tests (Hypothesis)

The project uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing.
Each property test runs a minimum of 100 iterations.

**Tag format**: Each property test is tagged with a comment:
`# Feature: qwen3-rl-finetuning-pipeline, Property N: <property_text>`

**Property 1 — JSONL parse-serialize-parse round-trip**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 1: JSONL parse-serialize-parse round-trip
@given(
    records=lists(
        builds(
            DatasetRecord,
            prompt=text(min_size=1).filter(str.strip),
            reference_answer=one_of(none(), text()),
        ),
        min_size=1,
    )
)
def test_jsonl_round_trip(records, tmp_path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({k: v for k, v in asdict(r).items() if v is not None})
            for r in records
        )
        + "\n"
    )
    assert load_dataset(str(jsonl)) == records
```

**Property 2 — Whitespace-only prompts are rejected**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 2: Whitespace-only prompts are rejected
@given(prompt=text(alphabet=characters(whitelist_categories=("Zs", "Cc")), min_size=1))
def test_whitespace_prompt_rejected(prompt, tmp_path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(json.dumps({"prompt": prompt}) + "\n")
    with pytest.raises(DatasetValidationError):
        load_dataset(str(jsonl))
```

**Property 3 — Non-finite rewards are sanitized to zero**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 3: Non-finite rewards are sanitized to zero
@given(
    bad_value=one_of(just(float("nan")), just(float("inf")), just(float("-inf")))
)
def test_nonfinite_reward_sanitized(bad_value):
    sanitized = sanitize_reward(bad_value)
    assert sanitized == 0.0
    assert math.isfinite(sanitized)
```

**Property 4 — ExactMatchReward is binary and correct**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 4: ExactMatchReward is binary and correct
@given(reference=text(), completion=text())
def test_exact_match_reward_binary(reference, completion):
    reward = ExactMatchReward(reference)(prompt="", completion=completion)
    assert reward in (0.0, 1.0)
    assert math.isfinite(reward)
    assert reward == (1.0 if completion == reference else 0.0)
```

**Property 5 — FormatReward is binary and correct**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 5: FormatReward is binary and correct
@given(pattern=from_regex(r"[a-z]+"), completion=text())
def test_format_reward_binary(pattern, completion):
    reward = FormatReward(pattern)(prompt="", completion=completion)
    assert reward in (0.0, 1.0)
    assert math.isfinite(reward)
    expected = 1.0 if re.search(pattern, completion) else 0.0
    assert reward == expected
```

**Property 6 — Modelfile generation round-trip**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 6: Modelfile generation round-trip
@given(
    gguf_path=text(min_size=1).filter(lambda s: s.strip()),
    model_name=text(min_size=1).filter(lambda s: s.strip()),
    temperature=floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    top_p=floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    num_ctx=integers(min_value=128, max_value=131072),
    system_prompt=one_of(none(), text()),
)
def test_modelfile_round_trip(
    gguf_path, model_name, temperature, top_p, num_ctx, system_prompt
):
    config = ModelfileConfig(
        gguf_path=gguf_path,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        system_prompt=system_prompt,
    )
    first = generate_modelfile(config)
    parsed_config = parse_modelfile(first)
    second = generate_modelfile(parsed_config)
    assert first == second
```

**Property 7 — Checkpoint serialization round-trip**
```python
# Feature: qwen3-rl-finetuning-pipeline, Property 7: Checkpoint serialization round-trip
@given(
    step=integers(min_value=0),
    epoch=integers(min_value=0),
    weights_path=text(min_size=1),
)
def test_checkpoint_round_trip(step, epoch, weights_path, tmp_path):
    ck = Checkpoint(step=step, epoch=epoch, adapter_weights_path=weights_path)
    filename = f"checkpoint_step_{step:06d}.json"
    path = tmp_path / filename
    path.write_text(json.dumps(asdict(ck)))
    loaded = Checkpoint(**json.loads(path.read_text()))
    assert loaded == ck
    assert path.name == filename
```

### Integration Tests

Integration tests (marked `@pytest.mark.integration`, skipped in CI unless
`TINKER_API_KEY` is set) cover:
- End-to-end `train` run on a 5-record JSONL fixture with 1 epoch and 1 step
- `export` subcommand against a pre-downloaded adapter fixture
- `ollama create` invocation (requires Ollama installed locally)
