# Implementation Plan: Qwen3 RL Fine-Tuning Pipeline

## Overview

Implement an end-to-end RL fine-tuning pipeline for `Qwen/Qwen3-8B` using the Tinker SDK.
The pipeline covers data loading, GRPO/CISPO training, LoRA export, GGUF conversion, and
Ollama deployment — all wired together through a CLI entry point. Tasks are ordered so that
foundational layers (exceptions, config, data models) are built before the components that
depend on them.

## Tasks

- [x] 1. Project scaffolding
  - Create the `qwen3_rl_pipeline/` package directory with `__init__.py` and all sub-package
    `__init__.py` files (`data/`, `training/`, `export/`, `deployment/`).
  - Create `tests/` directory tree mirroring the source structure with `__init__.py` files
    (`tests/data/`, `tests/training/`, `tests/export/`, `tests/deployment/`).
  - Create `requirements.txt` with exact-pinned runtime dependencies: `tinker-sdk`,
    `tinker-cookbook`, `hypothesis`, `pytest`, `pytest-asyncio`.
  - Create `requirements-dev.txt` with exact-pinned dev/test extras: `black`, `mypy`, `ruff`.
  - Create `pyproject.toml` declaring the package, entry point
    `qwen3-rl-pipeline = "qwen3_rl_pipeline.__main__:main"`, and tool config for Black and
    Ruff (88-char line limit).
  - Create `conftest.py` at the repo root registering the `integration` pytest mark and adding
    a skip condition when `TINKER_API_KEY` is absent.
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Custom exception hierarchy (`exceptions.py`)
  - [x] 2.1 Implement `exceptions.py`
    - Define `PipelineError(Exception)` as the base class.
    - Define `DatasetParseError`, `DatasetValidationError`, `ExportError`, `ConversionError`,
      and `DeploymentError` — all inheriting from `PipelineError`.
    - Add Google-style docstrings to every class.
    - _Requirements: 2.3, 2.4, 7.2, 7.4, 8.2, 8.4, 9.5_

  - [ ]* 2.2 Write unit tests for exception hierarchy
    - Verify each exception is a subclass of `PipelineError`.
    - Verify `except PipelineError` catches all derived exceptions.
    - _Requirements: 2.3, 2.4_

- [x] 3. Configuration dataclasses (`config.py`)
  - [x] 3.1 Implement `config.py`
    - Define frozen `TrainingConfig` dataclass with fields: `dataset_path`, `output_dir`,
      `model_name`, `epochs=3`, `steps_per_checkpoint=100`, `num_samples_per_group=4`,
      `learning_rate=1e-4`, `lora_rank=32`, `max_tokens=512`, `temperature=1.0`,
      `log_level="INFO"`.
    - Define frozen `ExportConfig` dataclass with fields: `adapter_path`, `output_dir`,
      `model_name`, `gguf_output_path`, `quantized_output_path`, `quantization_type="Q4_K_M"`,
      `llama_cpp_dir="/usr/local/bin"`.
    - Define frozen `ModelfileConfig` dataclass with fields: `gguf_path`, `model_name`,
      `temperature=0.7`, `top_p=0.9`, `num_ctx=4096`, `system_prompt: str | None = None`.
    - Add type hints and Google-style docstrings.
    - _Requirements: 6.1, 8.3, 9.2, 9.3, 10.1, 10.2_

  - [ ]* 3.2 Write unit tests for config dataclasses
    - Verify frozen dataclasses raise `FrozenInstanceError` on mutation.
    - Verify default values are applied correctly.
    - _Requirements: 6.1, 8.3_

- [x] 4. Data models (`data/models.py`)
  - [x] 4.1 Implement `DatasetRecord` dataclass
    - Define frozen `DatasetRecord` with `prompt: str` and
      `reference_answer: str | None = None`.
    - Implement `to_dict() -> dict[str, str]` that omits `reference_answer` when `None`.
    - Implement `from_dict(d: dict[str, object]) -> DatasetRecord` class method.
    - Add Google-style docstrings.
    - _Requirements: 2.1, 2.5_

  - [ ]* 4.2 Write property test for `DatasetRecord` round-trip (Property 1 — partial)
    - **Property 1: JSONL parse-serialize-parse round-trip**
    - **Validates: Requirements 2.1, 2.5**
    - Use `@given(builds(DatasetRecord, prompt=text(min_size=1).filter(str.strip), reference_answer=one_of(none(), text())))`.
    - Assert `DatasetRecord.from_dict(r.to_dict()) == r`.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 1: JSONL parse-serialize-parse round-trip`
    - _Requirements: 2.6_

- [x] 5. Dataset loader (`data/loader.py`) with validation
  - [x] 5.1 Implement `load_dataset(path: str) -> list[DatasetRecord]`
    - Parse the JSONL file line-by-line using `json.loads`.
    - Raise `DatasetParseError` (with 1-based line number and raw content) on invalid JSON.
    - Raise `DatasetValidationError` (with line number and field name) when `prompt` is absent
      or its stripped value is empty.
    - Populate `reference_answer` from the optional field when present.
    - Log total record count at INFO level via the `"qwen3_rl_pipeline"` logger.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 11.1, 11.2_

  - [ ]* 5.2 Write property test for JSONL round-trip (Property 1 — full)
    - **Property 1: JSONL parse-serialize-parse round-trip**
    - **Validates: Requirements 2.1, 2.2, 2.5, 2.6**
    - Use the `@given` strategy from the design document (lists of `DatasetRecord`, min_size=1).
    - Serialize records to a temp JSONL file, call `load_dataset`, assert equality.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 1: JSONL parse-serialize-parse round-trip`
    - _Requirements: 2.6_

  - [ ]* 5.3 Write property test for whitespace prompt rejection (Property 2)
    - **Property 2: Whitespace-only prompts are rejected**
    - **Validates: Requirements 2.4**
    - Use `@given(prompt=text(alphabet=characters(whitelist_categories=("Zs", "Cc")), min_size=1))`.
    - Write a single-line JSONL with that prompt, assert `DatasetValidationError` is raised.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 2: Whitespace-only prompts are rejected`
    - _Requirements: 2.4_

  - [ ]* 5.4 Write unit tests for loader error paths
    - Test invalid JSON line raises `DatasetParseError` with correct line number.
    - Test missing `prompt` key raises `DatasetValidationError`.
    - Test empty string `prompt` raises `DatasetValidationError`.
    - Test successful load returns correct record count and logs at INFO.
    - _Requirements: 2.2, 2.3, 2.4, 2.7_

- [x] 6. Checkpoint — Ensure data layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Tinker client factory (`clients.py`)
  - [x] 7.1 Implement `TinkerClients` dataclass and `initialize_clients()`
    - Define `TinkerClients` dataclass with `service_client`, `training_client`,
      `sampling_client` fields.
    - In `initialize_clients()`, read `TINKER_API_KEY` via `os.environ`; raise
      `EnvironmentError` (with the variable name in the message) if absent or empty — before
      any network call.
    - Call `ServiceClient()`, then `create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)`,
      then `create_sampling_client(base_model="Qwen/Qwen3-8B")`.
    - Catch and log any SDK exception at ERROR level, then re-raise.
    - Never log the value of `TINKER_API_KEY`.
    - _Requirements: 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5, 11.4_

  - [ ]* 7.2 Write unit tests for client factory
    - Test missing `TINKER_API_KEY` raises `EnvironmentError` before any SDK call.
    - Test empty `TINKER_API_KEY` raises `EnvironmentError`.
    - Test SDK exception is logged at ERROR and re-raised.
    - Mock `ServiceClient` and verify all three clients are constructed in order.
    - _Requirements: 1.4, 1.5, 3.4_

- [x] 8. Reward functions (`training/rewards.py`)
  - [x] 8.1 Implement `RewardFunction` protocol, `ExactMatchReward`, `FormatReward`, and `sanitize_reward`
    - Define `RewardFunction` as a `typing.Protocol` with `__call__(self, prompt: str, completion: str) -> float`.
    - Implement `ExactMatchReward(reference_answer: str)` returning `1.0` on exact equality,
      `0.0` otherwise.
    - Implement `FormatReward(pattern: str)` compiling the regex once at construction;
      returning `1.0` if `re.search` is truthy, `0.0` otherwise.
    - Implement `sanitize_reward(value: float) -> float` returning `0.0` for any non-finite
      input, the value unchanged otherwise.
    - Add type hints and Google-style docstrings to all public symbols.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 8.2 Write property test for non-finite reward sanitization (Property 3)
    - **Property 3: Non-finite rewards are sanitized to zero**
    - **Validates: Requirements 5.4**
    - Use `@given(one_of(just(float("nan")), just(float("inf")), just(float("-inf"))))`.
    - Assert `sanitize_reward(bad_value) == 0.0` and `math.isfinite(result)`.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 3: Non-finite rewards are sanitized to zero`
    - _Requirements: 5.4_

  - [ ]* 8.3 Write property test for `ExactMatchReward` binary correctness (Property 4)
    - **Property 4: ExactMatchReward is binary and correct**
    - **Validates: Requirements 5.1, 5.2**
    - Use `@given(reference=text(), completion=text())`.
    - Assert result is in `{0.0, 1.0}`, is finite, and equals `1.0` iff `completion == reference`.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 4: ExactMatchReward is binary and correct`
    - _Requirements: 5.1, 5.2_

  - [ ]* 8.4 Write property test for `FormatReward` binary correctness (Property 5)
    - **Property 5: FormatReward is binary and correct**
    - **Validates: Requirements 5.1, 5.3**
    - Use `@given(pattern=from_regex(r"[a-z]+"), completion=text())`.
    - Assert result is in `{0.0, 1.0}`, is finite, and matches `re.search` truth value.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 5: FormatReward is binary and correct`
    - _Requirements: 5.1, 5.3_

  - [ ]* 8.5 Write unit tests for reward functions
    - Test `sanitize_reward` passes finite values through unchanged.
    - Test `ExactMatchReward` with identical strings returns `1.0`.
    - Test `ExactMatchReward` with differing strings returns `0.0`.
    - Test `FormatReward` with a matching completion returns `1.0`.
    - Test `FormatReward` with a non-matching completion returns `0.0`.
    - _Requirements: 5.2, 5.3, 5.4_

- [x] 9. Checkpointer (`training/checkpointing.py`)
  - [x] 9.1 Implement `Checkpoint` dataclass and `Checkpointer` class
    - Define `Checkpoint` dataclass with `step: int`, `epoch: int`,
      `adapter_weights_path: str`.
    - Implement `Checkpointer.__init__(output_dir: str, save_every: int = 100)`.
    - Implement `maybe_save(step, epoch, weights_path)`: write
      `checkpoint_step_{step:06d}.json` when `step % save_every == 0`; catch write errors,
      log at WARNING, and continue.
    - Implement `load_latest() -> Checkpoint | None`: glob
      `checkpoint_step_*.json`, parse each, return the one with the highest step number.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 9.2 Write property test for checkpoint round-trip (Property 7)
    - **Property 7: Checkpoint serialization round-trip**
    - **Validates: Requirements 6.2, 6.5**
    - Use `@given(step=integers(min_value=0), epoch=integers(min_value=0), weights_path=text(min_size=1))`.
    - Serialize to `tmp_path / f"checkpoint_step_{step:06d}.json"`, deserialize, assert equality and filename pattern.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 7: Checkpoint serialization round-trip`
    - _Requirements: 6.2, 6.5_

  - [ ]* 9.3 Write unit tests for checkpointer
    - Test `maybe_save` writes a file at the correct path when `step % save_every == 0`.
    - Test `maybe_save` does not write when `step % save_every != 0`.
    - Test `maybe_save` logs WARNING and does not raise when the write fails.
    - Test `load_latest` returns `None` when the directory is empty.
    - Test `load_latest` returns the checkpoint with the highest step when multiple exist.
    - _Requirements: 6.1, 6.3, 6.4_

- [x] 10. Checkpoint — Ensure reward and checkpointing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. RL training loop (`training/loop.py`)
  - [x] 11.1 Implement `run_training_loop()`
    - Accept `config: TrainingConfig`, `dataset: list[DatasetRecord]`,
      `reward_fn: RewardFunction`, `clients: TinkerClients`, `checkpointer: Checkpointer`.
    - Build `RLDataset` / `EnvGroupBuilder` instances from the dataset records.
    - Iterate over `config.epochs`; within each epoch iterate over batches.
    - Per step: call `save_weights_and_get_sampling_client()`, `sample_async()` per group,
      score with `reward_fn`, call `sanitize_reward()` on each result, call
      `compute_advantages(..., loss_fn="cispo")`, assemble training data,
      `forward_backward_async(..., "cispo")`, `optim_step_async(AdamParams(...))`.
    - Log `step`, `epoch`, `mean_reward`, `mean_advantage` at INFO after each step.
    - Emit a progress log every 10 steps with elapsed time and ETA.
    - Catch exceptions from any async call, log at ERROR, skip the step, continue.
    - Call `checkpointer.maybe_save(step, epoch, weights_path)` after each step.
    - Return the final adapter weights path.
    - _Requirements: 4.1–4.10, 5.4, 6.1, 11.1, 11.2, 11.3, 11.5_

  - [ ]* 11.2 Write unit tests for training loop with mocked clients
    - Mock `TinkerClients` using `unittest.mock.AsyncMock` for all async methods.
    - Test that `save_weights_and_get_sampling_client` is called once per step.
    - Test that `reward_fn` is called for each completion and non-finite rewards are replaced
      with `0.0`.
    - Test that a step-level exception is caught, logged at ERROR, and training continues.
    - Test that `checkpointer.maybe_save` is called with the correct step and epoch.
    - Test that the loop returns the final weights path.
    - _Requirements: 4.1, 4.2, 4.4, 4.10, 5.4_

- [x] 12. LoRA adapter export (`export/adapter.py`)
  - [x] 12.1 Implement `download_adapter()` and `merge_adapter()`
    - `download_adapter(tinker_path, output_dir)`: call `weights.download(...)`, verify at
      least one `.safetensors` or `.bin` file exists in the output directory, raise
      `ExportError` if not; log at ERROR and re-raise on any exception.
    - `merge_adapter(adapter_path, output_path, base_model)`: call
      `weights.build_hf_model(...)`, verify `config.json` exists in the output directory,
      raise `ExportError` if not; log at ERROR and re-raise on any exception.
    - Log stage start/end at INFO via the `"qwen3_rl_pipeline"` logger.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 11.2_

  - [ ]* 12.2 Write unit tests for adapter export with mocked weights API
    - Mock `weights.download` and `weights.build_hf_model`.
    - Test `download_adapter` raises `ExportError` when no `.safetensors`/`.bin` file exists.
    - Test `download_adapter` succeeds when a `.safetensors` file is present.
    - Test `merge_adapter` raises `ExportError` when `config.json` is absent.
    - Test `merge_adapter` succeeds when `config.json` is present.
    - Test exceptions are logged at ERROR and re-raised.
    - _Requirements: 7.2, 7.4, 7.5_

- [x] 13. GGUF conversion and quantization (`export/converter.py`)
  - [x] 13.1 Implement `convert_to_gguf()` and `quantize_gguf()`
    - `convert_to_gguf(hf_model_dir, output_path, llama_cpp_dir)`: invoke
      `convert_hf_to_gguf.py` via `subprocess.run(..., capture_output=True, text=True)`;
      raise `ConversionError` with captured stderr on non-zero exit code.
    - `quantize_gguf(input_path, output_path, quantization_type, llama_cpp_dir)`: invoke
      `llama-quantize` via `subprocess.run`; raise `ConversionError` on non-zero exit code;
      verify output file exists and has size > 0, raise `ConversionError` if not; log file
      size at INFO on success.
    - Log stage start/end at INFO.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 11.2_

  - [ ]* 13.2 Write unit tests for converter with mocked subprocess
    - Patch `subprocess.run` to simulate zero and non-zero exit codes.
    - Test `convert_to_gguf` raises `ConversionError` with stderr on non-zero exit.
    - Test `quantize_gguf` raises `ConversionError` on non-zero exit.
    - Test `quantize_gguf` raises `ConversionError` when output file is missing.
    - Test `quantize_gguf` raises `ConversionError` when output file is empty (0 bytes).
    - Test successful quantization logs file size at INFO.
    - _Requirements: 8.2, 8.4, 8.5, 8.6_

- [x] 14. Checkpoint — Ensure export layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Ollama Modelfile generation and deployment (`deployment/ollama.py`)
  - [x] 15.1 Implement `generate_modelfile()`, `write_modelfile()`, `register_model()`, and `parse_modelfile()`
    - `generate_modelfile(config: ModelfileConfig) -> str`: pure function; write `FROM`
      directive with absolute GGUF path; write `PARAMETER temperature`, `PARAMETER top_p`,
      `PARAMETER num_ctx` in fixed order; write `SYSTEM """..."""` only when
      `system_prompt is not None`.
    - `parse_modelfile(content: str) -> ModelfileConfig`: parse a Modelfile string back into
      a `ModelfileConfig` (needed for the round-trip property test).
    - `write_modelfile(content: str, path: str) -> None`: write string to disk.
    - `register_model(model_name: str, modelfile_path: str) -> None`: invoke
      `ollama create <model_name> -f <modelfile_path>` via `subprocess.run`; raise
      `DeploymentError` with captured stderr on non-zero exit; log model name and Modelfile
      path at INFO on success.
    - Log stage start/end at INFO.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 11.2_

  - [x]* 15.2 Write property test for Modelfile round-trip (Property 6)
    - **Property 6: Modelfile generation round-trip**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.7**
    - Use the `@given` strategy from the design document (finite floats, positive int `num_ctx`,
      optional system prompt).
    - Assert `generate_modelfile(parse_modelfile(generate_modelfile(config))) == generate_modelfile(config)`.
    - Tag: `# Feature: qwen3-rl-finetuning-pipeline, Property 6: Modelfile generation round-trip`
    - _Requirements: 9.7_

  - [x]* 15.3 Write unit tests for Ollama deployment
    - Test `generate_modelfile` includes `FROM`, all three `PARAMETER` lines, and `SYSTEM`
      when `system_prompt` is set.
    - Test `generate_modelfile` omits `SYSTEM` when `system_prompt` is `None`.
    - Test `register_model` raises `DeploymentError` with stderr on non-zero exit.
    - Test `register_model` logs model name and Modelfile path at INFO on success.
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6_

- [x] 16. CLI entry point (`__main__.py` and `cli.py`)
  - [x] 16.1 Implement `__main__.py`
    - Call `asyncio.run(main())` guarded by `if __name__ == "__main__"`.
    - Catch any unhandled exception at the top level, log at CRITICAL via the
      `"qwen3_rl_pipeline"` logger, and `sys.exit(1)`.
    - _Requirements: 10.7_

  - [x] 16.2 Implement `cli.py` with `train` and `export` subcommands
    - Configure `argparse` with a top-level `--log-level` argument defaulting to `INFO`.
    - `train` subcommand: `--dataset` (required), `--epochs`, `--steps-per-checkpoint`,
      `--output-dir` (required), `--model-name` (required).
    - `export` subcommand: `--adapter-path` (required), `--output-dir` (required),
      `--model-name` (required).
    - Configure structured logging to stdout at the requested level using the
      `"qwen3_rl_pipeline"` root logger.
    - `train` handler: build `TrainingConfig`, call `initialize_clients()`, call
      `load_dataset()`, instantiate `Checkpointer`, call `run_training_loop()`, then
      call `download_adapter()`, `merge_adapter()`, `convert_to_gguf()`, `quantize_gguf()`,
      `generate_modelfile()`, `write_modelfile()`, `register_model()`.
    - `export` handler: build `ExportConfig`, call `download_adapter()`, `merge_adapter()`,
      `convert_to_gguf()`, `quantize_gguf()`, `generate_modelfile()`, `write_modelfile()`,
      `register_model()`.
    - Exit with code 0 on success.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 11.1, 11.2_

  - [ ]* 16.3 Write unit tests for CLI argument parsing and exit codes
    - Test `--help` on `train` and `export` exits with code 0.
    - Test missing required argument exits with code 2.
    - Test `--log-level DEBUG` sets the logger level to DEBUG.
    - Test successful `train` invocation (all components mocked) exits with code 0.
    - Test unhandled exception propagation logs at CRITICAL and exits with code 1.
    - _Requirements: 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 17. Integration test stubs (`tests/test_integration.py`)
  - [x] 17.1 Write integration test stubs
    - Mark all tests with `@pytest.mark.integration`.
    - Add a module-level `pytestmark` that skips the entire module when `TINKER_API_KEY` is
      not set (using the `conftest.py` skip condition).
    - Stub: end-to-end `train` run on a 5-record JSONL fixture with 1 epoch and 1 step.
    - Stub: `export` subcommand against a pre-downloaded adapter fixture.
    - Stub: `ollama create` invocation (requires Ollama installed locally).
    - Each stub should have a clear docstring describing what it will test when implemented.
    - _Requirements: 1.3, 3.1, 4.1, 7.1, 8.1, 9.4_

- [x] 18. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. README (`README.md`)
  - [x] 19.1 Write `README.md` with setup instructions and usage examples
    - Document prerequisites: Python 3.11+, `llama.cpp` binaries, Ollama.
    - Document environment setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
    - Document `TINKER_API_KEY` environment variable requirement.
    - Provide a `train` subcommand usage example with all flags.
    - Provide an `export` subcommand usage example with all flags.
    - Document how to run unit tests (`pytest tests/ -m "not integration"`) and integration
      tests (`TINKER_API_KEY=... pytest tests/ -m integration`).
    - _Requirements: 1.3, 10.1, 10.2_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP.
- Each task references specific requirements for traceability.
- Checkpoints at tasks 6, 10, 14, and 18 ensure incremental validation.
- Property tests (Properties 1–7) validate universal correctness guarantees defined in the
  design document; unit tests cover specific examples and error paths.
- All async Tinker API calls must be mocked with `unittest.mock.AsyncMock` in unit tests.
- Never log or expose the value of `TINKER_API_KEY` at any log level.
