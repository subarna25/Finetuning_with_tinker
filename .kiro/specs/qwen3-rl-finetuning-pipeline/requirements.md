# Requirements Document

## Introduction

This feature implements a Reinforcement Learning (RL) fine-tuning pipeline for the Qwen/Qwen3-8B
language model using the Tinker API from Thinking Machines Lab. The pipeline covers the full
lifecycle: training data preparation, RL fine-tuning via GRPO/CISPO, LoRA adapter export, GGUF
conversion, and local deployment through Ollama. The result is a locally runnable, fine-tuned
model that can be served via `ollama run`.

## Glossary

- **Pipeline**: The end-to-end system that takes raw training data and produces a locally
  deployable Ollama model.
- **Tinker_Client**: The `ServiceClient` instance from the `tinker` SDK that authenticates with
  the Tinker API using `TINKER_API_KEY`.
- **Training_Client**: The LoRA training client created via
  `service_client.create_lora_training_client()`.
- **Sampling_Client**: The rollout client created via `service_client.create_sampling_client()`
  or `training_client.save_weights_and_get_sampling_client()`.
- **Env**: A stateful `ProblemEnv` instance from `tinker-cookbook` representing one RL episode.
- **EnvGroup**: A group of `Env` instances built by `EnvGroupBuilder`, used for GRPO reward
  centering.
- **RLDataset**: A batch of `EnvGroupBuilder` instances assembled by `RLDatasetBuilder`.
- **Trajectory**: The sequence of (prompt, completion, reward) tuples collected during rollout
  for a single `Env`.
- **Advantage**: The normalized reward signal computed by `compute_advantages()` over a group of
  trajectories.
- **LoRA_Adapter**: The low-rank weight delta produced by the Training_Client and downloaded
  after training.
- **Merged_Model**: The full HF-format model produced by merging the base Qwen3-8B weights with
  the LoRA_Adapter via `weights.build_hf_model()`.
- **GGUF_Model**: The quantized model file produced by converting the Merged_Model using
  `llama.cpp`'s `convert_hf_to_gguf.py` and `llama-quantize`.
- **Modelfile**: The Ollama configuration file that references the GGUF_Model and sets inference
  parameters.
- **Reward_Function**: A user-supplied callable `(prompt: str, completion: str) -> float` that
  scores each rollout.
- **Dataset_Record**: A single training example with at minimum a `prompt` field and optionally
  a `reference_answer` field.
- **Checkpoint**: A saved snapshot of training state (step number, optimizer state, adapter
  weights path) written to disk.

---

## Requirements

### Requirement 1: Environment and Dependency Setup

**User Story:** As a developer, I want a reproducible Python environment with all required
dependencies pinned, so that the pipeline runs consistently across machines.

#### Acceptance Criteria

1. THE Pipeline SHALL declare all runtime dependencies in a `requirements.txt` file with exact
   version pins.
2. THE Pipeline SHALL declare all development and test dependencies in a `requirements-dev.txt`
   file with exact version pins.
3. WHEN a developer runs `python -m venv .venv && source .venv/bin/activate && pip install -r
   requirements.txt`, THE Pipeline SHALL install without errors.
4. THE Pipeline SHALL read the `TINKER_API_KEY` exclusively from the environment variable
   `TINKER_API_KEY` and SHALL NOT accept it as a command-line argument or hard-coded value.
5. IF `TINKER_API_KEY` is absent or empty at startup, THEN THE Pipeline SHALL raise a
   `EnvironmentError` with a message identifying the missing variable before making any network
   call.

---

### Requirement 2: Training Data Preparation

**User Story:** As an ML engineer, I want to load and validate training data from a structured
file, so that the RL training loop receives well-formed prompts and reference answers.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL accept a file path to a JSONL file where each line is a valid
   JSON object containing at minimum a `"prompt"` key with a non-empty string value.
2. WHEN a JSONL file is provided, THE Dataset_Loader SHALL parse every line into a
   `Dataset_Record` and return the full list without dropping records.
3. IF a line in the JSONL file is not valid JSON, THEN THE Dataset_Loader SHALL raise a
   `DatasetParseError` identifying the line number and the raw content of the offending line.
4. IF a parsed JSON object is missing the `"prompt"` key or its value is an empty string, THEN
   THE Dataset_Loader SHALL raise a `DatasetValidationError` identifying the line number and the
   field that failed validation.
5. THE Dataset_Loader SHALL support an optional `"reference_answer"` field; WHEN present, THE
   Dataset_Loader SHALL include it in the `Dataset_Record` unchanged.
6. FOR ALL valid JSONL files, parsing then serializing then parsing SHALL produce a list of
   `Dataset_Record` objects equal to the original list (round-trip property).
7. THE Dataset_Loader SHALL report the total number of loaded records to the logger at INFO
   level after a successful load.

---

### Requirement 3: Tinker Client Initialization

**User Story:** As an ML engineer, I want the pipeline to initialize Tinker API clients
correctly, so that training and sampling operations can proceed without authentication errors.

#### Acceptance Criteria

1. WHEN the pipeline starts, THE Tinker_Client SHALL be constructed by calling `ServiceClient()`
   with no arguments, relying on the `TINKER_API_KEY` environment variable.
2. WHEN `Tinker_Client` is initialized, THE Pipeline SHALL create a `Training_Client` by calling
   `service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)`.
3. WHEN `Tinker_Client` is initialized, THE Pipeline SHALL create an initial `Sampling_Client`
   by calling `service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")`.
4. IF any client initialization call raises an exception, THEN THE Pipeline SHALL log the
   exception at ERROR level and re-raise it, halting execution.
5. THE Pipeline SHALL construct all clients once at startup and inject them into downstream
   components — no component SHALL call `ServiceClient()` independently.

---

### Requirement 4: RL Training Loop

**User Story:** As an ML engineer, I want to run a GRPO/CISPO reinforcement learning training
loop over my dataset, so that the model learns to produce higher-reward completions.

#### Acceptance Criteria

1. THE Training_Loop SHALL accept a `RLDataset`, a `Reward_Function`, a `Training_Client`, and
   a `Sampling_Client` as explicit parameters.
2. WHEN a training step begins, THE Training_Loop SHALL call
   `training_client.save_weights_and_get_sampling_client()` to obtain an on-policy
   `Sampling_Client` for that step.
3. WHEN rollouts are collected, THE Training_Loop SHALL call
   `sampling_client.sample_async(prompt, num_samples=N, sampling_params=...)` for each
   `EnvGroup` in the current batch.
4. WHEN rollouts are complete, THE Training_Loop SHALL score each completion using the
   `Reward_Function` and attach the scalar reward to the corresponding `Trajectory`.
5. WHEN rewards are attached, THE Training_Loop SHALL call `compute_advantages(trajectory_groups,
   loss_fn="cispo")` to produce normalized `Advantage` values.
6. WHEN advantages are computed, THE Training_Loop SHALL call
   `training_client.forward_backward_async(data, "cispo")` with the assembled training data.
7. WHEN the backward pass completes, THE Training_Loop SHALL call
   `training_client.optim_step_async(AdamParams(learning_rate=...))` to update weights.
8. THE Training_Loop SHALL log the mean reward and mean advantage for each step at INFO level.
9. THE Training_Loop SHALL complete one full pass over the `RLDataset` per epoch and SHALL
   support a configurable number of epochs.
10. IF any async call raises an exception during a training step, THEN THE Training_Loop SHALL
    log the exception at ERROR level, skip that step, and continue with the next step.

---

### Requirement 5: Reward Function Interface

**User Story:** As an ML engineer, I want a well-defined reward function interface, so that I
can plug in task-specific scoring logic without modifying the training loop.

#### Acceptance Criteria

1. THE Reward_Function SHALL conform to the signature
   `(prompt: str, completion: str) -> float` and SHALL return a finite float value.
2. THE Pipeline SHALL provide a built-in `ExactMatchReward` that returns `1.0` when the
   completion exactly matches the `reference_answer` field of the `Dataset_Record` and `0.0`
   otherwise.
3. THE Pipeline SHALL provide a built-in `FormatReward` that returns `1.0` when the completion
   matches a user-supplied regex pattern and `0.0` otherwise.
4. IF a `Reward_Function` returns a non-finite value (NaN or infinity), THEN THE Training_Loop
   SHALL replace it with `0.0` and log a WARNING identifying the prompt and completion that
   produced the invalid reward.
5. THE Pipeline SHALL allow users to supply a custom `Reward_Function` by passing any callable
   matching the signature — no subclassing SHALL be required.

---

### Requirement 6: Checkpointing

**User Story:** As an ML engineer, I want the pipeline to save checkpoints during training, so
that I can resume from the last saved step after an interruption.

#### Acceptance Criteria

1. THE Checkpointer SHALL write a `Checkpoint` to disk after every N training steps, where N is
   a configurable parameter with a default value of 100.
2. THE Checkpoint SHALL record the current step number, the current epoch number, and the
   adapter weights path returned by the Tinker API.
3. WHEN the pipeline starts, THE Checkpointer SHALL scan the checkpoint directory and resume
   from the highest-numbered step if a valid `Checkpoint` exists.
4. IF writing a checkpoint fails, THEN THE Checkpointer SHALL log the error at WARNING level and
   continue training without halting.
5. THE Checkpointer SHALL serialize checkpoints as JSON files named
   `checkpoint_step_{step:06d}.json` in a configurable output directory.

---

### Requirement 7: LoRA Adapter Export

**User Story:** As an ML engineer, I want to download the trained LoRA adapter after training
completes, so that I can merge it with the base model for deployment.

#### Acceptance Criteria

1. WHEN training completes, THE Exporter SHALL call `weights.download(tinker_path=...,
   output_dir=...)` to download the LoRA_Adapter to a local directory.
2. WHEN the download completes, THE Exporter SHALL verify that the adapter directory contains at
   least one `.safetensors` or `.bin` file and SHALL raise an `ExportError` if no such file is
   found.
3. WHEN the adapter is verified, THE Exporter SHALL call `weights.build_hf_model(base_model=
   "Qwen/Qwen3-8B", adapter_path=..., output_path=...)` to produce the Merged_Model.
4. WHEN the merge completes, THE Exporter SHALL verify that the output directory contains a
   `config.json` file and SHALL raise an `ExportError` if it is absent.
5. IF any export step raises an exception, THEN THE Exporter SHALL log the exception at ERROR
   level and re-raise it.

---

### Requirement 8: GGUF Conversion and Quantization

**User Story:** As an ML engineer, I want to convert the merged HF model to a quantized GGUF
file, so that it can be loaded efficiently by Ollama on consumer hardware.

#### Acceptance Criteria

1. THE Converter SHALL invoke `llama.cpp`'s `convert_hf_to_gguf.py` script as a subprocess,
   passing the Merged_Model directory as input and a configurable output path for the GGUF file.
2. WHEN `convert_hf_to_gguf.py` exits with a non-zero return code, THE Converter SHALL raise a
   `ConversionError` containing the captured stderr output.
3. WHEN the GGUF file is produced, THE Converter SHALL invoke `llama-quantize` as a subprocess
   with a configurable quantization type, defaulting to `Q4_K_M`.
4. WHEN `llama-quantize` exits with a non-zero return code, THE Converter SHALL raise a
   `ConversionError` containing the captured stderr output.
5. WHEN quantization completes, THE Converter SHALL verify that the output quantized GGUF file
   exists on disk and has a size greater than 0 bytes, and SHALL raise a `ConversionError` if
   either condition is not met.
6. THE Converter SHALL log the file size of the quantized GGUF file at INFO level upon
   successful completion.

---

### Requirement 9: Ollama Modelfile Generation and Registration

**User Story:** As a developer, I want the pipeline to generate an Ollama Modelfile and register
the model, so that I can run the fine-tuned model locally with `ollama run`.

#### Acceptance Criteria

1. THE Modelfile_Generator SHALL produce a valid Ollama Modelfile that references the quantized
   GGUF_Model via a `FROM` directive pointing to its absolute path.
2. THE Modelfile_Generator SHALL accept configurable inference parameters — at minimum
   `temperature`, `top_p`, and `num_ctx` — and SHALL write them as `PARAMETER` directives in
   the Modelfile.
3. THE Modelfile_Generator SHALL accept an optional system prompt string and, WHEN provided,
   SHALL write it as a `SYSTEM` directive in the Modelfile.
4. WHEN the Modelfile is written to disk, THE Deployer SHALL invoke `ollama create <model_name>
   -f <modelfile_path>` as a subprocess.
5. WHEN `ollama create` exits with a non-zero return code, THE Deployer SHALL raise a
   `DeploymentError` containing the captured stderr output.
6. WHEN `ollama create` succeeds, THE Deployer SHALL log the model name and the path to the
   Modelfile at INFO level.
7. FOR ALL valid sets of inference parameters, generating a Modelfile then parsing it then
   generating it again SHALL produce an identical Modelfile (round-trip property).

---

### Requirement 10: Pipeline Orchestration and CLI

**User Story:** As a developer, I want a single CLI entry point that runs the full pipeline or
individual stages, so that I can automate training runs and debug individual steps.

#### Acceptance Criteria

1. THE CLI SHALL expose a `train` subcommand that accepts `--dataset`, `--epochs`,
   `--steps-per-checkpoint`, `--output-dir`, and `--model-name` as named arguments.
2. THE CLI SHALL expose an `export` subcommand that accepts `--adapter-path`, `--output-dir`,
   and `--model-name` as named arguments and runs only the export, conversion, and deployment
   stages.
3. WHEN `--help` is passed to any subcommand, THE CLI SHALL print a usage message describing all
   arguments and exit with code 0.
4. WHEN a required argument is missing, THE CLI SHALL print an error message identifying the
   missing argument and exit with code 2.
5. THE CLI SHALL configure structured logging to stdout at a level controlled by a `--log-level`
   argument defaulting to `INFO`.
6. WHEN the full `train` pipeline completes successfully, THE CLI SHALL exit with code 0.
7. IF any unhandled exception propagates to the CLI, THEN THE CLI SHALL log the exception at
   CRITICAL level and exit with code 1.

---

### Requirement 11: Observability and Logging

**User Story:** As an ML engineer, I want structured, consistent logs throughout the pipeline,
so that I can monitor training progress and diagnose failures.

#### Acceptance Criteria

1. THE Pipeline SHALL use Python's standard `logging` module with a single root logger named
   `"qwen3_rl_pipeline"`.
2. THE Pipeline SHALL emit a log entry at INFO level at the start and end of each major stage:
   data loading, client initialization, training loop, export, conversion, and deployment.
3. WHEN a training step completes, THE Training_Loop SHALL emit a structured log entry
   containing `step`, `epoch`, `mean_reward`, and `mean_advantage` as named fields.
4. THE Pipeline SHALL NOT log the value of `TINKER_API_KEY` or any other secret at any log
   level.
5. WHILE training is running, THE Pipeline SHALL emit a progress log at INFO level every 10
   steps showing elapsed time and estimated time remaining.
