"""Configuration dataclasses for the Qwen3 RL fine-tuning pipeline.

This module defines three frozen dataclasses that carry all configuration
needed by the training, export, and deployment stages of the pipeline.
Frozen dataclasses are used so that configuration objects are immutable
after construction, preventing accidental mutation during a run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the GRPO/CISPO RL training loop.

    Attributes:
        dataset_path: Path to the JSONL training dataset file.
        output_dir: Directory where checkpoints and exported artifacts are
            written.
        model_name: Ollama model name used for the final deployed model.
        epochs: Number of full passes over the dataset. Defaults to 3.
        steps_per_checkpoint: Number of optimiser steps between checkpoint
            saves. Defaults to 100.
        num_samples_per_group: Number of rollouts (completions) generated
            per ``EnvGroup`` during RL sampling. Defaults to 4.
        learning_rate: AdamW learning rate. Defaults to 1e-4.
        lora_rank: Rank of the LoRA adapter matrices. Defaults to 32.
        max_tokens: Maximum number of tokens to generate per rollout.
            Defaults to 512.
        temperature: Sampling temperature applied during RL rollouts.
            Defaults to 1.0.
        log_level: Python logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
            Defaults to ``"INFO"``.
    """

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
    """Configuration for the LoRA adapter export and GGUF conversion stages.

    Attributes:
        adapter_path: Tinker path to the LoRA adapter weights
            (e.g. ``"tinker://..."``).
        output_dir: Local directory where export artifacts are written.
        model_name: Ollama model name for the final registered model.
        gguf_output_path: Filesystem path for the unquantized GGUF file
            produced by ``convert_hf_to_gguf.py``.
        quantized_output_path: Filesystem path for the quantized GGUF file
            produced by ``llama-quantize``.
        quantization_type: llama.cpp quantization type string.
            Defaults to ``"Q4_K_M"``.
        llama_cpp_dir: Directory that contains the llama.cpp binaries
            (``convert_hf_to_gguf.py`` and ``llama-quantize``).
            Defaults to ``"/usr/local/bin"``.
    """

    adapter_path: str
    output_dir: str
    model_name: str
    gguf_output_path: str
    quantized_output_path: str
    quantization_type: str = "Q4_K_M"
    llama_cpp_dir: str = "/usr/local/bin"


@dataclass(frozen=True)
class ModelfileConfig:
    """Configuration for generating an Ollama Modelfile.

    Attributes:
        gguf_path: Absolute filesystem path to the quantized GGUF file that
            the Modelfile's ``FROM`` directive will reference.
        model_name: Ollama model name used when registering the model via
            ``ollama create``.
        temperature: Inference temperature written to the Modelfile
            ``PARAMETER temperature`` directive. Defaults to 0.7.
        top_p: Nucleus-sampling probability mass written to the Modelfile
            ``PARAMETER top_p`` directive. Defaults to 0.9.
        num_ctx: Context window size (in tokens) written to the Modelfile
            ``PARAMETER num_ctx`` directive. Defaults to 4096.
        system_prompt: Optional system prompt written as a ``SYSTEM``
            block in the Modelfile. When ``None`` the ``SYSTEM`` directive
            is omitted entirely. Defaults to ``None``.
    """

    gguf_path: str
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096
    system_prompt: str | None = None
