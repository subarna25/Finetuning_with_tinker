"""GGUF conversion and quantization utilities for the export pipeline.

This module provides two functions:

- :func:`convert_to_gguf`: converts a HuggingFace model directory to GGUF
  format using ``convert_hf_to_gguf.py`` from llama.cpp.
- :func:`quantize_gguf`: quantizes a GGUF file using ``llama-quantize`` from
  llama.cpp and verifies the output file is non-empty.

Both functions raise :class:`~qwen3_rl_pipeline.exceptions.ConversionError`
on failure and log progress at INFO level.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from qwen3_rl_pipeline.exceptions import ConversionError

logger = logging.getLogger("qwen3_rl_pipeline")


def convert_to_gguf(
    hf_model_dir: str,
    output_path: str,
    llama_cpp_dir: str,
) -> str:
    """Convert a HuggingFace model directory to GGUF format.

    Invokes ``convert_hf_to_gguf.py`` from the llama.cpp distribution via
    ``subprocess.run`` and raises :class:`ConversionError` if the process
    exits with a non-zero return code.

    Args:
        hf_model_dir: Path to the HuggingFace model directory to convert.
        output_path: Destination path for the output ``.gguf`` file.
        llama_cpp_dir: Directory containing ``convert_hf_to_gguf.py``.

    Returns:
        The ``output_path`` on success.

    Raises:
        ConversionError: If ``convert_hf_to_gguf.py`` exits with a non-zero
            return code. The captured stderr is included in the message.
    """
    logger.info("Converting HF model to GGUF...")
    cmd = [
        "python",
        f"{llama_cpp_dir}/convert_hf_to_gguf.py",
        hf_model_dir,
        "--outfile",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(
            f"convert_hf_to_gguf.py failed:\n{result.stderr}"
        )
    logger.info("GGUF file written to %s", output_path)
    return output_path


def quantize_gguf(
    input_path: str,
    output_path: str,
    quantization_type: str = "Q4_K_M",
    llama_cpp_dir: str = "/usr/local/bin",
) -> str:
    """Quantize a GGUF file using llama-quantize.

    Invokes ``llama-quantize`` from the llama.cpp distribution via
    ``subprocess.run``, then verifies that the output file exists and has a
    non-zero size.

    Args:
        input_path: Path to the source ``.gguf`` file to quantize.
        output_path: Destination path for the quantized ``.gguf`` file.
        quantization_type: llama.cpp quantization type string, e.g.
            ``"Q4_K_M"``. Defaults to ``"Q4_K_M"``.
        llama_cpp_dir: Directory containing the ``llama-quantize`` binary.
            Defaults to ``"/usr/local/bin"``.

    Returns:
        The ``output_path`` on success.

    Raises:
        ConversionError: If ``llama-quantize`` exits with a non-zero return
            code, or if the output file is missing or empty after the process
            completes.
    """
    logger.info("Quantizing GGUF with %s...", quantization_type)
    cmd = [
        f"{llama_cpp_dir}/llama-quantize",
        input_path,
        output_path,
        quantization_type,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(f"llama-quantize failed:\n{result.stderr}")

    output_file = Path(output_path)
    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise ConversionError(
            f"Quantized GGUF not found or empty: {output_path}"
        )

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("Quantized GGUF: %s (%.1f MB)", output_path, size_mb)
    return output_path
