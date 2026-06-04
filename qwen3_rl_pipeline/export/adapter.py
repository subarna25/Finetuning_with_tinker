"""LoRA adapter download and merge utilities for the export pipeline.

This module provides two functions:

- :func:`download_adapter`: downloads LoRA adapter weights from Tinker to a
  local directory and verifies that at least one weight file is present.
- :func:`merge_adapter`: merges a LoRA adapter into the base model weights
  and verifies that the resulting HuggingFace model directory is valid.

Both functions use deferred imports for ``tinker_cookbook`` so that the module
can be imported cleanly in environments where the package is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qwen3_rl_pipeline.exceptions import ExportError

if TYPE_CHECKING:
    pass  # No TYPE_CHECKING-only imports needed here

logger = logging.getLogger("qwen3_rl_pipeline")


def download_adapter(tinker_path: str, output_dir: str) -> str:
    """Download LoRA adapter weights from Tinker to a local directory.

    Calls ``tinker_cookbook.weights.download`` and then verifies that at least
    one ``.safetensors`` or ``.bin`` file was written to ``output_dir``.

    Args:
        tinker_path: The Tinker path identifying the adapter to download.
        output_dir: Local directory where the adapter weights will be saved.

    Returns:
        The ``output_dir`` path on success.

    Raises:
        ExportError: If no ``.safetensors`` or ``.bin`` file is found in
            ``output_dir`` after the download completes.
        Exception: Any exception raised by ``tinker_cookbook.weights.download``
            is logged at ERROR and re-raised.
    """
    logger.info("Downloading LoRA adapter from %s...", tinker_path)
    try:
        from tinker_cookbook import weights  # noqa: PLC0415

        weights.download(tinker_path=tinker_path, output_dir=output_dir)

        output_path = Path(output_dir)
        weight_files = list(output_path.glob("*.safetensors")) + list(
            output_path.glob("*.bin")
        )
        if not weight_files:
            raise ExportError(
                f"No .safetensors or .bin file found in {output_dir}"
            )

        logger.info("Adapter downloaded to %s", output_dir)
        return output_dir
    except Exception:
        logger.error(
            "Failed to download adapter from %s to %s",
            tinker_path,
            output_dir,
        )
        raise


def merge_adapter(
    adapter_path: str,
    output_path: str,
    base_model: str = "Qwen/Qwen3-8B",
    allow_download: bool = False,
) -> str:
    """Merge a LoRA adapter into the base model weights.

    .. warning::
        This function downloads the full base model (~16 GB) from HuggingFace.
        Pass ``allow_download=True`` explicitly to permit the download.
        By default this is blocked to prevent accidental large downloads.

    Args:
        adapter_path: Local path to the downloaded LoRA adapter directory.
        output_path: Directory where the merged HuggingFace model will be saved.
        base_model: HuggingFace model identifier for the base model.
        allow_download: Must be ``True`` to permit downloading the base model.
            Defaults to ``False`` (raises ``ExportError`` if not set).

    Returns:
        The ``output_path`` on success.

    Raises:
        ExportError: If ``allow_download`` is False, or if ``config.json`` is
            absent from ``output_path`` after the merge completes.
    """
    if not allow_download:
        raise ExportError(
            "merge_adapter would download the full base model (~16 GB). "
            "Pass allow_download=True to permit this, or use "
            "'publish-and-compare' subcommand to publish directly to "
            "HuggingFace without a local download."
        )
    logger.info("Merging adapter into base model %s...", base_model)
    try:
        from tinker_cookbook import weights  # noqa: PLC0415

        weights.build_hf_model(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
        )

        config_file = Path(output_path) / "config.json"
        if not config_file.exists():
            raise ExportError(f"config.json not found in {output_path}")

        logger.info("Merged model saved to %s", output_path)
        return output_path
    except Exception:
        logger.error(
            "Failed to merge adapter %s into base model %s at %s",
            adapter_path,
            base_model,
            output_path,
        )
        raise
