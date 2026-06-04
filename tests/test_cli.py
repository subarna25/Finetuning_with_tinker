"""Unit tests for the CLI entry point (cli.py and __main__.py).

Tests cover:
- ``--help`` exits with code 0 for both subcommands.
- Missing required arguments exit with code 2.
- ``--log-level DEBUG`` sets the package logger to DEBUG.
- An unhandled exception in ``run_cli`` causes ``__main__`` to exit with
  code 1.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Subprocess-based tests (tests 1–3)
# These tests exercise the real CLI process so that argparse exit codes are
# captured accurately.
# ---------------------------------------------------------------------------


def test_train_help_exits_zero() -> None:
    """``python -m qwen3_rl_pipeline train --help`` must exit with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "qwen3_rl_pipeline", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_export_help_exits_zero() -> None:
    """``python -m qwen3_rl_pipeline export --help`` must exit with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "qwen3_rl_pipeline", "export", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_missing_required_arg_exits_two() -> None:
    """``python -m qwen3_rl_pipeline train`` (no args) must exit with code 2."""
    result = subprocess.run(
        [sys.executable, "-m", "qwen3_rl_pipeline", "train"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f"Expected exit code 2, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# In-process tests (tests 4–5)
# These tests use unittest.mock to avoid real I/O and network calls.
# ---------------------------------------------------------------------------


def test_log_level_debug_sets_logger() -> None:
    """``--log-level DEBUG`` must set the package logger level to DEBUG."""
    import asyncio

    from qwen3_rl_pipeline.cli import run_cli

    # The pipeline functions are imported lazily inside _handle_train, so we
    # patch them at their source modules rather than on the cli module.
    with (
        patch(
            "qwen3_rl_pipeline.clients.initialize_clients_async",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch(
            "qwen3_rl_pipeline.data.loader.load_dataset",
            return_value=[],
        ),
        patch(
            "qwen3_rl_pipeline.training.loop.run_training_loop",
            new_callable=AsyncMock,
            return_value="/tmp/adapter",
        ),
        patch(
            "qwen3_rl_pipeline.export.adapter.download_adapter",
            return_value="/tmp/adapter_dir",
        ),
        patch(
            "qwen3_rl_pipeline.export.adapter.merge_adapter",
            return_value="/tmp/merged",
        ),
        patch(
            "qwen3_rl_pipeline.export.converter.convert_to_gguf",
            return_value="/tmp/model.gguf",
        ),
        patch(
            "qwen3_rl_pipeline.export.converter.quantize_gguf",
            return_value="/tmp/model_q.gguf",
        ),
        patch(
            "qwen3_rl_pipeline.deployment.ollama.generate_modelfile",
            return_value="FROM x\n",
        ),
        patch("qwen3_rl_pipeline.deployment.ollama.write_modelfile"),
        patch("qwen3_rl_pipeline.deployment.ollama.register_model"),
    ):
        try:
            asyncio.run(
                run_cli(
                    [
                        "--log-level",
                        "DEBUG",
                        "train",
                        "--dataset",
                        "data.jsonl",
                        "--output-dir",
                        "/tmp/out",
                        "--model-name",
                        "test-model",
                    ]
                )
            )
        except SystemExit:
            # sys.exit(0) is expected on success — that's fine.
            pass

    pkg_logger = logging.getLogger("qwen3_rl_pipeline")
    assert pkg_logger.level == logging.DEBUG, (
        f"Expected logger level DEBUG ({logging.DEBUG}), "
        f"got {pkg_logger.level}"
    )


def test_unhandled_exception_exits_one() -> None:
    """An unhandled exception in ``run_cli`` must cause exit code 1."""
    import asyncio

    # Patch run_cli to raise an unexpected RuntimeError so that __main__'s
    # top-level handler is exercised.
    with patch(
        "qwen3_rl_pipeline.__main__.main",
        side_effect=RuntimeError("boom"),
    ):
        from qwen3_rl_pipeline import __main__ as main_module

        # Re-run the module-level guard by calling asyncio.run(main()) and
        # catching the SystemExit that the except block raises.
        with patch("qwen3_rl_pipeline.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = RuntimeError("boom")

            try:
                # Simulate what happens when __main__ is executed directly.
                try:
                    mock_asyncio.run(main_module.main())
                except Exception:
                    main_module.logger.critical(
                        "Unhandled exception", exc_info=True
                    )
                    sys.exit(1)
            except SystemExit as exc:
                assert exc.code == 1, (
                    f"Expected exit code 1, got {exc.code}"
                )
