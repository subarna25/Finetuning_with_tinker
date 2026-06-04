"""Unit tests for qwen3_rl_pipeline.export.converter.

Patches ``subprocess.run`` to simulate zero and non-zero exit codes without
invoking any real llama.cpp binaries.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from qwen3_rl_pipeline.exceptions import ConversionError
from qwen3_rl_pipeline.export.converter import convert_to_gguf, quantize_gguf


# ---------------------------------------------------------------------------
# convert_to_gguf tests
# ---------------------------------------------------------------------------

class TestConvertToGguf:
    """Tests for :func:`convert_to_gguf`."""

    def test_convert_to_gguf_raises_on_nonzero_exit(self) -> None:
        """ConversionError is raised with stderr when exit code is non-zero."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error msg"
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ConversionError, match="error msg"):
                convert_to_gguf(
                    hf_model_dir="/models/hf",
                    output_path="/models/out.gguf",
                    llama_cpp_dir="/usr/local/bin",
                )

    def test_convert_to_gguf_succeeds_on_zero_exit(self) -> None:
        """Returns output_path when the subprocess exits with code 0."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = convert_to_gguf(
                hf_model_dir="/models/hf",
                output_path="/models/out.gguf",
                llama_cpp_dir="/usr/local/bin",
            )
        assert result == "/models/out.gguf"


# ---------------------------------------------------------------------------
# quantize_gguf tests
# ---------------------------------------------------------------------------

class TestQuantizeGguf:
    """Tests for :func:`quantize_gguf`."""

    def test_quantize_gguf_raises_on_nonzero_exit(self) -> None:
        """ConversionError is raised when llama-quantize exits non-zero."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="quant error"
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ConversionError, match="quant error"):
                quantize_gguf(
                    input_path="/models/in.gguf",
                    output_path="/models/out_q.gguf",
                )

    def test_quantize_gguf_raises_when_output_missing(
        self, tmp_path: Path
    ) -> None:
        """ConversionError is raised when the output file does not exist."""
        output_path = str(tmp_path / "out_q.gguf")
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ConversionError, match="Quantized GGUF not found or empty"
            ):
                quantize_gguf(
                    input_path="/models/in.gguf",
                    output_path=output_path,
                )

    def test_quantize_gguf_raises_when_output_empty(
        self, tmp_path: Path
    ) -> None:
        """ConversionError is raised when the output file is 0 bytes."""
        output_file = tmp_path / "out_q.gguf"
        output_file.write_bytes(b"")  # empty file
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ConversionError, match="Quantized GGUF not found or empty"
            ):
                quantize_gguf(
                    input_path="/models/in.gguf",
                    output_path=str(output_file),
                )

    def test_quantize_gguf_logs_file_size_on_success(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An INFO log containing the file size is emitted on success."""
        output_file = tmp_path / "out_q.gguf"
        output_file.write_bytes(b"x" * 1024)  # 1 KB of content
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            with caplog.at_level(logging.INFO, logger="qwen3_rl_pipeline"):
                result = quantize_gguf(
                    input_path="/models/in.gguf",
                    output_path=str(output_file),
                )

        assert result == str(output_file)
        size_logs = [
            r.message
            for r in caplog.records
            if "MB" in r.message and r.levelno == logging.INFO
        ]
        assert size_logs, "Expected an INFO log entry containing file size in MB"
