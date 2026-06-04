"""Unit tests for qwen3_rl_pipeline.export.adapter.

Uses ``sys.modules`` injection to mock ``tinker_cookbook`` so the module can
be tested without the package installed.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qwen3_rl_pipeline.exceptions import ExportError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tinker_cookbook_mock() -> tuple[types.ModuleType, MagicMock]:
    """Return a (tinker_cookbook module mock, weights mock) pair."""
    weights_mock = MagicMock()
    tc_mock = types.ModuleType("tinker_cookbook")
    tc_mock.weights = weights_mock  # type: ignore[attr-defined]
    return tc_mock, weights_mock


# ---------------------------------------------------------------------------
# download_adapter tests
# ---------------------------------------------------------------------------

class TestDownloadAdapter:
    """Tests for :func:`download_adapter`."""

    def test_download_adapter_raises_export_error_when_no_safetensors(
        self, tmp_path: Path
    ) -> None:
        """ExportError is raised when no .safetensors or .bin file exists."""
        tc_mock, weights_mock = _make_tinker_cookbook_mock()
        # weights.download does nothing — no files are created
        weights_mock.download.return_value = None

        with patch.dict(sys.modules, {"tinker_cookbook": tc_mock}):
            from qwen3_rl_pipeline.export.adapter import download_adapter

            with pytest.raises(ExportError, match="No .safetensors or .bin file found"):
                download_adapter(
                    tinker_path="tinker://some/path",
                    output_dir=str(tmp_path),
                )

    def test_download_adapter_succeeds_with_safetensors_file(
        self, tmp_path: Path
    ) -> None:
        """Returns output_dir when a .safetensors file is present."""
        tc_mock, weights_mock = _make_tinker_cookbook_mock()
        weights_mock.download.return_value = None
        # Pre-create the expected weight file
        (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00")

        with patch.dict(sys.modules, {"tinker_cookbook": tc_mock}):
            from qwen3_rl_pipeline.export.adapter import download_adapter

            result = download_adapter(
                tinker_path="tinker://some/path",
                output_dir=str(tmp_path),
            )

        assert result == str(tmp_path)
        weights_mock.download.assert_called_once_with(
            tinker_path="tinker://some/path",
            output_dir=str(tmp_path),
        )

    def test_download_adapter_succeeds_with_bin_file(
        self, tmp_path: Path
    ) -> None:
        """Returns output_dir when a .bin file is present."""
        tc_mock, weights_mock = _make_tinker_cookbook_mock()
        weights_mock.download.return_value = None
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00")

        with patch.dict(sys.modules, {"tinker_cookbook": tc_mock}):
            from qwen3_rl_pipeline.export.adapter import download_adapter

            result = download_adapter(
                tinker_path="tinker://some/path",
                output_dir=str(tmp_path),
            )

        assert result == str(tmp_path)

    def test_download_adapter_exception_logged_and_reraised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RuntimeError from weights.download is logged at ERROR and re-raised."""
        tc_mock, weights_mock = _make_tinker_cookbook_mock()
        weights_mock.download.side_effect = RuntimeError("network failure")

        with patch.dict(sys.modules, {"tinker_cookbook": tc_mock}):
            from qwen3_rl_pipeline.export.adapter import download_adapter

            with caplog.at_level(logging.ERROR, logger="qwen3_rl_pipeline"):
                with pytest.raises(RuntimeError, match="network failure"):
                    download_adapter(
                        tinker_path="tinker://some/path",
                        output_dir=str(tmp_path),
                    )

        assert any(
            record.levelno == logging.ERROR
            for record in caplog.records
        ), "Expected an ERROR-level log entry"


# ---------------------------------------------------------------------------
# merge_adapter tests
# ---------------------------------------------------------------------------

class TestMergeAdapter:
    """Tests for :func:`merge_adapter`."""

    def test_merge_adapter_raises_export_error_when_no_config_json(
        self, tmp_path: Path
    ) -> None:
        """ExportError is raised when config.json is absent from output_path."""
        tc_mock, weights_mock = _make_tinker_cookbook_mock()
        weights_mock.build_hf_model.return_value = None

        with patch.dict(sys.modules, {"tinker_cookbook": tc_mock}):
            from qwen3_rl_pipeline.export.adapter import merge_adapter

            with pytest.raises(ExportError, match="config.json not found"):
                merge_adapter(
                    adapter_path="/some/adapter",
                    output_path=str(tmp_path),
                    allow_download=True,
                )

    def test_merge_adapter_succeeds_with_config_json(
        self, tmp_path: Path
    ) -> None:
        """Returns output_path when config.json is present."""
        tc_mock, weights_mock = _make_tinker_cookbook_mock()
        weights_mock.build_hf_model.return_value = None
        (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')

        with patch.dict(sys.modules, {"tinker_cookbook": tc_mock}):
            from qwen3_rl_pipeline.export.adapter import merge_adapter

            result = merge_adapter(
                adapter_path="/some/adapter",
                output_path=str(tmp_path),
                base_model="Qwen/Qwen3-8B",
                allow_download=True,
            )

        assert result == str(tmp_path)
        weights_mock.build_hf_model.assert_called_once_with(
            base_model="Qwen/Qwen3-8B",
            adapter_path="/some/adapter",
            output_path=str(tmp_path),
        )
