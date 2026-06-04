"""Tests for the Ollama deployment module.

Covers:
- Property 6: Modelfile generation round-trip (Hypothesis)
- Unit tests for generate_modelfile, register_model
"""

# Feature: qwen3-rl-finetuning-pipeline, Property 6: Modelfile generation round-trip

from __future__ import annotations

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis.strategies import (
    floats,
    integers,
    none,
    one_of,
    text,
)

from qwen3_rl_pipeline.config import ModelfileConfig
from qwen3_rl_pipeline.deployment.ollama import (
    generate_modelfile,
    parse_modelfile,
    register_model,
)
from qwen3_rl_pipeline.exceptions import DeploymentError


# ---------------------------------------------------------------------------
# Property 6: Modelfile generation round-trip
# Validates: Requirements 9.1, 9.2, 9.3, 9.7
# ---------------------------------------------------------------------------


@given(
    gguf_path=text(min_size=1).filter(lambda s: s.strip() and "\n" not in s),
    model_name=text(min_size=1).filter(lambda s: s.strip()),
    temperature=floats(
        min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
    top_p=floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    num_ctx=integers(min_value=128, max_value=131072),
    system_prompt=one_of(
        none(), text().filter(lambda s: '"""' not in s)
    ),
)
@settings(deadline=None)
def test_modelfile_round_trip(
    gguf_path: str,
    model_name: str,
    temperature: float,
    top_p: float,
    num_ctx: int,
    system_prompt: str | None,
) -> None:
    """Generating, parsing, and re-generating a Modelfile yields the same string.

    **Validates: Requirements 9.1, 9.2, 9.3, 9.7**
    """
    config = ModelfileConfig(
        gguf_path=gguf_path,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        system_prompt=system_prompt,
    )
    first = generate_modelfile(config)
    parsed = parse_modelfile(first)
    second = generate_modelfile(parsed)
    assert first == second


# ---------------------------------------------------------------------------
# Unit tests — generate_modelfile
# ---------------------------------------------------------------------------


def test_generate_modelfile_includes_from_and_parameters() -> None:
    """generate_modelfile includes FROM and all three PARAMETER lines."""
    config = ModelfileConfig(
        gguf_path="/models/my-model.gguf",
        model_name="my-model",
        temperature=0.5,
        top_p=0.8,
        num_ctx=2048,
    )
    content = generate_modelfile(config)

    assert "FROM /models/my-model.gguf" in content
    assert "PARAMETER temperature 0.5" in content
    assert "PARAMETER top_p 0.8" in content
    assert "PARAMETER num_ctx 2048" in content


def test_generate_modelfile_includes_system_when_set() -> None:
    """generate_modelfile includes the SYSTEM block when system_prompt is set."""
    config = ModelfileConfig(
        gguf_path="/models/my-model.gguf",
        model_name="my-model",
        system_prompt="You are a helpful assistant.",
    )
    content = generate_modelfile(config)

    assert 'SYSTEM """You are a helpful assistant."""' in content


def test_generate_modelfile_omits_system_when_none() -> None:
    """generate_modelfile omits the SYSTEM directive when system_prompt is None."""
    config = ModelfileConfig(
        gguf_path="/models/my-model.gguf",
        model_name="my-model",
        system_prompt=None,
    )
    content = generate_modelfile(config)

    assert "SYSTEM" not in content


# ---------------------------------------------------------------------------
# Unit tests — register_model
# ---------------------------------------------------------------------------


def test_register_model_raises_deployment_error_on_nonzero_exit() -> None:
    """register_model raises DeploymentError when ollama create exits non-zero."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "error: model not found"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(DeploymentError) as exc_info:
            register_model("my-model", "/path/to/Modelfile")

    assert "ollama create failed" in str(exc_info.value)
    assert "error: model not found" in str(exc_info.value)


def test_register_model_logs_on_success(caplog: pytest.LogCaptureFixture) -> None:
    """register_model logs at INFO level on successful registration."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        with caplog.at_level(logging.INFO, logger="qwen3_rl_pipeline"):
            register_model("my-model", "/path/to/Modelfile")

    log_messages = caplog.text
    assert "my-model" in log_messages
    assert "/path/to/Modelfile" in log_messages
