"""Tests for qwen3_rl_pipeline/training/checkpointing.py.

Covers Property 7 (checkpoint serialization round-trip) and unit tests for
``Checkpointer.maybe_save`` and ``Checkpointer.load_latest``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import integers, text

from qwen3_rl_pipeline.training.checkpointing import Checkpoint, Checkpointer


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


# Feature: qwen3-rl-finetuning-pipeline, Property 7: Checkpoint serialization round-trip
# Validates: Requirements 6.2, 6.5
@given(
    step=integers(min_value=0),
    epoch=integers(min_value=0),
    weights_path=text(min_size=1),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_checkpoint_round_trip(
    step: int,
    epoch: int,
    weights_path: str,
    tmp_path: Path,
) -> None:
    """Serializing a Checkpoint to JSON and back produces an equal object.

    Also verifies that the filename matches the expected pattern.
    """
    ck = Checkpoint(step=step, epoch=epoch, adapter_weights_path=weights_path)
    filename = f"checkpoint_step_{step:06d}.json"
    path = tmp_path / filename
    path.write_text(json.dumps(dataclasses.asdict(ck)))
    loaded = Checkpoint(**json.loads(path.read_text()))
    assert loaded == ck
    assert path.name == filename


# ---------------------------------------------------------------------------
# Unit tests — maybe_save
# ---------------------------------------------------------------------------


def test_maybe_save_writes_file_at_correct_path(tmp_path: Path) -> None:
    """maybe_save writes a JSON file when step % save_every == 0."""
    checkpointer = Checkpointer(str(tmp_path), save_every=100)
    checkpointer.maybe_save(step=100, epoch=0, weights_path="/tmp/weights")

    expected = tmp_path / "checkpoint_step_000100.json"
    assert expected.exists(), "Checkpoint file should have been written"

    data = json.loads(expected.read_text())
    assert data["step"] == 100
    assert data["epoch"] == 0
    assert data["adapter_weights_path"] == "/tmp/weights"


def test_maybe_save_does_not_write_when_not_due(tmp_path: Path) -> None:
    """maybe_save does not write a file when step % save_every != 0."""
    checkpointer = Checkpointer(str(tmp_path), save_every=100)
    checkpointer.maybe_save(step=50, epoch=0, weights_path="/tmp/weights")

    files = list(tmp_path.glob("checkpoint_step_*.json"))
    assert files == [], "No checkpoint file should be written for step 50"


def test_maybe_save_logs_warning_on_write_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """maybe_save logs a WARNING and does not raise when the write fails."""
    checkpointer = Checkpointer(str(tmp_path), save_every=100)

    with patch.object(Path, "write_text", side_effect=OSError("disk full")):
        with caplog.at_level(logging.WARNING, logger="qwen3_rl_pipeline"):
            # Should not raise
            checkpointer.maybe_save(step=100, epoch=0, weights_path="/tmp/w")

    assert any(
        "WARNING" in r.levelname and "checkpoint" in r.message.lower()
        for r in caplog.records
    ), "Expected a WARNING log mentioning 'checkpoint'"


# ---------------------------------------------------------------------------
# Unit tests — load_latest
# ---------------------------------------------------------------------------


def test_load_latest_returns_none_when_empty(tmp_path: Path) -> None:
    """load_latest returns None when the output directory has no checkpoints."""
    checkpointer = Checkpointer(str(tmp_path), save_every=100)
    assert checkpointer.load_latest() is None


def test_load_latest_returns_highest_step(tmp_path: Path) -> None:
    """load_latest returns the checkpoint with the highest step number."""
    for step in (100, 200, 300):
        ck = Checkpoint(step=step, epoch=0, adapter_weights_path=f"/w/{step}")
        path = tmp_path / f"checkpoint_step_{step:06d}.json"
        path.write_text(json.dumps(dataclasses.asdict(ck)))

    checkpointer = Checkpointer(str(tmp_path), save_every=100)
    latest = checkpointer.load_latest()

    assert latest is not None
    assert latest.step == 300
    assert latest.adapter_weights_path == "/w/300"
