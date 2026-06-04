"""Tests for qwen3_rl_pipeline/data/loader.py.

Covers:
- Property 1: JSONL parse-serialize-parse round-trip
- Property 2: Whitespace-only prompts are rejected
- Unit tests for error paths and successful loading
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis.strategies import (
    builds,
    characters,
    lists,
    none,
    one_of,
    text,
)

from qwen3_rl_pipeline.data.loader import load_dataset
from qwen3_rl_pipeline.data.models import DatasetRecord
from qwen3_rl_pipeline.exceptions import DatasetParseError, DatasetValidationError

# ---------------------------------------------------------------------------
# Property 1: JSONL parse-serialize-parse round-trip
# ---------------------------------------------------------------------------

# Feature: qwen3-rl-finetuning-pipeline, Property 1: JSONL parse-serialize-parse round-trip
# Validates: Requirements 2.1, 2.2, 2.5, 2.6
@given(
    records=lists(
        builds(
            DatasetRecord,
            prompt=text(min_size=1).filter(str.strip),
            reference_answer=one_of(none(), text()),
        ),
        min_size=1,
    )
)
@settings(max_examples=100)
def test_jsonl_round_trip(records: list[DatasetRecord]) -> None:
    """Serializing records to JSONL and loading them back produces equal records."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as fh:
        fh.write(
            "\n".join(
                json.dumps({k: v for k, v in asdict(r).items() if v is not None})
                for r in records
            )
            + "\n"
        )
        tmp_path = fh.name
    result = load_dataset(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    assert result == records


# ---------------------------------------------------------------------------
# Property 2: Whitespace-only prompts are rejected
# ---------------------------------------------------------------------------

# Feature: qwen3-rl-finetuning-pipeline, Property 2: Whitespace-only prompts are rejected
# Validates: Requirements 2.4
@given(
    prompt=text(
        # Zs = Unicode space separators; whitelist_chars adds ASCII whitespace
        # control characters that Python's str.strip() recognises as whitespace.
        alphabet=characters(
            whitelist_categories=("Zs",),
            whitelist_characters="\t\n\r\x0b\x0c ",
        ),
        min_size=1,
    )
)
@settings(max_examples=100)
def test_whitespace_prompt_rejected(prompt: str) -> None:
    """A prompt consisting entirely of whitespace characters raises DatasetValidationError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as fh:
        fh.write(json.dumps({"prompt": prompt}) + "\n")
        tmp_path = fh.name
    try:
        with pytest.raises(DatasetValidationError):
            load_dataset(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Unit tests — error paths
# ---------------------------------------------------------------------------


def test_invalid_json_raises_parse_error(tmp_path: pytest.TempPathFactory) -> None:
    """A line that is not valid JSON raises DatasetParseError with the correct line number."""
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "valid line"}) + "\n"
        + "not valid json\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetParseError) as exc_info:
        load_dataset(str(jsonl))
    assert exc_info.value.line_number == 2
    assert "not valid json" in exc_info.value.raw_content


def test_invalid_json_on_first_line_raises_parse_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Invalid JSON on the very first line reports line_number == 1."""
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(DatasetParseError) as exc_info:
        load_dataset(str(jsonl))
    assert exc_info.value.line_number == 1


def test_missing_prompt_key_raises_validation_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A JSON object without a 'prompt' key raises DatasetValidationError."""
    jsonl = tmp_path / "no_prompt.jsonl"
    jsonl.write_text(json.dumps({"reference_answer": "answer"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset(str(jsonl))
    assert exc_info.value.field_name == "prompt"


def test_empty_string_prompt_raises_validation_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A 'prompt' field that is an empty string raises DatasetValidationError."""
    jsonl = tmp_path / "empty_prompt.jsonl"
    jsonl.write_text(json.dumps({"prompt": ""}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset(str(jsonl))
    assert exc_info.value.field_name == "prompt"


def test_whitespace_only_prompt_raises_validation_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A 'prompt' field that is only spaces raises DatasetValidationError."""
    jsonl = tmp_path / "ws_prompt.jsonl"
    jsonl.write_text(json.dumps({"prompt": "   "}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset(str(jsonl))
    assert exc_info.value.field_name == "prompt"


# ---------------------------------------------------------------------------
# Unit tests — successful load
# ---------------------------------------------------------------------------


def test_successful_load_returns_correct_count(tmp_path: pytest.TempPathFactory) -> None:
    """A valid JSONL file returns the correct number of DatasetRecord objects."""
    records = [
        DatasetRecord(prompt="What is 2+2?", reference_answer="4"),
        DatasetRecord(prompt="Name a planet.", reference_answer=None),
        DatasetRecord(prompt="Hello world."),
    ]
    jsonl = tmp_path / "valid.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({k: v for k, v in asdict(r).items() if v is not None})
            for r in records
        )
        + "\n",
        encoding="utf-8",
    )
    result = load_dataset(str(jsonl))
    assert len(result) == 3
    assert result == records


def test_successful_load_logs_at_info(
    tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """load_dataset logs the total record count at INFO level after a successful load."""
    jsonl = tmp_path / "valid.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "First question"}) + "\n"
        + json.dumps({"prompt": "Second question"}) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="qwen3_rl_pipeline"):
        result = load_dataset(str(jsonl))

    assert len(result) == 2
    assert any(
        "2" in record.message and "record" in record.message.lower()
        for record in caplog.records
        if record.name == "qwen3_rl_pipeline"
    )


def test_blank_lines_are_skipped(tmp_path: pytest.TempPathFactory) -> None:
    """Blank lines in the JSONL file are silently skipped."""
    jsonl = tmp_path / "blanks.jsonl"
    jsonl.write_text(
        "\n"
        + json.dumps({"prompt": "First"}) + "\n"
        + "   \n"
        + json.dumps({"prompt": "Second"}) + "\n"
        + "\n",
        encoding="utf-8",
    )
    result = load_dataset(str(jsonl))
    assert len(result) == 2
    assert result[0].prompt == "First"
    assert result[1].prompt == "Second"


def test_reference_answer_included_when_present(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Optional reference_answer is preserved in the returned DatasetRecord."""
    jsonl = tmp_path / "with_ref.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "What is 2+2?", "reference_answer": "4"}) + "\n",
        encoding="utf-8",
    )
    result = load_dataset(str(jsonl))
    assert result[0].reference_answer == "4"


def test_reference_answer_absent_when_not_in_file(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """When reference_answer is absent from the JSONL, the field is None."""
    jsonl = tmp_path / "no_ref.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "What is 2+2?"}) + "\n",
        encoding="utf-8",
    )
    result = load_dataset(str(jsonl))
    assert result[0].reference_answer is None
