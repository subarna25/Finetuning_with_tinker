"""JSONL dataset loader with line-level validation for the RL fine-tuning pipeline."""

from __future__ import annotations

import json
import logging

from qwen3_rl_pipeline.data.models import DatasetRecord
from qwen3_rl_pipeline.exceptions import DatasetParseError, DatasetValidationError

logger = logging.getLogger("qwen3_rl_pipeline")


def load_dataset(path: str) -> list[DatasetRecord]:
    """Load and validate a JSONL file into DatasetRecord objects.

    Reads the file at *path* line by line, skipping blank lines. Each
    non-blank line is parsed as JSON and validated before being converted
    into a :class:`~qwen3_rl_pipeline.data.models.DatasetRecord`.

    Args:
        path: Absolute or relative path to the JSONL file.

    Returns:
        List of validated :class:`~qwen3_rl_pipeline.data.models.DatasetRecord`
        objects, one per non-blank line.

    Raises:
        DatasetParseError: If any non-blank line is not valid JSON. The
            exception carries the 1-based line number and the raw line
            content.
        DatasetValidationError: If any parsed record is missing the
            ``"prompt"`` key or its stripped value is empty. The exception
            carries the 1-based line number and the field name
            ``"prompt"``.
    """
    records: list[DatasetRecord] = []

    with open(path, encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                continue

            try:
                parsed: dict[str, object] = json.loads(raw_line)
            except json.JSONDecodeError:
                raise DatasetParseError(line_number, raw_line)

            prompt = parsed.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise DatasetValidationError(line_number, "prompt")

            records.append(DatasetRecord.from_dict(parsed))

    logger.info("Loaded %d records from %s", len(records), path)
    return records
