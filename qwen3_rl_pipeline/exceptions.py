"""Custom exception hierarchy for the Qwen3 RL fine-tuning pipeline.

All pipeline-specific exceptions inherit from ``PipelineError``, so callers
can catch the entire family with a single ``except PipelineError`` clause.
"""


class PipelineError(Exception):
    """Base class for all pipeline-specific exceptions.

    Catch this class to handle any error raised by the pipeline without
    needing to enumerate every derived type.
    """


class DatasetParseError(PipelineError):
    """Raised when a JSONL line cannot be decoded as valid JSON.

    Attributes:
        line_number: 1-based index of the offending line in the JSONL file.
        raw_content: The raw string content of the offending line.
    """

    def __init__(self, line_number: int, raw_content: str) -> None:
        """Initialise with the line number and raw content of the bad line.

        Args:
            line_number: 1-based index of the offending line.
            raw_content: The raw string content of the offending line.
        """
        self.line_number: int = line_number
        self.raw_content: str = raw_content
        super().__init__(
            f"Line {line_number} is not valid JSON: {raw_content!r}"
        )


class DatasetValidationError(PipelineError):
    """Raised when a parsed record fails field-level validation.

    This is raised when a required field (e.g. ``prompt``) is absent from a
    record or its value is empty after stripping whitespace.

    Attributes:
        line_number: 1-based index of the offending line in the JSONL file.
        field_name: Name of the field that failed validation.
    """

    def __init__(self, line_number: int, field_name: str) -> None:
        """Initialise with the line number and the name of the invalid field.

        Args:
            line_number: 1-based index of the offending line.
            field_name: Name of the field that failed validation.
        """
        self.line_number: int = line_number
        self.field_name: str = field_name
        super().__init__(
            f"Line {line_number}: field '{field_name}' is missing or empty"
        )


class ExportError(PipelineError):
    """Raised when a LoRA adapter download or HuggingFace model merge fails.

    This covers failures in ``export/adapter.py``, such as a missing
    ``.safetensors`` / ``.bin`` file after download, or a missing
    ``config.json`` after the HF merge step.
    """


class ConversionError(PipelineError):
    """Raised when GGUF conversion or quantization fails.

    This is raised when ``convert_hf_to_gguf.py`` or ``llama-quantize``
    exits with a non-zero return code, or when the expected output file is
    absent or empty after the subprocess completes.
    """


class DeploymentError(PipelineError):
    """Raised when ``ollama create`` exits with a non-zero return code.

    The exception message includes the captured stderr output from the
    subprocess to aid debugging.
    """
