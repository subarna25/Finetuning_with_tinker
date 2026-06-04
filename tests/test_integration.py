"""Integration tests for the Qwen3 RL fine-tuning pipeline.

These tests require:
- A valid TINKER_API_KEY environment variable
- Ollama installed and running locally
- llama.cpp binaries available

All tests are automatically skipped when TINKER_API_KEY is not set.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_end_to_end_train_pipeline() -> None:
    """End-to-end train run on a 5-record JSONL fixture with 1 epoch and 1 step.

    Creates a temp JSONL file with 5 records, runs the full train pipeline
    via the CLI, and verifies that a quantized GGUF file and Ollama model
    are produced.
    """
    pytest.skip("Integration test stub — implement when TINKER_API_KEY is available")


@pytest.mark.integration
def test_export_subcommand_with_adapter_fixture() -> None:
    """Export subcommand against a pre-downloaded adapter fixture.

    Downloads a known adapter from Tinker, runs the export subcommand, and
    verifies the GGUF and Modelfile are produced correctly.
    """
    pytest.skip(
        "Integration test stub — implement when adapter fixture is available"
    )


@pytest.mark.integration
def test_ollama_create_invocation() -> None:
    """Ollama create invocation with a real GGUF file.

    Requires Ollama to be installed and running locally. Verifies that
    ``ollama create`` succeeds and the model appears in ``ollama list``.
    """
    pytest.skip(
        "Integration test stub — implement when Ollama is available locally"
    )
