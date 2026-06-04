"""Model comparison utilities for evaluating fine-tuned vs base model responses.

This module provides functions to:
1. Publish a fine-tuned LoRA adapter to HuggingFace Hub (merge happens on
   Tinker's servers — no local download required).
2. Query both the base model and the fine-tuned model with the same prompts.
3. Display a side-by-side comparison of responses.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger("qwen3_rl_pipeline")


@dataclass
class ComparisonResult:
    """Result of comparing base model vs fine-tuned model on a single prompt.

    Attributes:
        prompt: The input prompt used for both models.
        base_response: Response from the base Qwen3-8B model.
        finetuned_response: Response from the fine-tuned model.
        hf_repo_id: HuggingFace repo ID where the fine-tuned model was published.
    """

    prompt: str
    base_response: str
    finetuned_response: str
    hf_repo_id: str


async def publish_to_hf(
    tinker_path: str,
    hf_repo_id: str,
    base_model: str = "Qwen/Qwen3-8B",
    hf_token: str | None = None,
    output_dir: str = "/tmp/tinker_export",
) -> str:
    """Download LoRA adapter from Tinker, merge with base model, and publish to HF.

    Downloads the adapter to a temp directory, merges it with the base model,
    then uploads the merged model to HuggingFace Hub.

    Args:
        tinker_path: Tinker path to the LoRA adapter.
        hf_repo_id: HuggingFace repository ID (e.g. ``"user/oil-gas-qwen3"``).
        base_model: HuggingFace model ID of the base model.
        hf_token: HuggingFace API token. Uses HF_TOKEN env var if None.
        output_dir: Local directory for temporary adapter and merged model files.

    Returns:
        The HuggingFace repo URL of the published model.
    """
    import os  # noqa: PLC0415
    from tinker_cookbook import weights  # noqa: PLC0415

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HuggingFace token required. Set HF_TOKEN environment variable "
            "or pass --hf-token."
        )

    adapter_dir = os.path.join(output_dir, "adapter")
    merged_dir = os.path.join(output_dir, "merged_model")

    logger.info("Downloading LoRA adapter from Tinker...")
    weights.download(tinker_path=tinker_path, output_dir=adapter_dir)

    logger.info("Merging adapter with base model %s...", base_model)
    logger.info(
        "NOTE: This downloads the full base model (~16 GB) from HuggingFace. "
        "Consider using publish-and-compare with --skip-publish instead."
    )
    weights.build_hf_model(
        base_model=base_model,
        adapter_path=adapter_dir,
        output_path=merged_dir,
    )

    logger.info("Publishing merged model to HuggingFace Hub: %s", hf_repo_id)
    url = weights.publish_to_hf_hub(
        model_path=merged_dir,
        repo_id=hf_repo_id,
        private=True,
        token=token,
    )

    logger.info("Model published successfully: %s", url)
    return url


async def query_model(
    client: object,
    prompt: str,
    tokenizer: object,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Query a Tinker sampling client with a prompt and return the response.

    Args:
        client: A Tinker ``SamplingClient`` instance.
        prompt: The input prompt text.
        tokenizer: Tokenizer for encoding/decoding.
        max_tokens: Maximum tokens to generate. Defaults to 512.
        temperature: Sampling temperature. Defaults to 0.7.

    Returns:
        The decoded response string.
    """
    import tinker  # noqa: PLC0415

    tokens = tokenizer.encode(prompt)
    model_input = tinker.ModelInput.from_ints(tokens=tokens)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = await client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )

    return tokenizer.decode(result.sequences[0].tokens)


async def compare_models(
    service_client: object,
    tinker_path: str,
    prompts: list[str],
    hf_repo_id: str,
    base_model: str = "Qwen/Qwen3-8B",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[ComparisonResult]:
    """Query both base and fine-tuned models and return comparison results.

    Creates two sampling clients — one for the base model and one for the
    fine-tuned model — then queries both with each prompt in parallel.

    Args:
        service_client: A Tinker ``ServiceClient`` instance.
        tinker_path: Tinker path to the fine-tuned adapter weights.
        prompts: List of prompt strings to compare.
        hf_repo_id: HuggingFace repo ID (used for display only).
        base_model: HuggingFace model ID of the base model.
        max_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.

    Returns:
        List of ``ComparisonResult`` objects, one per prompt.
    """
    logger.info("Creating base model sampling client...")
    base_client = await service_client.create_sampling_client_async(
        base_model=base_model
    )

    logger.info("Creating fine-tuned model sampling client...")
    finetuned_client = await service_client.create_sampling_client_async(
        model_path=tinker_path
    )

    # Use the base model tokenizer for both (same vocabulary).
    tokenizer = base_client.get_tokenizer()

    results: list[ComparisonResult] = []

    for i, prompt in enumerate(prompts):
        logger.info("Comparing prompt %d/%d: %s...", i + 1, len(prompts), prompt[:60])

        # Query both models concurrently.
        base_response, finetuned_response = await asyncio.gather(
            query_model(base_client, prompt, tokenizer, max_tokens, temperature),
            query_model(finetuned_client, prompt, tokenizer, max_tokens, temperature),
        )

        results.append(
            ComparisonResult(
                prompt=prompt,
                base_response=base_response,
                finetuned_response=finetuned_response,
                hf_repo_id=hf_repo_id,
            )
        )

    return results


def print_comparison(results: list[ComparisonResult]) -> None:
    """Print a formatted side-by-side comparison of model responses.

    Args:
        results: List of ``ComparisonResult`` objects to display.
    """
    separator = "=" * 80

    print(f"\n{separator}")
    print("MODEL COMPARISON: Base Qwen3-8B vs Fine-tuned Model")
    print(f"Fine-tuned model: {results[0].hf_repo_id if results else 'N/A'}")
    print(separator)

    for i, result in enumerate(results, 1):
        print(f"\n{'─' * 80}")
        print(f"PROMPT {i}: {result.prompt}")
        print(f"{'─' * 80}")

        print("\n📦 BASE MODEL (Qwen/Qwen3-8B):")
        print(f"  {result.base_response.strip()}")

        print(f"\n✨ FINE-TUNED MODEL ({result.hf_repo_id}):")
        print(f"  {result.finetuned_response.strip()}")

    print(f"\n{separator}\n")


def save_comparison_to_file(
    results: list[ComparisonResult], output_path: str
) -> None:
    """Save comparison results to a JSON file.

    Args:
        results: List of ``ComparisonResult`` objects.
        output_path: Path to write the JSON output file.
    """
    import json  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Comparison results saved to %s", output_path)
