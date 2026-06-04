"""Custom training script with a smarter reward function for oil & gas Q&A."""

import asyncio
import logging
import sys

from qwen3_rl_pipeline.clients import initialize_clients_async
from qwen3_rl_pipeline.config import TrainingConfig
from qwen3_rl_pipeline.data.loader import load_dataset
from qwen3_rl_pipeline.deployment.ollama import (
    generate_modelfile,
    register_model,
    write_modelfile,
)
from qwen3_rl_pipeline.config import ModelfileConfig
from qwen3_rl_pipeline.export.adapter import download_adapter, merge_adapter
from qwen3_rl_pipeline.export.converter import convert_to_gguf, quantize_gguf
from qwen3_rl_pipeline.training.checkpointing import Checkpointer
from qwen3_rl_pipeline.training.loop import run_training_loop
from qwen3_rl_pipeline.training.rewards import sanitize_reward

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("qwen3_rl_pipeline")


def contains_answer_reward(reference_answer: str | None):
    """Return a reward function that checks if the completion contains the answer.

    Returns 1.0 if the reference answer appears anywhere in the completion
    (case-insensitive), 0.0 otherwise. Much more lenient than exact match.
    """
    def reward_fn(prompt: str, completion: str) -> float:
        if not reference_answer:
            return 0.0
        # Case-insensitive substring match
        if reference_answer.lower() in completion.lower():
            return 1.0
        # Also check if any key word from the reference appears
        key_words = [w for w in reference_answer.lower().split() if len(w) > 3]
        if key_words:
            matches = sum(1 for w in key_words if w in completion.lower())
            return matches / len(key_words)
        return 0.0
    return reward_fn


class DatasetAwareReward:
    """Reward function that looks up the reference answer from the dataset."""

    def __init__(self, dataset):
        # Build a lookup from prompt -> reference_answer
        self._lookup = {
            record.prompt: record.reference_answer
            for record in dataset
            if record.reference_answer
        }

    def __call__(self, prompt: str, completion: str) -> float:
        reference = self._lookup.get(prompt)
        if not reference:
            return 0.0
        # Case-insensitive substring match
        if reference.lower() in completion.lower():
            return 1.0
        # Partial credit: fraction of key words found
        key_words = [w for w in reference.lower().split() if len(w) > 3]
        if key_words:
            matches = sum(1 for w in key_words if w in completion.lower())
            return sanitize_reward(matches / len(key_words))
        return 0.0


async def main():
    config = TrainingConfig(
        dataset_path="data.jsonl",
        output_dir="./output",
        model_name="oil-gas-qwen3",
        epochs=1,
        steps_per_checkpoint=10,
        num_samples_per_group=4,
        learning_rate=1e-4,
        lora_rank=32,
        max_tokens=256,
        temperature=0.8,
    )

    logger.info("Loading dataset...")
    dataset = load_dataset(config.dataset_path)

    logger.info("Initializing Tinker clients...")
    clients = await initialize_clients_async(rank=config.lora_rank)

    # Use the smarter dataset-aware reward function
    reward_fn = DatasetAwareReward(dataset)
    checkpointer = Checkpointer(config.output_dir, config.steps_per_checkpoint)

    logger.info("Starting training loop...")
    adapter_path = await run_training_loop(
        config, dataset, reward_fn, clients, checkpointer
    )
    logger.info("Training complete. Adapter path: %s", adapter_path)

    # Export pipeline
    logger.info("Downloading adapter...")
    adapter_dir = download_adapter(adapter_path, f"{config.output_dir}/adapter")

    logger.info("Merging adapter into base model...")
    merged_dir = merge_adapter(adapter_dir, f"{config.output_dir}/merged_model")

    logger.info("Converting to GGUF...")
    gguf_path = convert_to_gguf(
        merged_dir,
        f"{config.output_dir}/model.gguf",
        llama_cpp_dir="/usr/local/bin",
    )

    logger.info("Quantizing...")
    quantized_path = quantize_gguf(
        gguf_path,
        f"{config.output_dir}/model_q.gguf",
        quantization_type="Q4_K_M",
        llama_cpp_dir="/usr/local/bin",
    )

    logger.info("Registering with Ollama...")
    modelfile_config = ModelfileConfig(
        gguf_path=quantized_path,
        model_name=config.model_name,
        temperature=0.7,
        system_prompt=(
            "You are an expert in the oil and gas industry. "
            "Answer questions accurately and concisely."
        ),
    )
    modelfile_content = generate_modelfile(modelfile_config)
    modelfile_path = f"{config.output_dir}/Modelfile"
    write_modelfile(modelfile_content, modelfile_path)
    register_model(config.model_name, modelfile_path)

    logger.info("Done! Run with: ollama run %s", config.model_name)


if __name__ == "__main__":
    asyncio.run(main())
