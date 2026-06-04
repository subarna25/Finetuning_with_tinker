# Qwen3 RL Fine-Tuning Pipeline

End-to-end reinforcement learning fine-tuning for `Qwen/Qwen3-8B` using the
[Tinker API](https://tinker-docs.thinkingmachines.ai/) from Thinking Machines Lab.
The pipeline covers data loading, GRPO/CISPO RL training, LoRA adapter export,
GGUF conversion, and local deployment via [Ollama](https://ollama.com/).

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.11 and 3.14 |
| [Tinker API key](https://thinkingmachines.ai/tinker/) | Set as `TINKER_API_KEY` env var |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | `convert_hf_to_gguf.py` and `llama-quantize` binaries |
| [Ollama](https://ollama.com/) | Running locally for model registration |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Set your Tinker API key
export TINKER_API_KEY="your-api-key-here"
```

## Usage

### Train a model

Fine-tune `Qwen/Qwen3-8B` on a JSONL dataset and deploy the result to Ollama:

```bash
python -m qwen3_rl_pipeline train \
    --dataset data/train.jsonl \
    --output-dir ./output \
    --model-name my-qwen3-model \
    --epochs 3 \
    --steps-per-checkpoint 100 \
    --lora-rank 32 \
    --learning-rate 1e-4 \
    --max-tokens 512 \
    --quantization-type Q4_K_M \
    --llama-cpp-dir /usr/local/bin
```

The `--dataset` file must be a JSONL file where each line is a JSON object with
at minimum a `"prompt"` key. An optional `"reference_answer"` key is used by the
default `ExactMatchReward` function:

```jsonl
{"prompt": "What is the capital of France?", "reference_answer": "Paris"}
{"prompt": "Solve: 2 + 2 = ?", "reference_answer": "4"}
```

### Export an existing adapter

If you already have a trained adapter on Tinker, skip training and go straight
to export:

```bash
python -m qwen3_rl_pipeline export \
    --adapter-path "tinker://<run-id>/sampler_weights/<checkpoint-name>" \
    --output-dir ./output \
    --model-name my-qwen3-model \
    --quantization-type Q4_K_M \
    --llama-cpp-dir /usr/local/bin
```

### Run the model

After a successful `train` or `export` run, the model is registered with Ollama:

```bash
ollama run my-qwen3-model
```

## Running Tests

### Unit tests (no API key required)

```bash
pytest tests/ -m "not integration"
```

### Integration tests (requires `TINKER_API_KEY` and Ollama)

```bash
TINKER_API_KEY=your-key pytest tests/ -m integration
```

## Project Structure

```
qwen3_rl_pipeline/
├── __main__.py          # CLI entry point
├── cli.py               # train and export subcommand handlers
├── config.py            # Frozen configuration dataclasses
├── clients.py           # Tinker client factory
├── exceptions.py        # Custom exception hierarchy
├── data/
│   ├── loader.py        # JSONL loading and validation
│   └── models.py        # DatasetRecord dataclass
├── training/
│   ├── loop.py          # Async GRPO/CISPO RL training loop
│   ├── rewards.py       # RewardFunction protocol + built-ins
│   └── checkpointing.py # Checkpoint read/write
├── export/
│   ├── adapter.py       # LoRA download + HF merge
│   └── converter.py     # GGUF conversion + quantization
└── deployment/
    └── ollama.py        # Modelfile generation + ollama create
```

## Pipeline Overview

```
JSONL Dataset
    ↓ load_dataset
DatasetRecords
    ↓ run_training_loop (GRPO/CISPO via Tinker)
LoRA Adapter (tinker://...)
    ↓ download_adapter + merge_adapter
Merged HF Model
    ↓ convert_to_gguf + quantize_gguf (llama.cpp)
Quantized GGUF (Q4_K_M)
    ↓ generate_modelfile + ollama create
Ollama Model → ollama run
```
