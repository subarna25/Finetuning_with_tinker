# Product

**Qwen3 RL Fine-Tuning Pipeline** is an end-to-end reinforcement learning fine-tuning tool for `Qwen/Qwen3-8B` using the [Tinker API](https://tinker-docs.thinkingmachines.ai/) from Thinking Machines Lab.

## What it does

Takes a JSONL prompt dataset and produces a locally-runnable Ollama model through this pipeline:

```
JSONL Dataset → RL Training (GRPO/CISPO via Tinker) → LoRA Adapter
→ HuggingFace Merge → GGUF Conversion (llama.cpp) → Quantization → Ollama Deployment
```

## Two entry points

- **`train`** — full pipeline: train on a dataset, export, convert, and register with Ollama
- **`export`** — skip training; download an existing Tinker adapter and export it to Ollama

## External dependencies

- **Tinker API** — remote RL training; requires `TINKER_API_KEY` env var
- **llama.cpp** — `convert_hf_to_gguf.py` and `llama-quantize` binaries for GGUF conversion
- **Ollama** — local model serving; must be running for registration to succeed
