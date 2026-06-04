# Model Comparison Web App

A simple web interface to compare base Qwen3-8B vs fine-tuned model responses side by side.

## Setup

```bash
source .venv/bin/activate
pip install fastapi uvicorn
```

## Run

```bash
export TINKER_API_KEY="your-tinker-key"
export ADAPTER_PATH="tinker://your-session-id/sampler_weights/final"

# Optional overrides
export BASE_MODEL="Qwen/Qwen3-8B"
export MAX_TOKENS="512"
export TEMPERATURE="0.7"

uvicorn webapp.app:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Features

- 12 predefined oil & gas prompts as clickable chips
- Custom prompt input
- Adjustable max tokens and temperature sliders
- Side-by-side comparison of base vs fine-tuned model
- Dark theme UI
