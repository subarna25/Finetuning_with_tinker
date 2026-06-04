"""FastAPI web application for comparing base vs fine-tuned model responses.

Two modes:
1. /api/compare  — simple side-by-side comparison (no scoring)
2. /api/evaluate — agent-based evaluation with LLM judge scoring,
                   streamed via Server-Sent Events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import tinker
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webapp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")

# Auto-detect adapter path from saved file if ADAPTER_PATH not set.
def _resolve_adapter_path() -> str:
    explicit = os.environ.get("ADAPTER_PATH", "")
    if explicit:
        return explicit
    # Search common output directories for saved path file.
    for candidate in ["output_v2", "output", "output_v3"]:
        path_file = os.path.join(candidate, "tinker_adapter_path.txt")
        if os.path.exists(path_file):
            with open(path_file) as f:
                path = f.read().strip()
            if path:
                logger.info("Auto-detected adapter path from %s: %s", path_file, path)
                return path
    return ""

ADAPTER_PATH = _resolve_adapter_path()
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

# ---------------------------------------------------------------------------
# Predefined prompts
# ---------------------------------------------------------------------------
PREDEFINED_PROMPTS = [
    "What is a blowout preventer (BOP) and what is its purpose?",
    "Explain the difference between upstream, midstream, and downstream operations.",
    "What is hydraulic fracturing (fracking) and how does it work?",
    "What is LNG and how is it produced?",
    "What is the purpose of drilling mud in oil well operations?",
    "What is the difference between sweet and sour crude oil?",
    "What is a Christmas tree in oil and gas?",
    "What is enhanced oil recovery (EOR)?",
    "What is a FPSO and where is it used?",
    "What is the purpose of a wellbore casing?",
    "What is reservoir pressure and why is it important?",
    "Explain the difference between conventional and unconventional oil resources.",
]

# ---------------------------------------------------------------------------
# Global clients
# ---------------------------------------------------------------------------
_clients: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Tinker clients and agents on startup."""
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key:
        logger.warning("TINKER_API_KEY not set — model queries will fail.")
    else:
        try:
            logger.info("Initializing Tinker clients...")
            service_client = tinker.ServiceClient()

            base_client = await service_client.create_sampling_client_async(
                base_model=BASE_MODEL
            )
            finetuned_client = await service_client.create_sampling_client_async(
                model_path=ADAPTER_PATH
            )
            judge_client = await service_client.create_sampling_client_async(
                base_model=JUDGE_MODEL
            )

            tokenizer = base_client.get_tokenizer()

            _clients["service"] = service_client
            _clients["base"] = base_client
            _clients["finetuned"] = finetuned_client
            _clients["judge"] = judge_client
            _clients["tokenizer"] = tokenizer

            # Build the evaluation orchestrator with 3-judge panel.
            from webapp.agents import (
                EvaluationOrchestrator,
                JudgePanelAgent,
                ModelAgent,
                SingleJudgeAgent,
            )

            # All 3 judges use the same model but different prompts/perspectives.
            judge1 = SingleJudgeAgent("Judge 1", "Technical Accuracy", judge_client, tokenizer)
            judge2 = SingleJudgeAgent("Judge 2", "Completeness", judge_client, tokenizer)
            judge3 = SingleJudgeAgent("Judge 3", "Practical Value", judge_client, tokenizer)

            _clients["orchestrator"] = EvaluationOrchestrator(
                base_agent=ModelAgent(
                    name="Base Model",
                    client=base_client,
                    tokenizer=tokenizer,
                    model_id=BASE_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                ),
                finetuned_agent=ModelAgent(
                    name="Fine-tuned Model",
                    client=finetuned_client,
                    tokenizer=tokenizer,
                    model_id="oil-gas-qwen3",
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                ),
                judge_panel=JudgePanelAgent(judges=[judge1, judge2, judge3]),
            )

            logger.info("All clients and agents initialized.")
        except Exception as e:
            logger.error("Failed to initialize: %s", e, exc_info=True)

    yield
    _clients.clear()


app = FastAPI(title="Oil & Gas Model Comparison", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE


class EvaluateRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _query(client: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    tokens = tokenizer.encode(prompt)
    model_input = tinker.ModelInput.from_ints(tokens=tokens)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature)
    result = await client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
    return tokenizer.decode(result.sequences[0].tokens)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/prompts")
async def get_prompts() -> dict:
    return {"prompts": PREDEFINED_PROMPTS}


@app.post("/api/compare")
async def compare(req: CompareRequest) -> dict:
    """Simple side-by-side comparison without scoring."""
    if "base" not in _clients:
        raise HTTPException(status_code=503, detail="Clients not initialized.")
    try:
        base_resp, ft_resp = await asyncio.gather(
            _query(_clients["base"], _clients["tokenizer"], req.prompt, req.max_tokens, req.temperature),
            _query(_clients["finetuned"], _clients["tokenizer"], req.prompt, req.max_tokens, req.temperature),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "prompt": req.prompt,
        "base_response": base_resp,
        "finetuned_response": ft_resp,
        "base_model": BASE_MODEL,
        "finetuned_model": "oil-gas-qwen3",
    }


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest) -> StreamingResponse:
    """Agent-based evaluation with LLM judge scoring, streamed via SSE."""
    if "orchestrator" not in _clients:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")

    orchestrator = _clients["orchestrator"]
    # Update agent settings from request.
    orchestrator.base_agent.max_tokens = req.max_tokens
    orchestrator.base_agent.temperature = req.temperature
    orchestrator.finetuned_agent.max_tokens = req.max_tokens
    orchestrator.finetuned_agent.temperature = req.temperature

    async def event_stream():
        async for event in orchestrator.evaluate(req.prompt):
            data = json.dumps(event)
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open("webapp/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
