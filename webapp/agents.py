"""Agent-based evaluation architecture for comparing model responses.

Architecture:
1. BaseModelAgent      — queries the base Qwen3-8B model
2. FinetunedAgent      — queries the fine-tuned model
3. JudgePanelAgent     — runs 3 judges in parallel, aggregates by majority vote

The three judges use different evaluation perspectives:
- Judge 1 (Technical Accuracy): focuses on correctness and domain terminology
- Judge 2 (Completeness):       focuses on coverage of key concepts
- Judge 3 (Practical Value):    focuses on actionability and engineering relevance
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator

import tinker

logger = logging.getLogger("webapp.agents")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Response from a single model agent."""
    agent_name: str
    model_id: str
    response: str
    tokens_generated: int = 0
    error: str | None = None


@dataclass
class SingleJudgeVerdict:
    """Verdict from one individual judge."""
    judge_name: str
    perspective: str
    base_score: float
    finetuned_score: float
    winner: str
    reasoning: str


@dataclass
class PanelVerdict:
    """Aggregated verdict from the full judge panel."""
    # Individual judge verdicts
    judges: list[SingleJudgeVerdict]
    # Aggregated scores (average across judges)
    base_score_avg: float
    finetuned_score_avg: float
    # Majority vote winner
    winner: str
    # Vote counts
    base_votes: int
    finetuned_votes: int
    tie_votes: int
    # Summary
    panel_reasoning: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Model agent
# ---------------------------------------------------------------------------

class ModelAgent:
    """Agent that queries a Tinker sampling client and returns a response."""

    def __init__(
        self,
        name: str,
        client: object,
        tokenizer: object,
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.name = name
        self.client = client
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run(self, prompt: str) -> AgentResponse:
        """Generate a response for the given prompt."""
        try:
            tokens = self.tokenizer.encode(prompt)
            model_input = tinker.ModelInput.from_ints(tokens=tokens)
            params = tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            result = await self.client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
            )
            response_tokens = result.sequences[0].tokens
            response_text = self.tokenizer.decode(response_tokens)
            return AgentResponse(
                agent_name=self.name,
                model_id=self.model_id,
                response=response_text,
                tokens_generated=len(response_tokens),
            )
        except Exception as e:
            logger.error("Agent %s failed: %s", self.name, e, exc_info=True)
            return AgentResponse(
                agent_name=self.name,
                model_id=self.model_id,
                response="",
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Individual judge agent
# ---------------------------------------------------------------------------

# Three judge prompts — each focuses on a different evaluation dimension.
JUDGE_PROMPTS = {
    "Technical Accuracy": """You are Judge 1 — a petroleum engineer evaluating technical accuracy.

Focus ONLY on: Are the facts correct? Is the terminology accurate? Are the mechanisms explained correctly?

QUESTION: {prompt}
BASE MODEL: {base_response}
FINE-TUNED MODEL: {finetuned_response}

Score each answer 0-10 for TECHNICAL ACCURACY only.
Respond in EXACT format:
BASE_SCORE: [0-10]
FINETUNED_SCORE: [0-10]
WINNER: [base/finetuned/tie]
REASONING: [2 sentences on technical accuracy only]""",

    "Completeness": """You are Judge 2 — an oil & gas training specialist evaluating completeness.

Focus ONLY on: Does the answer cover all key concepts? Are important aspects missing? Is the depth appropriate?

QUESTION: {prompt}
BASE MODEL: {base_response}
FINE-TUNED MODEL: {finetuned_response}

Score each answer 0-10 for COMPLETENESS only.
Respond in EXACT format:
BASE_SCORE: [0-10]
FINETUNED_SCORE: [0-10]
WINNER: [base/finetuned/tie]
REASONING: [2 sentences on completeness only]""",

    "Practical Value": """You are Judge 3 — a drilling operations manager evaluating practical value.

Focus ONLY on: Is this answer useful to a working engineer? Does it connect theory to practice? Is it actionable?

QUESTION: {prompt}
BASE MODEL: {base_response}
FINE-TUNED MODEL: {finetuned_response}

Score each answer 0-10 for PRACTICAL VALUE only.
Respond in EXACT format:
BASE_SCORE: [0-10]
FINETUNED_SCORE: [0-10]
WINNER: [base/finetuned/tie]
REASONING: [2 sentences on practical value only]""",
}


class SingleJudgeAgent:
    """One judge in the panel, focused on a specific evaluation dimension."""

    def __init__(
        self,
        judge_name: str,
        perspective: str,
        client: object,
        tokenizer: object,
        max_tokens: int = 256,
    ) -> None:
        self.judge_name = judge_name
        self.perspective = perspective
        self.client = client
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._prompt_template = JUDGE_PROMPTS[perspective]

    async def run(
        self,
        prompt: str,
        base_response: str,
        finetuned_response: str,
    ) -> SingleJudgeVerdict:
        """Score both responses from this judge's perspective."""
        judge_prompt = self._prompt_template.format(
            prompt=prompt,
            base_response=base_response or "(no response)",
            finetuned_response=finetuned_response or "(no response)",
        )

        try:
            tokens = self.tokenizer.encode(judge_prompt)
            model_input = tinker.ModelInput.from_ints(tokens=tokens)
            params = tinker.SamplingParams(max_tokens=self.max_tokens, temperature=0.1)
            result = await self.client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=params
            )
            verdict_text = self.tokenizer.decode(result.sequences[0].tokens)
            return self._parse(verdict_text)
        except Exception as e:
            logger.error("Judge %s failed: %s", self.judge_name, e, exc_info=True)
            return SingleJudgeVerdict(
                judge_name=self.judge_name,
                perspective=self.perspective,
                base_score=5.0,
                finetuned_score=5.0,
                winner="tie",
                reasoning=f"Judge failed: {e}",
            )

    def _parse(self, text: str) -> SingleJudgeVerdict:
        import re  # noqa: PLC0415

        def extract(pattern: str, default: str = "") -> str:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else default

        try:
            base_score = max(0.0, min(10.0, float(extract(r"BASE_SCORE:\s*([0-9.]+)", "5"))))
            ft_score = max(0.0, min(10.0, float(extract(r"FINETUNED_SCORE:\s*([0-9.]+)", "5"))))
        except ValueError:
            base_score = ft_score = 5.0

        winner_raw = extract(r"WINNER:\s*(\w+)", "tie").lower()
        if "fine" in winner_raw:
            winner = "finetuned"
        elif winner_raw == "base":
            winner = "base"
        else:
            winner = "tie" if abs(base_score - ft_score) < 0.5 else (
                "finetuned" if ft_score > base_score else "base"
            )

        return SingleJudgeVerdict(
            judge_name=self.judge_name,
            perspective=self.perspective,
            base_score=base_score,
            finetuned_score=ft_score,
            winner=winner,
            reasoning=extract(r"REASONING:\s*(.+?)(?:\n[A-Z_]+:|$)", "See scores."),
        )


# ---------------------------------------------------------------------------
# Judge panel — runs 3 judges in parallel and aggregates
# ---------------------------------------------------------------------------

class JudgePanelAgent:
    """Runs 3 judges in parallel and aggregates their verdicts.

    Aggregation:
    - Scores: average across all judges
    - Winner: majority vote (2 out of 3)
    """

    def __init__(self, judges: list[SingleJudgeAgent]) -> None:
        self.judges = judges

    async def run(
        self,
        prompt: str,
        base_response: str,
        finetuned_response: str,
    ) -> PanelVerdict:
        """Run all judges in parallel and aggregate results."""
        verdicts = await asyncio.gather(
            *[j.run(prompt, base_response, finetuned_response) for j in self.judges],
            return_exceptions=False,
        )

        base_scores = [v.base_score for v in verdicts]
        ft_scores = [v.finetuned_score for v in verdicts]
        base_avg = sum(base_scores) / len(base_scores)
        ft_avg = sum(ft_scores) / len(ft_scores)

        # Majority vote
        base_votes = sum(1 for v in verdicts if v.winner == "base")
        ft_votes = sum(1 for v in verdicts if v.winner == "finetuned")
        tie_votes = sum(1 for v in verdicts if v.winner == "tie")

        if ft_votes > base_votes and ft_votes > tie_votes:
            winner = "finetuned"
        elif base_votes > ft_votes and base_votes > tie_votes:
            winner = "base"
        else:
            winner = "tie" if abs(base_avg - ft_avg) < 0.5 else (
                "finetuned" if ft_avg > base_avg else "base"
            )

        # Build panel summary
        vote_summary = f"Votes — Base: {base_votes}, Fine-tuned: {ft_votes}, Tie: {tie_votes}."
        score_summary = f"Average scores — Base: {base_avg:.1f}/10, Fine-tuned: {ft_avg:.1f}/10."
        panel_reasoning = f"{vote_summary} {score_summary}"

        return PanelVerdict(
            judges=list(verdicts),
            base_score_avg=base_avg,
            finetuned_score_avg=ft_avg,
            winner=winner,
            base_votes=base_votes,
            finetuned_votes=ft_votes,
            tie_votes=tie_votes,
            panel_reasoning=panel_reasoning,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EvaluationOrchestrator:
    """Orchestrates model agents and judge panel, streams results via SSE."""

    def __init__(
        self,
        base_agent: ModelAgent,
        finetuned_agent: ModelAgent,
        judge_panel: JudgePanelAgent,
    ) -> None:
        self.base_agent = base_agent
        self.finetuned_agent = finetuned_agent
        self.judge_panel = judge_panel

    async def evaluate(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Run all agents and yield SSE progress events."""
        yield {"event": "start", "data": {"prompt": prompt}}
        yield {"event": "status", "data": {"message": "Generating responses from both models..."}}

        base_resp, ft_resp = await asyncio.gather(
            self.base_agent.run(prompt),
            self.finetuned_agent.run(prompt),
        )

        yield {"event": "base_response", "data": {
            "agent": base_resp.agent_name,
            "model": base_resp.model_id,
            "response": base_resp.response,
            "tokens": base_resp.tokens_generated,
            "error": base_resp.error,
        }}

        yield {"event": "finetuned_response", "data": {
            "agent": ft_resp.agent_name,
            "model": ft_resp.model_id,
            "response": ft_resp.response,
            "tokens": ft_resp.tokens_generated,
            "error": ft_resp.error,
        }}

        yield {"event": "status", "data": {"message": "3 judges evaluating responses in parallel..."}}

        panel = await self.judge_panel.run(
            prompt=prompt,
            base_response=base_resp.response,
            finetuned_response=ft_resp.response,
        )

        # Stream individual judge verdicts first.
        for v in panel.judges:
            yield {"event": "judge_verdict", "data": {
                "judge_name": v.judge_name,
                "perspective": v.perspective,
                "base_score": v.base_score,
                "finetuned_score": v.finetuned_score,
                "winner": v.winner,
                "reasoning": v.reasoning,
            }}

        # Then stream the aggregated panel verdict.
        yield {"event": "panel_verdict", "data": {
            "base_score_avg": panel.base_score_avg,
            "finetuned_score_avg": panel.finetuned_score_avg,
            "winner": panel.winner,
            "base_votes": panel.base_votes,
            "finetuned_votes": panel.finetuned_votes,
            "tie_votes": panel.tie_votes,
            "panel_reasoning": panel.panel_reasoning,
            "error": panel.error,
        }}

        yield {"event": "complete", "data": {"prompt": prompt}}
