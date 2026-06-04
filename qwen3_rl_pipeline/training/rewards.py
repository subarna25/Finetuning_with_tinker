"""Reward function protocol and built-in implementations for RL training.

This module defines the ``RewardFunction`` protocol and concrete implementations:
- ``ExactMatchReward`` — binary exact string match
- ``FormatReward`` — regex pattern match
- ``SubstringReward`` — case-insensitive substring + partial keyword match
- ``LLMJudgeReward`` — uses a Tinker-hosted grader LLM with a rubric (async)
- ``sanitize_reward`` — replaces non-finite values with 0.0
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any, Protocol

logger = logging.getLogger("qwen3_rl_pipeline")


class RewardFunction(Protocol):
    """Protocol for reward functions used in the RL training loop.

    Any callable that accepts a prompt and a completion and returns a float
    satisfies this protocol — no subclassing required.
    """

    def __call__(self, prompt: str, completion: str) -> float:
        """Score a single (prompt, completion) pair.

        Args:
            prompt: The input prompt that was given to the model.
            completion: The model's generated completion.

        Returns:
            A scalar reward value. Should be finite; non-finite values will be
            sanitized to ``0.0`` by the training loop.
        """
        ...


class ExactMatchReward:
    """Reward function that returns 1.0 on exact string equality.

    Example:
        >>> reward = ExactMatchReward("Paris")
        >>> reward(prompt="Capital of France?", completion="Paris")
        1.0
    """

    def __init__(self, reference_answer: str) -> None:
        """Initialise with the expected reference answer.

        Args:
            reference_answer: The string that completions are compared against.
        """
        self._reference_answer = reference_answer

    def __call__(self, prompt: str, completion: str) -> float:
        """Return 1.0 if completion exactly matches the reference, else 0.0."""
        return 1.0 if completion == self._reference_answer else 0.0


class FormatReward:
    """Reward function that returns 1.0 when a completion matches a regex.

    Example:
        >>> reward = FormatReward(r"\\d+")
        >>> reward(prompt="", completion="The answer is 42")
        1.0
    """

    def __init__(self, pattern: str) -> None:
        """Compile the regex pattern at construction time.

        Args:
            pattern: A regular expression string. Compiled with ``re.compile``.
        """
        self._pattern: re.Pattern[str] = re.compile(pattern)

    def __call__(self, prompt: str, completion: str) -> float:
        """Return 1.0 if the pattern matches anywhere in completion, else 0.0."""
        return 1.0 if self._pattern.search(completion) else 0.0


class SubstringReward:
    """Reward function using case-insensitive substring + partial keyword match.

    Returns 1.0 if the reference answer appears anywhere in the completion
    (case-insensitive). Falls back to partial credit based on keyword overlap.

    This is more lenient than ExactMatchReward and works well for open-ended
    answers where the model generates full sentences.

    Example:
        >>> reward = SubstringReward({"What is Paris?": "capital of France"})
        >>> reward(prompt="What is Paris?", completion="Paris is the capital of France.")
        1.0
    """

    def __init__(self, dataset: list[Any]) -> None:
        """Build a prompt → reference_answer lookup from the dataset.

        Args:
            dataset: List of DatasetRecord objects with prompt and
                reference_answer fields.
        """
        self._lookup: dict[str, str] = {
            record.prompt: record.reference_answer
            for record in dataset
            if record.reference_answer
        }

    def __call__(self, prompt: str, completion: str) -> float:
        """Score based on substring match and keyword overlap.

        Args:
            prompt: The input prompt used to look up the reference answer.
            completion: The model's generated completion.

        Returns:
            1.0 for full match, 0.0–1.0 for partial keyword match, 0.0 if
            no reference answer is available.
        """
        reference = self._lookup.get(prompt)
        if not reference:
            return 0.0
        if reference.lower() in completion.lower():
            return 1.0
        key_words = [w for w in reference.lower().split() if len(w) > 3]
        if key_words:
            matches = sum(1 for w in key_words if w in completion.lower())
            return sanitize_reward(matches / len(key_words))
        return 0.0


class LLMJudgeReward:
    """Reward function that uses a Tinker-hosted LLM to grade completions.

    Uses the native Tinker cookbook ``Rubric`` class and a ``SamplingClient``
    to call a grader LLM (e.g. Qwen3-30B) that scores the model's response
    against a rubric. This is the most powerful reward signal for open-ended
    domain Q&A tasks.

    The grader LLM is called asynchronously. Since the training loop calls
    reward functions synchronously, this class runs the async grader call
    in the current event loop using ``asyncio.get_event_loop().run_until_complete``.

    Example rubric for oil & gas Q&A::

        rubric_str = (
            "Score 1.0 if the answer correctly explains what a BOP is, "
            "mentions it prevents blowouts, and describes its location at the "
            "wellhead. Score 0.5 if only 2 of 3 criteria are met. Score 0.0 "
            "if the answer is wrong or missing key information."
        )
        reward = LLMJudgeReward(
            grader_client=sampling_client,
            default_rubric_str=rubric_str,
            grader_model="Qwen/Qwen3-30B-A3B-Instruct",
        )

    Args:
        grader_client: A Tinker ``SamplingClient`` configured with the grader
            model. Use a stronger model than the one being trained.
        default_rubric_str: Default rubric text used when no per-prompt rubric
            is available. Should describe what a good answer looks like and
            how to score it.
        grader_model: HuggingFace model ID of the grader model. Used for
            tokenization. Defaults to ``"Qwen/Qwen3-30B-A3B-Instruct"``.
        format_coef: Weight for format compliance in the reward. Defaults to
            ``0.1`` (10% of total reward).
        prompt_rubrics: Optional dict mapping prompt strings to custom rubric
            strings. When provided, the per-prompt rubric overrides the
            default rubric for that prompt.
    """

    def __init__(
        self,
        grader_client: Any,
        default_rubric_str: str,
        grader_model: str = "Qwen/Qwen3-30B-A3B-Instruct",
        format_coef: float = 0.1,
        prompt_rubrics: dict[str, str] | None = None,
    ) -> None:
        """Initialise the LLM judge reward function.

        Args:
            grader_client: Tinker SamplingClient for the grader model.
            default_rubric_str: Default rubric text for scoring.
            grader_model: HuggingFace model ID for the grader tokenizer.
            format_coef: Weight for format compliance (0.0–1.0).
            prompt_rubrics: Optional per-prompt rubric overrides.
        """
        self._grader_client = grader_client
        self._default_rubric_str = default_rubric_str
        self._grader_model = grader_model
        self._format_coef = format_coef
        self._prompt_rubrics = prompt_rubrics or {}

        # Import Tinker cookbook Rubric lazily.
        try:
            from tinker_cookbook.recipes.rubric.data import Rubric  # noqa: PLC0415
            self._rubric_cls = Rubric
        except ImportError:
            logger.warning(
                "tinker_cookbook.recipes.rubric not available — "
                "LLMJudgeReward will return 0.0 for all completions."
            )
            self._rubric_cls = None

    def _build_grader_prompt(
        self, prompt: str, completion: str, rubric_str: str
    ) -> str:
        """Build the grader prompt string.

        Args:
            prompt: The original question/prompt.
            completion: The model's generated answer.
            rubric_str: The rubric criteria for scoring.

        Returns:
            A formatted prompt string for the grader LLM.
        """
        return (
            "I will show you a question, a model's answer, and a rubric.\n"
            "Please grade the answer based on the rubric.\n\n"
            f"<question>\n{prompt}\n</question>\n\n"
            f"<answer>\n{completion}\n</answer>\n\n"
            f"<rubric>\n{rubric_str}\n</rubric>\n\n"
            "Please grade the answer based on the rubric. "
            "Output your score between 0 and 1 wrapped in <score>...</score>"
        )

    async def _grade_async(self, prompt: str, completion: str) -> float:
        """Call the grader LLM asynchronously and extract the score.

        Args:
            prompt: The original question.
            completion: The model's answer to grade.

        Returns:
            A float score between 0.0 and 1.0.
        """
        if self._rubric_cls is None:
            return 0.0

        import tinker  # noqa: PLC0415

        rubric_str = self._prompt_rubrics.get(prompt, self._default_rubric_str)
        grader_prompt_text = self._build_grader_prompt(
            prompt, completion, rubric_str
        )

        try:
            # Tokenize the grader prompt.
            tokenizer = self._grader_client.get_tokenizer()
            tokens = tokenizer.encode(grader_prompt_text)
            model_input = tinker.ModelInput.from_ints(tokens=tokens)

            sampling_params = tinker.SamplingParams(
                max_tokens=64,
                temperature=0.0,  # deterministic grading
            )

            result = await self._grader_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )

            grader_response = tokenizer.decode(result.sequences[0].tokens)

            # Extract score from <score>...</score> tags.
            match = re.search(r"<score>(.*?)</score>", grader_response, re.DOTALL)
            if match:
                score = float(match.group(1).strip())
                return sanitize_reward(max(0.0, min(1.0, score)))
            else:
                logger.warning(
                    "Grader did not return a score in <score> tags. "
                    "Response: %s",
                    grader_response[:200],
                )
                return 0.0

        except Exception:
            logger.error(
                "LLM grader call failed for prompt: %s",
                prompt[:100],
                exc_info=True,
            )
            return 0.0

    def __call__(self, prompt: str, completion: str) -> float:
        """Grade the completion using the LLM judge.

        When called from inside an async context (which is always the case
        in the training loop), schedules the async grader call as a task
        and returns 0.0 immediately — the actual score is retrieved via
        ``grade_async`` which the training loop should call directly.

        For synchronous use, falls back to creating a new event loop.

        Args:
            prompt: The input prompt.
            completion: The model's generated completion.

        Returns:
            A float score between 0.0 and 1.0, or 0.0 on failure.
        """
        try:
            # Try to get the running loop.
            loop = asyncio.get_running_loop()
            # We're inside an async context — create a task and return 0.0.
            # The training loop should call grade_async() directly instead.
            logger.warning(
                "LLMJudgeReward.__call__ invoked from async context. "
                "Use await reward_fn.grade_async(prompt, completion) instead."
            )
            return 0.0
        except RuntimeError:
            # No running loop — safe to use asyncio.run.
            try:
                return asyncio.run(self._grade_async(prompt, completion))
            except Exception:
                logger.error("LLMJudgeReward failed", exc_info=True)
                return 0.0

    async def grade_async(self, prompt: str, completion: str) -> float:
        """Async version of the reward function for use in the training loop.

        Args:
            prompt: The input prompt.
            completion: The model's generated completion.

        Returns:
            A float score between 0.0 and 1.0.
        """
        return await self._grade_async(prompt, completion)


def sanitize_reward(value: float) -> float:
    """Replace non-finite reward values with 0.0.

    Args:
        value: The raw reward value to sanitize.

    Returns:
        ``0.0`` if ``value`` is NaN, +∞, or −∞; otherwise returns ``value``
        unchanged.
    """
    return 0.0 if not math.isfinite(value) else value
