"""Tests for qwen3_rl_pipeline/training/rewards.py.

Covers property-based tests (Properties 3, 4, 5) and unit tests for all
public symbols: ``sanitize_reward``, ``ExactMatchReward``, and ``FormatReward``.
"""

from __future__ import annotations

import math
import re

import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import from_regex, just, one_of, text

from qwen3_rl_pipeline.training.rewards import (
    ExactMatchReward,
    FormatReward,
    sanitize_reward,
)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


# Feature: qwen3-rl-finetuning-pipeline, Property 3: Non-finite rewards are sanitized to zero
@given(one_of(just(float("nan")), just(float("inf")), just(float("-inf"))))
def test_nonfinite_reward_sanitized(bad_value: float) -> None:
    """Validates: Requirements 5.4"""
    result = sanitize_reward(bad_value)
    assert result == 0.0
    assert math.isfinite(result)


# Feature: qwen3-rl-finetuning-pipeline, Property 4: ExactMatchReward is binary and correct
@given(reference=text(), completion=text())
@settings(deadline=None)
def test_exact_match_reward_binary(reference: str, completion: str) -> None:
    """Validates: Requirements 5.1, 5.2"""
    reward = ExactMatchReward(reference)(prompt="", completion=completion)
    assert reward in (0.0, 1.0)
    assert math.isfinite(reward)
    assert reward == (1.0 if completion == reference else 0.0)


# Feature: qwen3-rl-finetuning-pipeline, Property 5: FormatReward is binary and correct
@given(pattern=from_regex(r"[a-z]+"), completion=text())
def test_format_reward_binary(pattern: str, completion: str) -> None:
    """Validates: Requirements 5.1, 5.3"""
    # Skip patterns that are not valid regexes (can occur with some Hypothesis
    # versions where from_regex generates strings containing metacharacters).
    try:
        re.compile(pattern)
    except re.PatternError:
        assume(False)
    reward = FormatReward(pattern)(prompt="", completion=completion)
    assert reward in (0.0, 1.0)
    assert math.isfinite(reward)
    expected = 1.0 if re.search(pattern, completion) else 0.0
    assert reward == expected


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_sanitize_reward_passes_finite_values() -> None:
    """Finite floats pass through sanitize_reward unchanged."""
    for value in (0.0, 1.0, -1.0, 0.5, 100.0, -0.001):
        assert sanitize_reward(value) == value


def test_exact_match_reward_identical_strings() -> None:
    """ExactMatchReward returns 1.0 when completion equals the reference."""
    reward = ExactMatchReward("hello world")
    assert reward(prompt="", completion="hello world") == 1.0


def test_exact_match_reward_different_strings() -> None:
    """ExactMatchReward returns 0.0 when completion differs from the reference."""
    reward = ExactMatchReward("hello world")
    assert reward(prompt="", completion="Hello World") == 0.0


def test_format_reward_matching_completion() -> None:
    """FormatReward returns 1.0 when the completion matches the pattern."""
    reward = FormatReward(r"\d+")
    assert reward(prompt="", completion="The answer is 42") == 1.0


def test_format_reward_nonmatching_completion() -> None:
    """FormatReward returns 0.0 when the completion does not match the pattern."""
    reward = FormatReward(r"\d+")
    assert reward(prompt="", completion="No numbers here") == 0.0
