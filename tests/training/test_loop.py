"""Unit tests for qwen3_rl_pipeline/training/loop.py.

All Tinker API calls are mocked with ``unittest.mock.AsyncMock`` so that
these tests run without ``tinker`` or ``tinker_cookbook`` installed.
The ``tinker_cookbook`` import is patched via ``sys.modules`` before the
module under test is imported.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch tinker_cookbook and tinker into sys.modules before importing loop.py
# so that any lazy import inside run_training_loop resolves to our stubs.
# ---------------------------------------------------------------------------

_tinker_cookbook_stub = types.ModuleType("tinker_cookbook")
_tinker_cookbook_rl_stub = types.ModuleType("tinker_cookbook.rl")
_tinker_cookbook_stub.rl = _tinker_cookbook_rl_stub
sys.modules.setdefault("tinker_cookbook", _tinker_cookbook_stub)
sys.modules.setdefault("tinker_cookbook.rl", _tinker_cookbook_rl_stub)

_tinker_stub = types.ModuleType("tinker")
# Add the types that loop.py uses directly from the tinker module.
_tinker_stub.ModelInput = MagicMock(name="ModelInput")  # type: ignore[attr-defined]
_tinker_stub.SamplingParams = MagicMock(name="SamplingParams")  # type: ignore[attr-defined]
_tinker_stub.AdamParams = MagicMock(name="AdamParams")  # type: ignore[attr-defined]
_tinker_stub.Datum = MagicMock(name="Datum")  # type: ignore[attr-defined]
_tinker_stub.TensorData = MagicMock(name="TensorData")  # type: ignore[attr-defined]
_tinker_stub.EncodedTextChunk = MagicMock(name="EncodedTextChunk")  # type: ignore[attr-defined]
sys.modules.setdefault("tinker", _tinker_stub)

# Now it is safe to import the module under test.
from qwen3_rl_pipeline.clients import TinkerClients  # noqa: E402
from qwen3_rl_pipeline.config import TrainingConfig  # noqa: E402
from qwen3_rl_pipeline.data.models import DatasetRecord  # noqa: E402
from qwen3_rl_pipeline.training.checkpointing import Checkpointer  # noqa: E402
from qwen3_rl_pipeline.training.loop import run_training_loop  # noqa: E402
from qwen3_rl_pipeline.training.rewards import ExactMatchReward  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_result(tokens: list[int] | None = None) -> MagicMock:
    """Return a mock sample result whose tokens decode to 'test completion'."""
    result = MagicMock()
    seq = MagicMock()
    seq.tokens = tokens if tokens is not None else [1, 2, 3]
    seq.logprobs = [-0.5, -0.3, -0.4]
    result.sequences = [seq]
    return result


def _make_clients(
    sample_result: MagicMock | None = None,
    fb_side_effect: Exception | None = None,
) -> TinkerClients:
    """Build a ``TinkerClients`` instance with all async methods mocked."""
    if sample_result is None:
        sample_result = _make_sample_result()

    on_policy_sc = MagicMock()
    on_policy_sc.sample_async = AsyncMock(return_value=sample_result)

    # Mock forward_backward future with result_async method.
    fb_future = MagicMock()
    fb_future.result_async = AsyncMock(return_value=MagicMock(loss=0.5))

    # Mock optim future with result_async method.
    optim_future = MagicMock()
    optim_future.result_async = AsyncMock(return_value=MagicMock())

    training_client = MagicMock()
    training_client.save_weights_and_get_sampling_client = MagicMock(
        return_value=on_policy_sc
    )
    # Mock get_tokenizer to return a tokenizer that encodes/decodes strings.
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "test completion"
    training_client.get_tokenizer = MagicMock(return_value=mock_tokenizer)

    if fb_side_effect is not None:
        training_client.forward_backward_async = AsyncMock(
            side_effect=fb_side_effect
        )
    else:
        training_client.forward_backward_async = AsyncMock(return_value=fb_future)

    training_client.optim_step_async = AsyncMock(return_value=optim_future)

    # Mock save_weights_for_sampler to return a real-looking Tinker path.
    save_weights_result = MagicMock()
    save_weights_result.path = "tinker://test-session/sampler_weights/final"
    save_weights_future = MagicMock()
    save_weights_future.result.return_value = save_weights_result
    training_client.save_weights_for_sampler = MagicMock(
        return_value=save_weights_future
    )

    sampling_client = MagicMock()

    return TinkerClients(
        service_client=MagicMock(),
        training_client=training_client,
        sampling_client=sampling_client,
    )


def _make_config(**overrides: object) -> TrainingConfig:
    """Return a minimal ``TrainingConfig`` suitable for unit tests."""
    defaults: dict[str, object] = {
        "dataset_path": "/tmp/data.jsonl",
        "output_dir": "/tmp/output",
        "model_name": "test-model",
        "epochs": 1,
        "num_samples_per_group": 1,
        "steps_per_checkpoint": 1,
        "learning_rate": 1e-4,
        "max_tokens": 64,
        "temperature": 1.0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def _make_dataset(n: int = 1) -> list[DatasetRecord]:
    """Return a list of ``n`` simple dataset records."""
    return [
        DatasetRecord(prompt=f"prompt_{i}", reference_answer=f"answer_{i}")
        for i in range(n)
    ]


def _make_checkpointer(tmp_path: object) -> Checkpointer:
    """Return a ``Checkpointer`` that saves every step."""
    return Checkpointer(str(tmp_path), save_every=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_weights_called_once_per_step(tmp_path: object) -> None:
    """save_weights_and_get_sampling_client is called exactly once per step.

    With 1 epoch and 1 record (batch_size=1) there is exactly 1 step.
    """
    clients = _make_clients()
    config = _make_config(epochs=1, num_samples_per_group=1)
    dataset = _make_dataset(1)
    reward_fn = ExactMatchReward("answer_0")
    checkpointer = _make_checkpointer(tmp_path)

    await run_training_loop(config, dataset, reward_fn, clients, checkpointer)

    clients.training_client.save_weights_and_get_sampling_client.assert_called_once_with()


@pytest.mark.asyncio
async def test_reward_fn_called_for_each_completion(tmp_path: object) -> None:
    """reward_fn is called once for each record in the dataset."""
    clients = _make_clients()
    config = _make_config(epochs=1, num_samples_per_group=3)
    dataset = _make_dataset(3)

    call_count = 0

    def counting_reward(prompt: str, completion: str) -> float:
        nonlocal call_count
        call_count += 1
        return 1.0

    checkpointer = _make_checkpointer(tmp_path)
    await run_training_loop(config, dataset, counting_reward, clients, checkpointer)

    # 3 records in 1 batch → reward_fn called 3 times
    assert call_count == 3


@pytest.mark.asyncio
async def test_nonfinite_reward_replaced_with_zero(
    tmp_path: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A NaN reward from reward_fn is sanitized to 0.0 in the logged mean_reward."""
    clients = _make_clients()
    config = _make_config(epochs=1, num_samples_per_group=1)
    dataset = _make_dataset(1)

    def nan_reward(prompt: str, completion: str) -> float:
        return float("nan")

    checkpointer = _make_checkpointer(tmp_path)

    with caplog.at_level(logging.INFO, logger="qwen3_rl_pipeline"):
        await run_training_loop(config, dataset, nan_reward, clients, checkpointer)

    # The logged mean_reward should be 0.0000 (sanitized from NaN)
    step_logs = [r.message for r in caplog.records if "mean_reward" in r.message]
    assert step_logs, "Expected at least one log record containing 'mean_reward'"
    assert "0.0000" in step_logs[0], (
        f"Expected mean_reward=0.0000 in log, got: {step_logs[0]!r}"
    )


@pytest.mark.asyncio
async def test_step_exception_caught_and_training_continues(
    tmp_path: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A RuntimeError from forward_backward_async is caught and training continues.

    With 2 records and batch_size=1 there are 2 steps. The first step raises;
    the second step should still execute (optim_step_async called once).
    """
    clients = _make_clients(fb_side_effect=RuntimeError("simulated failure"))
    config = _make_config(epochs=1, num_samples_per_group=1)
    dataset = _make_dataset(2)
    reward_fn = ExactMatchReward("answer_0")
    checkpointer = _make_checkpointer(tmp_path)

    with caplog.at_level(logging.ERROR, logger="qwen3_rl_pipeline"):
        # Should not raise even though forward_backward_async raises.
        await run_training_loop(config, dataset, reward_fn, clients, checkpointer)

    # At least one ERROR log should mention the skipped step.
    error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_logs, "Expected at least one ERROR log for the failed step"

    # forward_backward_async was called for both steps (both raise).
    assert clients.training_client.forward_backward_async.call_count == 2


@pytest.mark.asyncio
async def test_checkpointer_called_with_correct_step_and_epoch(
    tmp_path: object,
) -> None:
    """checkpointer.maybe_save is called with the correct step and epoch."""
    clients = _make_clients()
    config = _make_config(epochs=1, num_samples_per_group=1)
    dataset = _make_dataset(1)
    reward_fn = ExactMatchReward("answer_0")

    mock_checkpointer = MagicMock(spec=Checkpointer)

    await run_training_loop(config, dataset, reward_fn, clients, mock_checkpointer)

    mock_checkpointer.maybe_save.assert_called_once()
    call_args = mock_checkpointer.maybe_save.call_args
    assert call_args.args[0] == 0, f"Expected step=0, got {call_args.args[0]}"
    assert call_args.args[1] == 0, f"Expected epoch=0, got {call_args.args[1]}"
    assert isinstance(call_args.args[2], str), "weights_path should be a string"


@pytest.mark.asyncio
async def test_loop_returns_final_weights_path(tmp_path: object) -> None:
    """run_training_loop returns a non-empty string as the final weights path."""
    clients = _make_clients()
    config = _make_config(epochs=1, num_samples_per_group=1)
    dataset = _make_dataset(1)
    reward_fn = ExactMatchReward("answer_0")
    checkpointer = _make_checkpointer(tmp_path)

    result = await run_training_loop(config, dataset, reward_fn, clients, checkpointer)

    assert isinstance(result, str), "Return value should be a string"
    assert result, "Return value should be non-empty"
