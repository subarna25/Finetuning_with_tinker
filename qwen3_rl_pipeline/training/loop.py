"""Async RL training loop for the Qwen3 fine-tuning pipeline.

Supports two loss functions from the Tinker cookbook:

- ``cross_entropy``: reward-weighted SFT loss. Simple and reliable.
  Fields: ``target_tokens``, ``weights`` (loss mask with reward weight).

- ``importance_sampling``: proper RL loss (GRPO-style). Uses logprobs from
  the sampling policy to compute importance weights.
  Fields: ``target_tokens``, ``logprobs``, ``advantages``, ``mask``.

The ``tinker`` package is imported lazily so this module can be imported
cleanly in environments where it is not installed (e.g. unit testing).

JWT token expiry is handled automatically — if an AuthenticationError is
raised mid-training, the clients are reinitialized and the step is retried
once before being skipped.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

from qwen3_rl_pipeline.config import TrainingConfig
from qwen3_rl_pipeline.data.models import DatasetRecord
from qwen3_rl_pipeline.training.checkpointing import Checkpointer
from qwen3_rl_pipeline.training.rewards import RewardFunction, sanitize_reward

if TYPE_CHECKING:
    from qwen3_rl_pipeline.clients import TinkerClients

logger = logging.getLogger("qwen3_rl_pipeline")

LossFn = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]


def _build_cross_entropy_datum(
    tinker: object,
    TensorData: type,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    advantage: float,
) -> object:
    """Build a Datum for cross-entropy (reward-weighted SFT) loss.

    Uses ``weights`` field as the loss mask, scaled by the advantage.
    This is the simplest RL loss — equivalent to SFT with reward weighting.

    Args:
        tinker: The tinker module.
        TensorData: The TensorData class.
        prompt_tokens: Tokenized prompt.
        completion_tokens: Tokenized completion.
        advantage: GRPO advantage for this completion.

    Returns:
        A tinker.Datum ready for forward_backward with cross_entropy loss.
    """
    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    seq_len = len(input_tokens)
    prompt_len = len(prompt_tokens)

    # Reward weight: clamp advantage to [0, 1] range.
    reward_weight = max(0.0, min(1.0, 0.5 + advantage))
    mask = (
        [0.0] * (prompt_len - 1)
        + [reward_weight] * (seq_len - (prompt_len - 1))
    )

    n = seq_len
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=target_tokens[:n],
                dtype="int64",
                shape=[n],
            ),
            "weights": TensorData(
                data=mask[:n],
                dtype="float32",
                shape=[n],
            ),
        },
    )


def _build_importance_sampling_datum(
    tinker: object,
    TensorData: type,
    torch: object,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    logprobs: list[float],
    advantage: float,
) -> object:
    """Build a Datum for importance_sampling (proper RL) loss.

    Uses the cookbook pattern exactly:
    - ``target_tokens``: right-shifted token IDs
    - ``logprobs``: per-token log-probabilities from the sampling policy
    - ``advantages``: per-token advantage (broadcast from scalar)
    - ``mask``: 0 for prompt tokens, 1 for completion tokens

    This matches the Tinker cookbook's ``trajectory_to_data`` function.

    Args:
        tinker: The tinker module.
        TensorData: The TensorData class.
        torch: The torch module.
        prompt_tokens: Tokenized prompt.
        completion_tokens: Tokenized completion.
        logprobs: Per-token log-probabilities from the on-policy sampler.
        advantage: GRPO advantage for this completion.

    Returns:
        A tinker.Datum ready for forward_backward with importance_sampling loss.
    """
    all_tokens = prompt_tokens + completion_tokens
    # Right-shift: input = all_tokens[:-1], targets = all_tokens[1:]
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    seq_len = len(input_tokens)
    prompt_len = len(prompt_tokens)

    # Mask: 0 for prompt positions, 1 for completion positions.
    # Shifted by 1 because of the right-shift (cookbook pattern).
    mask = [0.0] * (prompt_len - 1) + [1.0] * (seq_len - (prompt_len - 1))

    # Per-token advantages: broadcast scalar to completion length, 0 for prompt.
    adv = [0.0] * (prompt_len - 1) + [advantage] * (seq_len - (prompt_len - 1))

    # Logprobs: pad prompt positions with 0, use completion logprobs.
    # The sampler returns logprobs for completion tokens only.
    padded_logprobs = [0.0] * (prompt_len - 1) + logprobs[: seq_len - (prompt_len - 1)]

    n = seq_len
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(target_tokens[:n], dtype=torch.long)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(padded_logprobs[:n], dtype=torch.float32)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(adv[:n], dtype=torch.float32)
            ),
            "mask": TensorData.from_torch(
                torch.tensor(mask[:n], dtype=torch.float32)
            ),
        },
    )


async def _reinitialize_clients(
    clients: "TinkerClients",
    base_model: str = "Qwen/Qwen3-8B",
    rank: int = 32,
) -> "TinkerClients":
    """Reinitialize Tinker clients after a JWT expiry.

    Creates a fresh ServiceClient (which reads TINKER_API_KEY from env)
    and rebuilds the training and sampling clients.

    Args:
        clients: The expired TinkerClients instance (used for config only).
        base_model: HuggingFace model ID for the base model.
        rank: LoRA rank for the training client.

    Returns:
        A new TinkerClients instance with fresh authentication.
    """
    import tinker  # noqa: PLC0415
    from qwen3_rl_pipeline.clients import TinkerClients  # noqa: PLC0415

    logger.warning(
        "JWT token expired — reinitializing Tinker clients with fresh token..."
    )
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model,
        rank=rank,
    )
    sampling_client = await service_client.create_sampling_client_async(
        base_model=base_model,
    )
    logger.info("Tinker clients reinitialized successfully.")
    return TinkerClients(
        service_client=service_client,
        training_client=training_client,
        sampling_client=sampling_client,
    )


async def run_training_loop(
    config: TrainingConfig,
    dataset: list[DatasetRecord],
    reward_fn: RewardFunction,
    clients: "TinkerClients",
    checkpointer: Checkpointer,
    loss_fn: LossFn = "cross_entropy",
) -> str:
    """Run the RL training loop.

    Supports two loss functions:
    - ``cross_entropy``: reward-weighted SFT (default, most stable)
    - ``importance_sampling``: proper RL loss using logprobs (cookbook pattern)

    For each step:
    1. Get an on-policy SamplingClient.
    2. Sample completions, score with reward_fn, compute GRPO advantages.
    3. Build Datum objects using the selected loss function schema.
    4. Call forward_backward_async + optim_step_async.
    5. Log metrics and checkpoint.
    6. After all epochs, save final weights and return the Tinker path.

    Args:
        config: Frozen training configuration dataclass.
        dataset: List of dataset records to train on.
        reward_fn: Callable that scores a (prompt, completion) pair.
        clients: Container holding the Tinker training and sampling clients.
        checkpointer: Manages periodic checkpoint writes.
        loss_fn: Loss function to use. One of ``cross_entropy``,
            ``importance_sampling``, ``ppo``, ``cispo``, ``dro``.
            Defaults to ``cross_entropy``.

    Returns:
        Real Tinker path string (tinker://...) for the final adapter weights.
    """
    import tinker  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from tinker import TensorData  # noqa: PLC0415

    logger.info("Training with loss_fn=%s", loss_fn)

    # For importance_sampling we need logprobs from the sampler.
    needs_logprobs = loss_fn in ("importance_sampling", "ppo", "cispo", "dro")

    pairs: list[tuple[str, str | None]] = [
        (record.prompt, record.reference_answer) for record in dataset
    ]

    batch_size = config.num_samples_per_group
    step = 0
    start_time = time.monotonic()
    tokenizer = clients.training_client.get_tokenizer()

    for epoch in range(config.epochs):
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]

            try:
                # ── Step 1: on-policy sampler ──────────────────────────────
                on_policy_sc = (
                    clients.training_client.save_weights_and_get_sampling_client()
                )

                sampling_params = tinker.SamplingParams(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                # ── Step 2: rollouts ───────────────────────────────────────
                training_data: list = []
                rewards: list[float] = []

                for prompt, _ref in batch:
                    prompt_tokens = tokenizer.encode(prompt)
                    model_input = tinker.ModelInput.from_ints(tokens=prompt_tokens)

                    sample_result = await on_policy_sc.sample_async(
                        prompt=model_input,
                        num_samples=config.num_samples_per_group,
                        sampling_params=sampling_params,
                    )

                    # Score each completion.
                    group_rewards: list[float] = []
                    for seq in sample_result.sequences:
                        completion = tokenizer.decode(seq.tokens)
                        # Use async grading if the reward function supports it
                        # (e.g. LLMJudgeReward).
                        if hasattr(reward_fn, "grade_async"):
                            raw = await reward_fn.grade_async(prompt, completion)
                        else:
                            raw = reward_fn(prompt, completion)
                        group_rewards.append(sanitize_reward(raw))

                    rewards.extend(group_rewards)

                    # GRPO advantages: reward - group mean.
                    group_mean = (
                        sum(group_rewards) / len(group_rewards)
                        if group_rewards else 0.0
                    )
                    advantages = [r - group_mean for r in group_rewards]

                    # Build Datum per completion.
                    for seq, advantage in zip(sample_result.sequences, advantages):
                        completion_tokens = list(seq.tokens)

                        if needs_logprobs:
                            # Get per-token logprobs via compute_logprobs_async.
                            # Build the full sequence (prompt + completion) as input.
                            full_tokens = prompt_tokens + completion_tokens
                            full_input = tinker.ModelInput.from_ints(tokens=full_tokens)
                            try:
                                lp_result = await on_policy_sc.compute_logprobs_async(
                                    full_input
                                )
                                # logprobs covers all tokens; we only need completion portion.
                                all_lp = [float(x) if x is not None else 0.0
                                          for x in lp_result]
                                # Completion logprobs start after prompt tokens.
                                lp = all_lp[len(prompt_tokens):]
                            except Exception:
                                logger.warning(
                                    "compute_logprobs_async failed, using zeros",
                                    exc_info=True,
                                )
                                lp = [0.0] * len(completion_tokens)

                            datum = _build_importance_sampling_datum(
                                tinker, TensorData, torch,
                                prompt_tokens, completion_tokens, lp, advantage,
                            )
                        else:
                            datum = _build_cross_entropy_datum(
                                tinker, TensorData,
                                prompt_tokens, completion_tokens, advantage,
                            )

                        training_data.append(datum)

                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

                # ── Step 3: forward-backward + optimiser ───────────────────
                if training_data:
                    fb_future = await clients.training_client.forward_backward_async(
                        data=training_data,
                        loss_fn=loss_fn,
                    )
                    fwd_result = await fb_future.result_async()

                    # Extract loss value — it's in loss_fn_outputs as TensorData,
                    # or in metrics dict depending on the Tinker version.
                    loss_val = 0.0
                    try:
                        if fwd_result.metrics:
                            # Tinker returns 'loss:sum' (total loss over all tokens)
                            # and 'clock_cycle:unique'. Normalize by batch size.
                            if "loss:sum" in fwd_result.metrics:
                                loss_val = float(fwd_result.metrics["loss:sum"]) / max(1, len(training_data))
                            else:
                                # Fallback: try other common key names.
                                for key in ("loss", "mean_loss", "total_loss", "ce_loss"):
                                    if key in fwd_result.metrics:
                                        loss_val = float(fwd_result.metrics[key])
                                        break
                    except Exception:
                        loss_val = 0.0

                    # Remove debug log after first step confirmed metrics structure.

                    optim_future = await clients.training_client.optim_step_async(
                        tinker.AdamParams(learning_rate=config.learning_rate)
                    )
                    await optim_future.result_async()

                # ── Step 4: logging ────────────────────────────────────────
                logger.info(
                    "step=%d epoch=%d loss_fn=%s mean_reward=%.4f loss=%.4f",
                    step, epoch, loss_fn, mean_reward,
                    loss_val if training_data else 0.0,
                )

                # Write metrics to JSONL file for visualization.
                import json, os  # noqa: PLC0415, E401
                metrics_file = os.path.join(config.output_dir, "training_metrics.jsonl")
                os.makedirs(config.output_dir, exist_ok=True)
                try:
                    with open(metrics_file, "a") as mf:
                        mf.write(json.dumps({
                            "step": step,
                            "epoch": epoch,
                            "mean_reward": float(mean_reward),
                            "loss": float(loss_val) if training_data else 0.0,
                            "loss_fn": loss_fn,
                        }) + "\n")
                except Exception:
                    logger.warning("Failed to write metrics to file", exc_info=True)

                if step > 0 and step % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    total_steps = config.epochs * max(
                        1, (len(pairs) + batch_size - 1) // batch_size
                    )
                    eta = (elapsed / (step + 1)) * (total_steps - step - 1)
                    logger.info(
                        "Progress: step=%d elapsed=%.1fs eta=%.1fs",
                        step, elapsed, eta,
                    )

                # ── Step 5: checkpoint ─────────────────────────────────────
                checkpointer.maybe_save(step, epoch, "pending-final-save")

            except Exception as exc:
                # Check if this is a JWT expiry — if so, reinitialize and retry once.
                exc_type = type(exc).__name__
                is_auth_error = (
                    "AuthenticationError" in exc_type
                    or "401" in str(exc)
                    or "Invalid JWT" in str(exc)
                )
                if is_auth_error:
                    logger.warning(
                        "JWT token expired at step=%d epoch=%d — "
                        "reinitializing clients and retrying step...",
                        step, epoch,
                    )
                    try:
                        clients = await _reinitialize_clients(
                            clients,
                            base_model="Qwen/Qwen3-8B",
                            rank=config.lora_rank,
                        )
                        tokenizer = clients.training_client.get_tokenizer()
                        logger.info("Retry after token refresh will happen on next step.")
                    except Exception as reinit_exc:
                        logger.error(
                            "Failed to reinitialize clients: %s — "
                            "skipping step=%d",
                            reinit_exc, step,
                        )
                else:
                    logger.error(
                        "Error at step=%d epoch=%d — skipping step",
                        step, epoch, exc_info=True,
                    )

            step += 1

    # ── Final: save weights with no expiry ────────────────────────────────
    logger.info("Saving final weights to Tinker (no expiry)...")
    save_future = clients.training_client.save_weights_for_sampler(
        name="final",
        ttl_seconds=None,  # Never expires
    )
    save_result = save_future.result()
    weights_path: str = save_result.path
    logger.info("Final weights saved to: %s", weights_path)
    return weights_path
