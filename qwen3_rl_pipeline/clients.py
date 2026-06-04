"""Tinker client factory for the Qwen3 RL fine-tuning pipeline.

This module constructs and returns all three Tinker API clients needed by the
training loop. The ``tinker`` package is an optional runtime dependency; it is
imported lazily inside ``initialize_clients`` so that the module can be imported
cleanly in environments where ``tinker`` is not installed (e.g. during unit
testing).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tinker  # noqa: F401 — used only for type annotations

logger = logging.getLogger("qwen3_rl_pipeline")


@dataclass
class TinkerClients:
    """Container for the three Tinker API client objects.

    Attributes:
        service_client: The top-level ``tinker.ServiceClient`` instance.
        training_client: A LoRA training client created from ``service_client``.
        sampling_client: A sampling client created from ``service_client``.
    """

    service_client: Any
    training_client: Any
    sampling_client: Any


def initialize_clients(
    base_model: str = "Qwen/Qwen3-8B",
    rank: int = 32,
) -> TinkerClients:
    """Construct and return all Tinker API clients (sync version).

    Use ``initialize_clients_async`` when calling from an async context.
    """
    api_key = os.environ.get("TINKER_API_KEY", "")
    if not api_key.strip():
        raise EnvironmentError(
            "TINKER_API_KEY environment variable is not set or empty"
        )

    logger.info("Initializing Tinker clients...")

    try:
        import tinker  # noqa: PLC0415

        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=rank,
        )
        sampling_client = service_client.create_sampling_client(
            base_model=base_model,
        )
    except Exception:
        logger.error("Failed to initialize Tinker clients", exc_info=True)
        raise

    logger.info("Tinker clients initialized successfully")
    return TinkerClients(
        service_client=service_client,
        training_client=training_client,
        sampling_client=sampling_client,
    )


async def initialize_clients_async(
    base_model: str = "Qwen/Qwen3-8B",
    rank: int = 32,
) -> TinkerClients:
    """Construct and return all Tinker API clients from an async context.

    Uses the async variants of the Tinker SDK methods to avoid deadlocks.

    Args:
        base_model: HuggingFace model identifier. Defaults to ``"Qwen/Qwen3-8B"``.
        rank: LoRA rank. Defaults to ``32``.

    Returns:
        A :class:`TinkerClients` instance.

    Raises:
        EnvironmentError: If ``TINKER_API_KEY`` is not set or empty.
        Exception: Any SDK exception is logged at ERROR and re-raised.
    """
    api_key = os.environ.get("TINKER_API_KEY", "")
    if not api_key.strip():
        raise EnvironmentError(
            "TINKER_API_KEY environment variable is not set or empty"
        )

    logger.info("Initializing Tinker clients...")

    try:
        import tinker  # noqa: PLC0415

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
        )
        sampling_client = await service_client.create_sampling_client_async(
            base_model=base_model,
        )
    except Exception:
        logger.error("Failed to initialize Tinker clients", exc_info=True)
        raise

    logger.info("Tinker clients initialized successfully")
    return TinkerClients(
        service_client=service_client,
        training_client=training_client,
        sampling_client=sampling_client,
    )
