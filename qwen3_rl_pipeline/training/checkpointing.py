"""Checkpoint dataclass and Checkpointer for persisting training state.

Checkpoints are written as JSON files named
``checkpoint_step_{step:06d}.json`` in a configurable output directory.
Write failures are caught and logged at WARNING so that training is never
interrupted by a checkpoint error.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

logger = logging.getLogger("qwen3_rl_pipeline")


@dataclasses.dataclass
class Checkpoint:
    """Snapshot of training state at a given step.

    Attributes:
        step: The global training step number at the time of the checkpoint.
        epoch: The epoch number at the time of the checkpoint.
        adapter_weights_path: Path to the saved LoRA adapter weights.
    """

    step: int
    epoch: int
    adapter_weights_path: str


class Checkpointer:
    """Manages periodic saving and loading of training checkpoints.

    Checkpoints are written as JSON files to ``output_dir`` whenever
    ``step % save_every == 0``.  The most recent checkpoint can be
    retrieved with :meth:`load_latest`.

    Example:
        >>> ck = Checkpointer("/tmp/checkpoints", save_every=100)
        >>> ck.maybe_save(step=100, epoch=0, weights_path="/tmp/weights")
        >>> latest = ck.load_latest()
    """

    def __init__(self, output_dir: str, save_every: int = 100) -> None:
        """Initialise the checkpointer.

        Args:
            output_dir: Directory where checkpoint JSON files are written.
                Created on first save if it does not already exist.
            save_every: Checkpoint is written when ``step % save_every == 0``.
                Defaults to ``100``.
        """
        self._output_dir = Path(output_dir)
        self.save_every = save_every

    def maybe_save(
        self,
        step: int,
        epoch: int,
        weights_path: str,
    ) -> None:
        """Write a checkpoint file if the step is a multiple of ``save_every``.

        The file is named ``checkpoint_step_{step:06d}.json`` and contains the
        JSON-serialized :class:`Checkpoint`.  Any write error is caught, logged
        at WARNING, and silently ignored so that training can continue.

        Args:
            step: Current global training step.
            epoch: Current epoch number.
            weights_path: Path to the adapter weights saved at this step.
        """
        if step % self.save_every != 0:
            return

        checkpoint = Checkpoint(
            step=step,
            epoch=epoch,
            adapter_weights_path=weights_path,
        )
        filename = f"checkpoint_step_{step:06d}.json"
        path = self._output_dir / filename

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dataclasses.asdict(checkpoint)))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write checkpoint %s: %s",
                path,
                exc,
            )

    def load_latest(self) -> Checkpoint | None:
        """Return the checkpoint with the highest step number, or ``None``.

        Scans ``output_dir`` for files matching ``checkpoint_step_*.json``,
        parses each as a :class:`Checkpoint`, and returns the one with the
        largest ``step`` value.

        Returns:
            The most recent :class:`Checkpoint`, or ``None`` if no checkpoint
            files exist in ``output_dir``.
        """
        candidates = list(self._output_dir.glob("checkpoint_step_*.json"))
        if not candidates:
            return None

        checkpoints: list[Checkpoint] = []
        for path in candidates:
            try:
                data = json.loads(path.read_text())
                checkpoints.append(Checkpoint(**data))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read checkpoint %s: %s", path, exc)

        if not checkpoints:
            return None

        return max(checkpoints, key=lambda c: c.step)
