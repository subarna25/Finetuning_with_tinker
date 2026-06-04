"""Data model for a single training example in the RL fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRecord:
    """A single training example with a prompt and an optional reference answer.

    Attributes:
        prompt: The input text presented to the model during training.
        reference_answer: An optional ground-truth answer used by reward
            functions such as ``ExactMatchReward``. When ``None``, the record
            carries no reference and reward functions that require one must
            handle the absence themselves.
    """

    prompt: str
    reference_answer: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Serialize the record to a plain dictionary.

        The ``"prompt"`` key is always present. The ``"reference_answer"`` key
        is included only when :attr:`reference_answer` is not ``None``.

        Returns:
            A dictionary with at minimum ``{"prompt": self.prompt}``, and
            optionally ``{"reference_answer": self.reference_answer}`` when
            the field is set.
        """
        result: dict[str, str] = {"prompt": self.prompt}
        if self.reference_answer is not None:
            result["reference_answer"] = self.reference_answer
        return result

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> DatasetRecord:
        """Construct a ``DatasetRecord`` from a plain dictionary.

        Reads ``"prompt"`` (required) and ``"reference_answer"`` (optional)
        from *d*. Any other keys in the dictionary are silently ignored.

        Args:
            d: A dictionary that must contain a ``"prompt"`` key whose value
                is a string. The ``"reference_answer"`` key is optional; when
                present its value is used as-is.

        Returns:
            A new :class:`DatasetRecord` populated from the dictionary.

        Raises:
            KeyError: If the ``"prompt"`` key is absent from *d*.
            TypeError: If the value of ``"prompt"`` or ``"reference_answer"``
                is not a string (or ``None`` for ``reference_answer``).
        """
        prompt = d["prompt"]
        if not isinstance(prompt, str):
            raise TypeError(
                f"'prompt' must be a str, got {type(prompt).__name__!r}"
            )
        raw_ref = d.get("reference_answer")
        if raw_ref is not None and not isinstance(raw_ref, str):
            raise TypeError(
                f"'reference_answer' must be a str or None, "
                f"got {type(raw_ref).__name__!r}"
            )
        reference_answer: str | None = raw_ref  # type: ignore[assignment]
        return cls(prompt=prompt, reference_answer=reference_answer)
