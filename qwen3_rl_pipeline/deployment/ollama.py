"""Ollama Modelfile generation and model deployment utilities.

This module provides four functions:

- :func:`generate_modelfile`: pure function that renders a Modelfile string
  from a :class:`~qwen3_rl_pipeline.config.ModelfileConfig`.
- :func:`parse_modelfile`: parses a Modelfile string back into a
  :class:`~qwen3_rl_pipeline.config.ModelfileConfig` (inverse of
  ``generate_modelfile`` for the round-trip property).
- :func:`write_modelfile`: writes a Modelfile string to disk.
- :func:`register_model`: invokes ``ollama create`` via ``subprocess.run``
  to register the model with the local Ollama daemon.

All functions raise :class:`~qwen3_rl_pipeline.exceptions.DeploymentError`
on failure and log progress at INFO level.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from qwen3_rl_pipeline.config import ModelfileConfig
from qwen3_rl_pipeline.exceptions import DeploymentError

logger = logging.getLogger("qwen3_rl_pipeline")


def generate_modelfile(config: ModelfileConfig) -> str:
    """Render a Modelfile string from a ModelfileConfig.

    This is a pure function — it performs no I/O and always produces the
    same output for the same input.  The output format is::

        FROM /absolute/path/to/model.gguf
        PARAMETER temperature 0.7
        PARAMETER top_p 0.9
        PARAMETER num_ctx 4096
        SYSTEM \"\"\"<system_prompt>\"\"\"

    ``PARAMETER`` directives are written in the fixed order ``temperature``,
    ``top_p``, ``num_ctx``.  The ``SYSTEM`` directive is omitted when
    ``config.system_prompt`` is ``None``.

    Args:
        config: Modelfile configuration dataclass.

    Returns:
        A Modelfile string with a trailing newline.
    """
    lines: list[str] = [
        f"FROM {config.gguf_path}",
        f"PARAMETER temperature {config.temperature}",
        f"PARAMETER top_p {config.top_p}",
        f"PARAMETER num_ctx {config.num_ctx}",
    ]
    if config.system_prompt is not None:
        lines.append(f'SYSTEM """{config.system_prompt}"""')
    return "\n".join(lines) + "\n"


def parse_modelfile(content: str) -> ModelfileConfig:
    """Parse a Modelfile string back into a ModelfileConfig.

    This function is the inverse of :func:`generate_modelfile`: for any
    config produced by ``generate_modelfile``, calling
    ``generate_modelfile(parse_modelfile(content))`` returns the same
    string.

    Parsed fields:

    - ``FROM <path>`` → ``gguf_path``
    - ``PARAMETER temperature <value>`` → ``temperature`` (float)
    - ``PARAMETER top_p <value>`` → ``top_p`` (float)
    - ``PARAMETER num_ctx <value>`` → ``num_ctx`` (int)
    - ``SYSTEM \"\"\"<content>\"\"\"`` → ``system_prompt`` (str or None)

    The ``model_name`` field cannot be recovered from a Modelfile; it is
    set to ``""`` as a placeholder.

    Args:
        content: A Modelfile string as produced by :func:`generate_modelfile`.

    Returns:
        A :class:`ModelfileConfig` populated from the parsed directives.
    """
    gguf_path: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096
    system_prompt: str | None = None

    # Extract SYSTEM block first (may span multiple lines) using a regex
    # that matches SYSTEM """...""" across newlines.
    system_match = re.search(r'^SYSTEM """(.*?)"""', content, re.DOTALL | re.MULTILINE)
    if system_match:
        system_prompt = system_match.group(1)

    for line in content.split("\n"):
        # Use the raw line (no strip) so that paths or prompts containing
        # Unicode whitespace characters are preserved exactly as written.
        if line.startswith("FROM "):
            gguf_path = line[len("FROM "):]
        elif line.startswith("PARAMETER temperature "):
            temperature = float(line[len("PARAMETER temperature "):])
        elif line.startswith("PARAMETER top_p "):
            top_p = float(line[len("PARAMETER top_p "):])
        elif line.startswith("PARAMETER num_ctx "):
            num_ctx = int(line[len("PARAMETER num_ctx "):])

    return ModelfileConfig(
        gguf_path=gguf_path,
        model_name="",
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        system_prompt=system_prompt,
    )


def write_modelfile(content: str, path: str) -> None:
    """Write a Modelfile string to disk.

    Args:
        content: The Modelfile string to write.
        path: Destination filesystem path for the Modelfile.
    """
    Path(path).write_text(content, encoding="utf-8")


def register_model(model_name: str, modelfile_path: str) -> None:
    """Register a model with the local Ollama daemon.

    Invokes ``ollama create <model_name> -f <modelfile_path>`` via
    ``subprocess.run`` and raises :class:`DeploymentError` if the process
    exits with a non-zero return code.

    Args:
        model_name: The Ollama model name to register (e.g. ``"my-model"``).
        modelfile_path: Filesystem path to the Modelfile to use.

    Raises:
        DeploymentError: If ``ollama create`` exits with a non-zero return
            code.  The captured stderr is included in the exception message.
    """
    logger.info("Registering Ollama model %s...", model_name)
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DeploymentError(f"ollama create failed:\n{result.stderr}")
    logger.info(
        "Model %s registered from %s", model_name, modelfile_path
    )
