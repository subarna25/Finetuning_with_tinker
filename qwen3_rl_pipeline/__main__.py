"""Entry point for ``python -m qwen3_rl_pipeline``.

Running ``python -m qwen3_rl_pipeline`` delegates to :func:`run_cli` and
wraps the entire execution in a top-level exception handler that logs any
unhandled error at CRITICAL level before exiting with code 1.
"""

from __future__ import annotations

import asyncio
import logging
import sys

logger = logging.getLogger("qwen3_rl_pipeline")


async def main() -> None:
    """Async entry point — delegates to the CLI runner.

    Imports :func:`~qwen3_rl_pipeline.cli.run_cli` lazily so that the
    module can be imported without triggering all CLI-level imports.
    """
    from qwen3_rl_pipeline.cli import run_cli

    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        sys.exit(1)
