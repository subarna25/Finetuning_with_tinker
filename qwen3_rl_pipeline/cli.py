"""CLI entry point for the Qwen3 RL fine-tuning pipeline.

Exposes two subcommands:

- ``train``: run the full RL training loop and export the resulting model.
- ``export``: download an existing adapter and export it to Ollama.

Usage::

    python -m qwen3_rl_pipeline train \\
        --dataset data.jsonl \\
        --output-dir ./out \\
        --model-name my-model

    python -m qwen3_rl_pipeline export \\
        --adapter-path tinker://... \\
        --output-dir ./out \\
        --model-name my-model
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("qwen3_rl_pipeline")


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    Returns:
        A fully configured :class:`argparse.ArgumentParser` with ``train``
        and ``export`` subcommands attached.
    """
    parser = argparse.ArgumentParser(
        prog="qwen3_rl_pipeline",
        description="Qwen3 RL fine-tuning pipeline",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # ------------------------------------------------------------------
    # train subcommand
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Run the RL training loop and export the resulting model.",
    )
    train_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL training dataset.",
    )
    train_parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Output directory for checkpoints and artifacts.",
    )
    train_parser.add_argument(
        "--model-name",
        required=True,
        dest="model_name",
        help="Ollama model name for the final registered model.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of full passes over the dataset (default: 3).",
    )
    train_parser.add_argument(
        "--steps-per-checkpoint",
        type=int,
        default=100,
        dest="steps_per_checkpoint",
        help="Steps between checkpoint saves (default: 100).",
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        dest="lora_rank",
        help="Rank of the LoRA adapter matrices (default: 32).",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        dest="learning_rate",
        help="AdamW learning rate (default: 1e-4).",
    )
    train_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        dest="max_tokens",
        help="Maximum tokens to generate per rollout (default: 512).",
    )
    train_parser.add_argument(
        "--allow-download",
        action="store_true",
        dest="allow_download",
        help="Allow downloading the full base model (~16 GB) for GGUF conversion. "
             "Disabled by default to prevent accidental large downloads. "
             "Use 'publish-and-compare' subcommand to avoid local downloads.",
    )
    train_parser.add_argument(
        "--loss-fn",
        default="cross_entropy",
        choices=["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"],
        dest="loss_fn",
        help=(
            "Loss function for training. "
            "'cross_entropy' = reward-weighted SFT (default, most stable). "
            "'importance_sampling' = proper RL loss using logprobs (cookbook pattern). "
            "'ppo' / 'cispo' / 'dro' = advanced RL losses."
        ),
    )
    train_parser.add_argument(
        "--reward-type",
        default="substring",
        choices=["exact", "substring", "llm-judge"],
        dest="reward_type",
        help="Reward function type: 'exact' (exact match), 'substring' "
             "(case-insensitive substring, default), or 'llm-judge' "
             "(LLM-as-judge with rubric grading).",
    )
    train_parser.add_argument(
        "--grader-model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        dest="grader_model",
        help="Grader model for llm-judge reward (default: meta-llama/Llama-3.1-70B-Instruct). "
             "Other options: Qwen/Qwen3-30B-A3B-Instruct, Qwen/Qwen3-8B",
    )
    train_parser.add_argument(
        "--default-rubric",
        default=None,
        dest="default_rubric",
        help="Default rubric string for llm-judge reward. If not provided, "
             "a generic oil & gas Q&A rubric is used.",
    )
    train_parser.add_argument(
        "--llama-cpp-dir",
        default="/usr/local/bin",
        dest="llama_cpp_dir",
        help="Directory containing llama.cpp binaries (default: /usr/local/bin).",
    )

    train_parser.add_argument(
        "--quantization-type",
        default="Q4_K_M",
        dest="quantization_type",
        help="llama.cpp quantization type (default: Q4_K_M).",
    )
    export_parser = subparsers.add_parser(
        "export",
        help="Download an existing adapter and export it to Ollama.",
    )
    export_parser.add_argument(
        "--adapter-path",
        required=False,
        default=None,
        dest="adapter_path",
        help="Tinker path to the LoRA adapter (tinker://...). "
             "If omitted, --adapter-path-file must be provided.",
    )
    export_parser.add_argument(
        "--adapter-path-file",
        required=False,
        default=None,
        dest="adapter_path_file",
        help="Path to a text file containing the Tinker adapter path "
             "(written automatically by the train subcommand to "
             "<output-dir>/tinker_adapter_path.txt).",
    )
    export_parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Output directory for export artifacts.",
    )
    export_parser.add_argument(
        "--model-name",
        required=True,
        dest="model_name",
        help="Ollama model name for the final registered model.",
    )
    export_parser.add_argument(
        "--allow-download",
        action="store_true",
        dest="allow_download",
        help="Allow downloading the full base model (~16 GB) for GGUF conversion.",
    )
    export_parser.add_argument(
        "--quantization-type",
        default="Q4_K_M",
        dest="quantization_type",
        help="llama.cpp quantization type (default: Q4_K_M).",
    )
    export_parser.add_argument(
        "--llama-cpp-dir",
        default="/usr/local/bin",
        dest="llama_cpp_dir",
        help="Directory containing llama.cpp binaries (default: /usr/local/bin).",
    )

    # ------------------------------------------------------------------
    # publish-and-compare subcommand
    # ------------------------------------------------------------------
    compare_parser = subparsers.add_parser(
        "publish-and-compare",
        help="Publish fine-tuned model to HuggingFace Hub and compare "
             "responses with the base model side-by-side.",
    )
    compare_parser.add_argument(
        "--adapter-path",
        required=False,
        default=None,
        dest="adapter_path",
        help="Tinker path to the LoRA adapter (tinker://...). "
             "If omitted, reads from --output-dir/tinker_adapter_path.txt.",
    )
    compare_parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Output directory (used to find tinker_adapter_path.txt if "
             "--adapter-path is not provided).",
    )
    compare_parser.add_argument(
        "--hf-repo-id",
        required=True,
        dest="hf_repo_id",
        help="HuggingFace repository ID to publish to "
             "(e.g. 'your-username/oil-gas-qwen3').",
    )
    compare_parser.add_argument(
        "--hf-token",
        required=False,
        default=None,
        dest="hf_token",
        help="HuggingFace API token. If not provided, uses HF_TOKEN env var.",
    )
    compare_parser.add_argument(
        "--prompts",
        required=False,
        default=None,
        nargs="+",
        dest="prompts",
        help="One or more prompts to compare. If not provided, uses 5 sample "
             "prompts from the oil & gas domain.",
    )
    compare_parser.add_argument(
        "--prompts-file",
        required=False,
        default=None,
        dest="prompts_file",
        help="Path to a text file with one prompt per line.",
    )
    compare_parser.add_argument(
        "--output-file",
        required=False,
        default=None,
        dest="output_file",
        help="Path to save comparison results as JSON (optional).",
    )
    compare_parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        dest="base_model",
        help="Base model HuggingFace ID (default: Qwen/Qwen3-8B).",
    )
    compare_parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        dest="max_tokens",
        help="Maximum tokens to generate per response (default: 256).",
    )
    compare_parser.add_argument(
        "--skip-publish",
        action="store_true",
        dest="skip_publish",
        help="Skip publishing to HuggingFace and use an existing HF repo.",
    )

    return parser


async def _handle_train(args: argparse.Namespace) -> None:
    """Execute the ``train`` subcommand.

    Builds a :class:`~qwen3_rl_pipeline.config.TrainingConfig`, runs the
    full RL training loop, then exports the resulting adapter through the
    GGUF conversion and Ollama registration pipeline.

    Args:
        args: Parsed argument namespace from the ``train`` subparser.
    """
    from qwen3_rl_pipeline.clients import initialize_clients_async
    from qwen3_rl_pipeline.config import TrainingConfig
    from qwen3_rl_pipeline.data.loader import load_dataset
    from qwen3_rl_pipeline.training.checkpointing import Checkpointer
    from qwen3_rl_pipeline.training.loop import run_training_loop
    from qwen3_rl_pipeline.training.rewards import (
        ExactMatchReward,
        LLMJudgeReward,
        SubstringReward,
    )

    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        steps_per_checkpoint=args.steps_per_checkpoint,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
    )

    clients = await initialize_clients_async(rank=config.lora_rank)
    dataset = load_dataset(args.dataset)

    # Build the reward function based on --reward-type.
    reward_type = getattr(args, "reward_type", "substring")
    if reward_type == "exact":
        reward_fn = ExactMatchReward("")
        logger.info("Using ExactMatchReward")
    elif reward_type == "llm-judge":
        grader_model = getattr(args, "grader_model", "Qwen/Qwen3-30B-A3B-Instruct")
        default_rubric = getattr(args, "default_rubric", None) or (
            "Score the answer on a scale of 0.0 to 1.0 based on accuracy, "
            "completeness, and relevance to the question. "
            "1.0 = fully correct and complete. "
            "0.5 = partially correct or missing key details. "
            "0.0 = incorrect or irrelevant."
        )
        # Create a grader sampling client using the grader model.
        grader_client = await clients.service_client.create_sampling_client_async(
            base_model=grader_model
        )
        reward_fn = LLMJudgeReward(
            grader_client=grader_client,
            default_rubric_str=default_rubric,
            grader_model=grader_model,
        )
        logger.info("Using LLMJudgeReward with grader model: %s", grader_model)
    else:
        reward_fn = SubstringReward(dataset)
        logger.info("Using SubstringReward")
    checkpointer = Checkpointer(args.output_dir, args.steps_per_checkpoint)

    adapter_path = await run_training_loop(
        config, dataset, reward_fn, clients, checkpointer,
        loss_fn=getattr(args, "loss_fn", "cross_entropy"),
    )

    # ── Save the Tinker adapter path ───────────────────────────────────────
    # Weights are saved permanently on Tinker (ttl_seconds=None).
    # No local download, GGUF conversion, or Ollama registration needed.
    import os  # noqa: PLC0415
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_path_file = f"{args.output_dir}/tinker_adapter_path.txt"
    with open(adapter_path_file, "w") as f:
        f.write(adapter_path)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Adapter path: %s", adapter_path)
    logger.info("Saved to: %s", adapter_path_file)
    logger.info("")
    logger.info("To use in the web app:")
    logger.info("  export ADAPTER_PATH='%s'", adapter_path)
    logger.info("  uvicorn webapp.app:app --reload --port 8000")
    logger.info("=" * 60)

    sys.exit(0)


async def _handle_export(args: argparse.Namespace) -> None:
    """Execute the ``export`` subcommand.

    Downloads an existing LoRA adapter from Tinker and exports it through
    the GGUF conversion and Ollama registration pipeline.

    Args:
        args: Parsed argument namespace from the ``export`` subparser.
    """
    from qwen3_rl_pipeline.config import ModelfileConfig
    from qwen3_rl_pipeline.deployment.ollama import (
        generate_modelfile,
        register_model,
        write_modelfile,
    )
    from qwen3_rl_pipeline.export.adapter import download_adapter, merge_adapter
    from qwen3_rl_pipeline.export.converter import convert_to_gguf, quantize_gguf

    # Resolve adapter path — either directly or from a saved path file.
    adapter_path = args.adapter_path
    if not adapter_path:
        if args.adapter_path_file:
            with open(args.adapter_path_file) as f:
                adapter_path = f.read().strip()
        else:
            # Auto-detect from output_dir
            default_path_file = f"{args.output_dir}/tinker_adapter_path.txt"
            import os  # noqa: PLC0415
            if os.path.exists(default_path_file):
                with open(default_path_file) as f:
                    adapter_path = f.read().strip()
                logger.info("Loaded adapter path from %s", default_path_file)
            else:
                raise ValueError(
                    "No adapter path provided. Use --adapter-path or "
                    "--adapter-path-file, or ensure "
                    f"{default_path_file} exists from a previous train run."
                )

    adapter_dir = download_adapter(
        adapter_path, f"{args.output_dir}/adapter"
    )
    merged_dir = merge_adapter(
        adapter_dir,
        f"{args.output_dir}/merged_model",
        allow_download=getattr(args, "allow_download", False),
    )
    gguf_path = convert_to_gguf(
        merged_dir, f"{args.output_dir}/model.gguf", args.llama_cpp_dir
    )
    quantized_path = quantize_gguf(
        gguf_path,
        f"{args.output_dir}/model_q.gguf",
        args.quantization_type,
        args.llama_cpp_dir,
    )

    modelfile_config = ModelfileConfig(
        gguf_path=quantized_path,
        model_name=args.model_name,
    )
    modelfile_content = generate_modelfile(modelfile_config)
    modelfile_path = f"{args.output_dir}/Modelfile"
    write_modelfile(modelfile_content, modelfile_path)
    register_model(args.model_name, modelfile_path)

    sys.exit(0)


async def _handle_compare(args: argparse.Namespace) -> None:
    """Execute the ``publish-and-compare`` subcommand.

    Publishes the fine-tuned model to HuggingFace Hub (merge on Tinker's
    servers, no local download), then queries both the base model and the
    fine-tuned model with the same prompts and prints a side-by-side
    comparison.

    Args:
        args: Parsed argument namespace from the ``publish-and-compare``
            subparser.
    """
    import os  # noqa: PLC0415

    from qwen3_rl_pipeline.clients import initialize_clients_async
    from qwen3_rl_pipeline.evaluation.compare import (
        compare_models,
        print_comparison,
        publish_to_hf,
        save_comparison_to_file,
    )

    # ── Resolve adapter path ───────────────────────────────────────────────
    adapter_path = args.adapter_path
    if not adapter_path:
        default_path_file = f"{args.output_dir}/tinker_adapter_path.txt"
        if os.path.exists(default_path_file):
            with open(default_path_file) as f:
                adapter_path = f.read().strip()
            logger.info("Loaded adapter path from %s", default_path_file)
        else:
            raise ValueError(
                "No adapter path provided. Use --adapter-path or ensure "
                f"{default_path_file} exists from a previous train run."
            )

    # ── Resolve prompts ────────────────────────────────────────────────────
    prompts: list[str] = []
    if args.prompts:
        prompts = list(args.prompts)
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default oil & gas sample prompts.
        prompts = [
            "What is a blowout preventer and what is its purpose?",
            "Explain the difference between upstream and downstream operations.",
            "What is hydraulic fracturing and how does it work?",
            "What is the purpose of drilling mud in oil well operations?",
            "What is LNG and how is it produced?",
        ]
        logger.info("No prompts provided — using 5 default oil & gas prompts.")

    # ── Initialize Tinker clients ──────────────────────────────────────────
    clients = await initialize_clients_async()

    # ── Publish to HuggingFace (unless --skip-publish) ────────────────────
    if not args.skip_publish:
        await publish_to_hf(
            tinker_path=adapter_path,
            hf_repo_id=args.hf_repo_id,
            base_model=args.base_model,
            hf_token=args.hf_token,
            output_dir=f"{args.output_dir}/hf_export",
        )
    else:
        logger.info(
            "Skipping HuggingFace publish (--skip-publish). "
            "Using existing repo: %s", args.hf_repo_id
        )

    # ── Compare base vs fine-tuned model ──────────────────────────────────
    logger.info("Querying base and fine-tuned models for comparison...")
    results = await compare_models(
        service_client=clients.service_client,
        tinker_path=adapter_path,
        prompts=prompts,
        hf_repo_id=args.hf_repo_id,
        base_model=args.base_model,
        max_tokens=args.max_tokens,
    )

    # ── Print results ──────────────────────────────────────────────────────
    print_comparison(results)

    # ── Save to file if requested ──────────────────────────────────────────
    if args.output_file:
        save_comparison_to_file(results, args.output_file)

    sys.exit(0)


async def run_cli(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler.

    Configures structured logging to stdout at the requested level, then
    delegates to :func:`_handle_train` or :func:`_handle_export` based on
    the subcommand chosen by the user.  If no subcommand is provided, the
    help text is printed and the process exits with code 2.

    Args:
        argv: Argument list to parse.  Defaults to ``sys.argv[1:]`` when
            ``None``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging before anything else so that all subsequent log
    # calls respect the requested level.
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Also set the package logger level explicitly so that it is not
    # filtered by a pre-existing root-logger configuration.
    logging.getLogger("qwen3_rl_pipeline").setLevel(log_level)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(2)

    if args.subcommand == "train":
        await _handle_train(args)
    elif args.subcommand == "export":
        await _handle_export(args)
    elif args.subcommand == "publish-and-compare":
        await _handle_compare(args)
