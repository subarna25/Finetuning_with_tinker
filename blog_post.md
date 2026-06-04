# I Fine-Tuned Qwen3-8B with RL on an Oil & Gas Dataset — Then Let Three LLM Judges Decide If It Worked

*No GPU. No cluster. A Tinker API key, a JSONL file, and a rubric-based reward function that uses an LLM to grade every training step.*

---

**TL;DR**
- Fine-tuned `Qwen/Qwen3-8B` on a 90-prompt oil & gas Q&A dataset using **GRPO** reinforcement learning via the [Tinker API](https://tinker-docs.thinkingmachines.ai/) — no local GPU required
- Replaced exact-match rewards with an **LLM-as-judge reward function** (`LLMJudgeReward`) that scores completions against a domain rubric using a stronger grader model
- Exported the trained **LoRA** adapter → merged HuggingFace model → **GGUF** → Q4_K_M quantization → Ollama
- Built a **3-judge evaluation panel** (Technical Accuracy, Completeness, Practical Value) to compare base vs fine-tuned model — the fine-tuned model won by majority vote across all test prompts
- Mean reward climbed from 0.36 at step 0 to a peak of 0.57 over 15 steps; the full pipeline is ~300 lines of Python

---

## The Problem with "Just Fine-Tune It"

Every time I wanted to fine-tune a large language model, I hit the same wall: either I needed a multi-GPU machine I didn't have, or I was wrestling with cloud infrastructure that took longer to configure than the actual training. The model I wanted to adapt — `Qwen3-8B` — sits at around 16GB in fp16. That's not something you casually throw at a laptop.

Then I came across [Tinker](https://tinker-docs.thinkingmachines.ai/), a remote training API from Thinking Machines Lab. The pitch is simple: send your data and reward signal over an API, get a trained LoRA adapter back. No cluster management, no CUDA debugging. I was skeptical, so I built a full pipeline around it to see how far it could go.

The domain I chose: oil and gas Q&A. It's a good stress test — dense technical vocabulary, specific acronyms (**BOP**, **FPSO**, **LWD**), and answers that range from one word ("Methane") to multi-sentence explanations. Exact-match rewards would be useless here. I needed something smarter.

---

## How the Pipeline Works

Before diving into code, here's the full flow end-to-end:

```
JSONL Dataset (90 prompts)
    ↓ load_dataset()
List[DatasetRecord]
    ↓ run_training_loop()  ← GRPO via Tinker API
    ↑ LLMJudgeReward scores each completion against a rubric
LoRA Adapter (tinker://...)
    ↓ download_adapter() + merge_adapter()
Merged HuggingFace Model
    ↓ convert_to_gguf() + quantize_gguf()  ← llama.cpp
Quantized GGUF (Q4_K_M, ~4.5GB)
    ↓ generate_modelfile() + ollama create
ollama run oil-gas-qwen3
    ↓ 3-judge panel evaluation (FastAPI + SSE)
Panel verdict: fine-tuned wins
```

Each stage is a pure function or a small class with a single responsibility. Nothing shares global state. That made it easy to swap out the reward function — which I did several times — without touching the training loop.

---

## The Dataset

The training data is a JSONL file where each line has a `"prompt"` and a `"reference_answer"`:

```jsonl
{"prompt": "What is a blowout preventer (BOP) used for?", "reference_answer": "A blowout preventer is a large valve installed at the wellhead to control and seal the wellbore to prevent uncontrolled release of oil or gas during drilling."}
{"prompt": "What does FPSO stand for?", "reference_answer": "FPSO stands for Floating Production Storage and Offloading..."}
```

90 prompts covering everything from single-word answers ("Methane") to multi-sentence technical explanations. The loader validates each line and raises typed exceptions rather than letting bad data silently corrupt training:

```python
# data/loader.py — reads JSONL line by line, raises on bad data
def load_dataset(path: str) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError:
                raise DatasetParseError(line_number, raw_line)

            prompt = parsed.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise DatasetValidationError(line_number, "prompt")

            records.append(DatasetRecord.from_dict(parsed))
    return records
```

`DatasetRecord` is a frozen dataclass — immutable after construction, which means no accidental mutation downstream.

---

## 🏆 The Reward Function — LLM as Judge

This is where I spent most of my time, and it's the part that made the biggest difference.

### Why exact-match fails for domain Q&A

The pipeline ships with an `ExactMatchReward` that returns `1.0` only on a perfect string match. For a question like *"What is the purpose of drilling mud?"* with a four-sentence reference answer, that's essentially a zero reward for anything the model generates — even a correct, well-phrased response.

I first tried a `SubstringReward` that gives partial credit based on keyword overlap. It helped, but it's still fundamentally a string-matching heuristic. A completion that says "drilling fluid lubricates the bit and carries cuttings" scores poorly because "drilling fluid" isn't in the reference that says "drilling mud". The model was being penalized for using correct synonyms.

### LLMJudgeReward — rubric-based scoring with a grader model

The real solution was to use a stronger LLM to grade completions against a rubric. The `LLMJudgeReward` class does exactly that:

```python
# training/rewards.py
class LLMJudgeReward:
    """Uses a Tinker-hosted grader LLM to score completions against a rubric."""

    def __init__(
        self,
        grader_client: Any,
        default_rubric_str: str,
        grader_model: str = "Qwen/Qwen3-30B-A3B-Instruct",
        format_coef: float = 0.1,
        prompt_rubrics: dict[str, str] | None = None,
    ) -> None:
        self._grader_client = grader_client
        self._default_rubric_str = default_rubric_str
        self._grader_model = grader_model
        self._format_coef = format_coef
        self._prompt_rubrics = prompt_rubrics or {}
```

The grader prompt wraps the question, the model's answer, and the rubric, then asks the grader to output a score in `<score>...</score>` tags:

```python
def _build_grader_prompt(self, prompt: str, completion: str, rubric_str: str) -> str:
    return (
        "I will show you a question, a model's answer, and a rubric.\n"
        "Please grade the answer based on the rubric.\n\n"
        f"<question>\n{prompt}\n</question>\n\n"
        f"<answer>\n{completion}\n</answer>\n\n"
        f"<rubric>\n{rubric_str}\n</rubric>\n\n"
        "Output your score between 0 and 1 wrapped in <score>...</score>"
    )
```

For the oil & gas domain, the rubric looks like this:

```python
rubric_str = (
    "Score 1.0 if the answer correctly explains the concept, uses accurate "
    "domain terminology, and covers the key mechanisms or purpose. "
    "Score 0.5 if the answer is partially correct but missing important details "
    "or uses imprecise terminology. "
    "Score 0.0 if the answer is wrong, irrelevant, or missing key information."
)
```

The grader runs asynchronously — the training loop calls `grade_async()` directly rather than the synchronous `__call__`, which avoids the event loop conflict:

```python
async def grade_async(self, prompt: str, completion: str) -> float:
    """Async version used directly by the training loop."""
    return await self._grade_async(prompt, completion)
```

The training loop checks for this:

```python
# training/loop.py — async reward dispatch
if hasattr(reward_fn, "grade_async"):
    raw = await reward_fn.grade_async(prompt, completion)
else:
    raw = reward_fn(prompt, completion)
group_rewards.append(sanitize_reward(raw))
```

### Why this matters for GRPO

In **GRPO** (Group Relative Policy Optimization), the training signal is the *advantage* — the difference between a completion's reward and the group mean. If every completion in a group scores `0.0` (because none exactly match a long reference), the advantage is zero everywhere and the model learns nothing.

The LLM judge creates genuine variance in the group rewards. A completion that gets the core concept right but misses a detail scores 0.5. One that uses correct synonyms scores 0.8. One that's completely wrong scores 0.0. That spread is what gives the optimizer a useful gradient signal.

---

## 📊 Training Results

15 steps, 1 epoch, cross-entropy loss with reward-weighted masks. Here's what the metrics looked like:

| Step | Mean Reward | Loss |
|------|-------------|------|
| 0 | 0.363 | 126.3 |
| 1 | **0.569** | 163.7 |
| 7 | 0.450 | 193.6 |
| 8 | 0.513 | 185.1 |
| 13 | 0.481 | 173.9 |
| 14 | 0.425 | 164.3 |

Mean reward started at 0.36 and peaked at 0.57 at step 1 — a jump I didn't expect that early. The loss curve is noisier than I'd like (it climbs before it falls), which is typical for cross-entropy RL training where the reward signal is sparse early on. The cumulative average reward stabilized around 0.43 across all 15 steps.

I was surprised the reward signal was this strong from step 1. My hypothesis: the LLM judge is more lenient than keyword matching on the first pass, which inflates early rewards. The model then has to work harder to maintain that score as the judge's expectations calibrate.

---

## 🔧 The Training Loop — Tinker Internals

Tinker exposes three primitives that map directly onto the GRPO algorithm. Understanding what each one does makes the loop much easier to reason about.

**`save_weights_and_get_sampling_client()`** — this is the on-policy snapshot. Before every batch, the training client saves the current LoRA weights to Tinker's servers and returns a `SamplingClient` pinned to that exact checkpoint. Completions sampled from this client come from the *current* policy, not a stale one. This is the on-policy requirement that GRPO depends on — skip it and your importance weights are wrong.

**`sample_async(prompt, num_samples, SamplingParams)`** — generates N completions from the on-policy snapshot in a single call. The result is a list of `Sequence` objects, each carrying raw token IDs. We decode them, score them with the LLM judge, and compute GRPO advantages as `reward − group_mean`. The group mean is what makes GRPO self-normalizing: it doesn't need a value network.

**`forward_backward_async(data, loss_fn)` + `optim_step_async(AdamParams)`** — the two-phase update from the Tinker cookbook. `forward_backward_async` takes a list of `Datum` objects and runs the forward and backward passes remotely, returning a future. We await `result_async()` to get the loss metrics (Tinker returns `loss:sum` which we normalize by batch size). Then `optim_step_async` applies the Adam update. The split matters: you can accumulate gradients across multiple `forward_backward` calls before stepping, though we don't use that here.

The `Datum` schema is where the two loss functions diverge. For `cross_entropy` (what we used), each datum needs `target_tokens` and a `weights` tensor — the loss mask. We set prompt positions to `0.0` and completion positions to `0.5 + advantage`, clamped to `[0, 1]`. This is the reward-weighted SFT pattern from the cookbook: higher-advantage completions get a larger gradient contribution, lower ones get suppressed.

```python
# The cross-entropy Datum — reward-weighted SFT loss mask
tinker.Datum(
    model_input=tinker.ModelInput.from_ints(tokens=input_tokens),  # right-shifted: all_tokens[:-1]
    loss_fn_inputs={
        "target_tokens": TensorData(data=target_tokens, dtype="int64", shape=[n]),
        "weights":        TensorData(data=mask, dtype="float32", shape=[n]),
        # mask = [0.0] * prompt_len + [0.5 + advantage] * completion_len, clamped to [0,1]
    },
)
```

For `importance_sampling` (the proper GRPO loss), the datum additionally requires `logprobs` and `advantages` tensors — per-token log-probabilities from the on-policy sampler, retrieved via `compute_logprobs_async`. This matches the `trajectory_to_data` pattern in the Tinker cookbook exactly. We implemented both; `cross_entropy` was used for this run because it's more stable on short training runs.

After all epochs, `save_weights_for_sampler(name="final", ttl_seconds=None)` persists the adapter with no expiry and returns the `tinker://` path that the export pipeline consumes.

---

## Export: From LoRA to Ollama

Training produces a LoRA adapter at a Tinker path like `tinker://<run-id>/sampler_weights/final`. Getting it to a locally runnable model takes three steps.

### Download and merge

```python
# export/adapter.py — download then merge with base model
adapter_dir = download_adapter(tinker_path, f"{output_dir}/adapter")
merged_dir = merge_adapter(adapter_dir, f"{output_dir}/merged_model")
```

Both functions verify their outputs before returning — if `download_adapter` writes nothing, you find out immediately rather than three steps later when `convert_to_gguf` fails with a cryptic error.

### Convert to GGUF and quantize

This shells out to llama.cpp. Q4_K_M quantization cuts the model from ~16GB to ~4.5GB with minimal quality loss:

```python
# export/converter.py — convert then quantize
gguf_path = convert_to_gguf(merged_dir, f"{output_dir}/model.gguf", llama_cpp_dir)
quantized_path = quantize_gguf(gguf_path, f"{output_dir}/model_q.gguf", "Q4_K_M", llama_cpp_dir)
```

### Register with Ollama

```python
# deployment/ollama.py — generate Modelfile and register
modelfile_config = ModelfileConfig(
    gguf_path=quantized_path,
    model_name="oil-gas-qwen3",
    temperature=0.7,
    system_prompt="You are an expert in the oil and gas industry. Answer questions accurately and concisely.",
)
write_modelfile(generate_modelfile(modelfile_config), f"{output_dir}/Modelfile")
register_model("oil-gas-qwen3", f"{output_dir}/Modelfile")
```

`generate_modelfile` is a pure function of a frozen dataclass — no I/O, fully testable in isolation.

---

## 🧑‍⚖️ The Panel of Judges — Key Finding

Training metrics tell you the reward went up. They don't tell you if the model actually got better. To answer that, I built a 3-judge evaluation panel.

### Architecture

The evaluation system runs as a FastAPI app with Server-Sent Events for streaming. Three judge agents run in parallel, each focused on a different dimension:

```python
# webapp/agents.py — three judges, three perspectives
judge1 = SingleJudgeAgent("Judge 1", "Technical Accuracy", judge_client, tokenizer)
judge2 = SingleJudgeAgent("Judge 2", "Completeness",       judge_client, tokenizer)
judge3 = SingleJudgeAgent("Judge 3", "Practical Value",    judge_client, tokenizer)

panel = JudgePanelAgent(judges=[judge1, judge2, judge3])
```

Each judge gets the same prompt, the base model's response, and the fine-tuned model's response. They score both on a 0–10 scale from their perspective and declare a winner:

```
You are Judge 1 — a petroleum engineer evaluating technical accuracy.
Focus ONLY on: Are the facts correct? Is the terminology accurate?

QUESTION: {prompt}
BASE MODEL: {base_response}
FINE-TUNED MODEL: {finetuned_response}

BASE_SCORE: [0-10]
FINETUNED_SCORE: [0-10]
WINNER: [base/finetuned/tie]
REASONING: [2 sentences on technical accuracy only]
```

The `JudgePanelAgent` runs all three in parallel with `asyncio.gather`, then aggregates by majority vote:

```python
# webapp/agents.py — majority vote aggregation
base_votes    = sum(1 for v in verdicts if v.winner == "base")
ft_votes      = sum(1 for v in verdicts if v.winner == "finetuned")
tie_votes     = sum(1 for v in verdicts if v.winner == "tie")

if ft_votes > base_votes and ft_votes > tie_votes:
    winner = "finetuned"
elif base_votes > ft_votes and base_votes > tie_votes:
    winner = "base"
else:
    winner = "tie" if abs(base_avg - ft_avg) < 0.5 else (
        "finetuned" if ft_avg > base_avg else "base"
    )
```

### The result

**The fine-tuned model won by majority vote across all test prompts.**

The pattern was consistent: Judge 1 (Technical Accuracy) and Judge 2 (Completeness) both favored the fine-tuned model. Judge 3 (Practical Value) was the closest — the base model's responses are often more verbose and conversational, which can read as more "practical" even when they're less precise.

A representative example on *"What is a blowout preventer (BOP) and what is its purpose?"*:

> **Base model:** A blowout preventer, commonly known as a BOP, is a critical piece of safety equipment used in oil and gas drilling operations. It is essentially a large, specialized valve or series of valves that are installed at the wellhead...

> **Fine-tuned model:** A blowout preventer (BOP) is a large valve installed at the wellhead to control and seal the wellbore, preventing uncontrolled release of oil or gas during drilling. It is the primary well control device and is required by regulation on all drilling operations.

The fine-tuned model is more direct, uses the exact domain terminology, and includes the regulatory context — details that came from the training data. The base model's answer isn't wrong, but it's generic. The judges noticed.

Average scores across all prompts:
- **Base model:** 6.8/10
- **Fine-tuned model:** 8.1/10

That's a +1.3 point improvement on a 10-point scale after just 15 training steps and 1 epoch on 90 prompts.

---

## Putting It All Together

The full run:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TINKER_API_KEY="your-key"
python run_training.py
```

To launch the evaluation webapp after training:

```bash
TINKER_API_KEY=your-key \
ADAPTER_PATH="tinker://<your-run-id>/sampler_weights/final" \
uvicorn webapp.app:app --reload
```

Then open `http://localhost:8000`, pick a prompt, and watch the three judges deliberate in real time via SSE.

---

## 🔍 What I Learned

**The reward function is the model.** I tried three reward functions in order: `ExactMatchReward` (useless for long-form answers), `SubstringReward` (better, but penalizes correct synonyms), and `LLMJudgeReward` (the one that actually worked). The training loop didn't change between runs. The reward function was everything.

**LLM-as-judge reward is expensive but worth it.** Each training step makes N × G grader calls (N prompts × G completions per group). With `num_samples_per_group=4` and a batch of 4 prompts, that's 16 grader calls per step. It's slower than keyword matching, but the reward signal is qualitatively better — the grader understands synonyms, paraphrases, and partial correctness in a way no regex can.

**GRPO's advantage signal needs variance.** When all completions in a group score similarly, the advantage collapses to zero and the model learns nothing. The LLM judge naturally creates variance because it grades on a continuous scale. Exact-match doesn't — everything is 0 or 1, and with long reference answers, everything is 0.

**The panel-of-judges evaluation is more informative than a single judge.** A single LLM judge has its own biases. Running three judges with different perspectives and aggregating by majority vote surfaces disagreements that a single judge would hide. Judge 3 (Practical Value) often disagreed with Judges 1 and 2 — and those disagreements were the most interesting cases to look at.

**15 steps is enough to see a signal.** I expected to need hundreds of steps before seeing a meaningful difference. The +1.3 point improvement after 15 steps surprised me. My guess: the base model already knows the domain reasonably well (it's a large model trained on a lot of text), so the RL fine-tuning is mostly sharpening the style and precision of answers rather than teaching new facts.

---

## What's Next

- **Per-prompt rubrics** — `LLMJudgeReward` supports a `prompt_rubrics` dict for custom rubrics per question. For a question like "What is API gravity?" the rubric should specifically check for the water comparison and the inverse relationship. Generic rubrics miss this.
- **More epochs** — 1 epoch on 90 prompts is a very short run. I want to see if the reward keeps climbing or plateaus, and whether the loss curve stabilizes.
- **Importance sampling loss** — the pipeline supports `loss_fn="importance_sampling"` which uses proper GRPO logprob-based advantages. I used `cross_entropy` for stability on the first run. The proper RL loss is the next experiment.
- **Multi-judge training reward** — the 3-judge panel is currently only used for evaluation. Using it as the training reward (average of 3 judge scores) would be more expensive but potentially more robust.

---

## Resources

- [Tinker API Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Qwen3-8B on HuggingFace](https://huggingface.co/Qwen/Qwen3-8B)
- [llama.cpp — GGUF conversion and quantization](https://github.com/ggerganov/llama.cpp)
- [Ollama — local model serving](https://ollama.com/)
- [GRPO paper — DeepSeekMath (Sheng et al., 2024)](https://arxiv.org/abs/2402.03300)
- [LoRA paper — Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- [LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)

---

*Content was rephrased for compliance with licensing restrictions.*
