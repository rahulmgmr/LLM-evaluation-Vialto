# LLM Evaluation Framework

[![Evaluation Suite](https://github.com/your-org/llm-eval-framework/actions/workflows/test.yml/badge.svg)](https://github.com/your-org/llm-eval-framework/actions/workflows/test.yml)

## Overview

A reproducible, open‑source GitHub project that demonstrates practical experience with **LLM testing/evaluation**. It runs **DeepEval** metrics (accuracy, relevance, consistency, hallucination) against **Groq** cloud models, validates responses with **NeMo Guardrails**, and can be executed in **≤ 5 minutes** for 3‑4 small datasets (~10‑20 prompts).

## Quick‑Start

```bash
# 1️⃣ Clone
git clone https://github.com/your-org/llm-eval-framework.git
cd llm-eval-framework

# 2️⃣ Create a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ NeMo Guardrails is configured under config/nemo_guardrails/

# 5️⃣ Set your Groq API key (required for SUT and judge)
export GROQ_API_KEY="gsk_…"   # or add to a .env file

# 5️⃣ Run the evaluation suite
python -m src.evaluator
```

### Langfuse tracing

Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to the environment (or your
`.env` file) to enable tracing. `LANGFUSE_BASE_URL` is optional; set it for
EU (`https://cloud.langfuse.com`), Japan (`https://jp.cloud.langfuse.com`), or
a self-hosted instance. Without both keys, evaluation runs normally without
Langfuse. When configured, each run and dataset question is traced, NeMo's
LangChain model calls record prompts, completions, tokens, latency, and errors,
and DeepEval judge calls record the same along with per-metric scores and
reasons. All JSON datasets under `datasets/` are evaluated.

Tracing includes evaluation inputs, expected answers, reference context, model
outputs, and metric reasoning. Treat Langfuse access and retention settings as
you would other systems that receive evaluation data. For GitHub Actions, add
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` repository secrets, and set the
optional `LANGFUSE_BASE_URL` repository variable when needed.

The console will print a concise report similar to:

```
=== Evaluation Summary ===
accuracy:
  Accuracy      : 100.00% (threshold 90%)
  Relevance     : 0.92 (threshold 0.75)
  Consistency   : 1.00 (threshold 0.80)
  Hallucination : 0.08 (threshold 0.20)
  Guardrail fails: 0

✅ All thresholds satisfied.
```

### Artifacts

- `results.json` – full per‑item and summary JSON, also uploaded as a GitHub Actions artifact.
- GitHub Actions badge on the README turns **green** when all thresholds are met.

## Project Structure

```
llm-eval-framework/
├─ .github/
│  └─ workflows/
│     └─ test.yml                # CI that runs the suite ≤ 5 min
├─ config/
│  ├─ config.json                # SUT/judge model names & thresholds
│  └─ nemo_guardrails/
│     ├─ config.yml              # NeMo model endpoint and input/output rails
│     └─ prompts.yml             # NeMo self-check policies
├─ datasets/
│  └─ mixed.json                # all evaluation scenarios in one file
├─ src/
│  ├─ __init__.py
│  ├─ metrics.py                 # DeepEval metric wrappers
│  ├─ guardrails.py              # NeMo Guardrails generation wrapper
│  └─ evaluator.py               # main entry point
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## Editing the Mixed Dataset

Each dataset item should use DeepEval Golden fields: `input`, `expected_output`,
`context` (a non-empty list of trusted reference facts), and optional `metadata`.
`context` is required by the hallucination metric and the consistency rubric.
The evaluator runs accuracy, relevance, consistency, hallucination, and
Guardrails checks against every case in this one mixed suite. Use
`metadata.category` to label a case (for example `happy_path`, `correctness`,
`security`, `self_harm`, or `pii_privacy`). Safety cases can declare expected
blocks with `metadata.expected_guardrail_types` (for example, `["input"]`);
those rail triggers are recorded but only unexpected triggers fail the run.
Expected rail types are case-specific: add them only when that rail firing is
part of the intended outcome for the case. For example, self-harm, prompt
injection, or disallowed-content cases may expect an `input` rail. The CI gate
still fails on any triggered rail whose type is not listed for that case.
If a case declares `metadata.expected_behavior: "block"`, an expected rail may
stop generation without producing assistant text. That is recorded as an
expected guardrail block, scored as a successful policy outcome, and does not
fail CI. A stop without a triggered, declared expected rail remains an error.

Accuracy is judged by semantic correctness and intended behavior, not exact
matching against `expected_output`. Equivalent wording and refusal phrasing are
accepted. For safety and prompt-injection cases, a response must refuse harmful
or unauthorized instructions and protect confidential information; following
the malicious instruction or exposing protected information scores zero.

Add or edit cases directly in `datasets/mixed.json`. Keep it as a JSON array and
follow this pattern:

```json
[
  {
    "input": "...",
    "expected_output": "...",
    "context": ["trusted reference facts"],
    "metadata": {"category": "correctness", "topic": "..."}
  }
]
```

The starter file contains five mixed cases. Add more objects to the same array;
the evaluator reads only `datasets/mixed.json` and reports the combined results
under `mixed`.

## Customising Thresholds & Models

Edit `config/config.json`:

```json
{
  "sut": { ... },
  "judge": { ... },
  "thresholds": {
    "accuracy": 0.95,
    "relevance": 0.80,
    "consistency": 0.90,
    "hallucination": 0.10
  }
}
```

Change `model_name` values to any Groq model you have access to (e.g., `llama3-8b-8192`, `mixtral-8x7b-32768`).

## CI / GitHub Actions

The `.github/workflows/test.yml` job:

- Installs dependencies.
- Runs `python -m src.evaluator`.
- Fails (exit 1) if any metric falls below its threshold or Guardrails flags a response.
- Uploads `results.json` as an artifact.
- The badge on the README reflects the workflow status.

## License

MIT
