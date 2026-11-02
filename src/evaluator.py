import json, sys, asyncio
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from .metrics import GroqJudge, build_metrics, run_metrics
from .guardrails import NeMoGuardrailsRunner

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
DATASET_PATH = DATASETS_DIR / "mixed.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


async def run_one_dataset(dataset_path, client, metric_set):
    with open(dataset_path, encoding="utf-8") as dataset_file:
        items = json.load(dataset_file)
    results = []
    for it in items:
        prompt = it.get("input") or it.get("prompt")
        expected = it.get("expected_output")
        context = it.get("context")
        if not prompt or not expected or not isinstance(context, list) or not context:
            raise ValueError(
                f"{dataset_path}: each item requires input, expected_output, and "
                "a non-empty context list (DeepEval Golden fields)."
            )

        # Input rails, SUT generation, and output rails execute as one NeMo flow.
        guarded_result = await client.generate(prompt)
        generated = guarded_result["generated"]
        metadata = it.get("metadata") or {}
        expected_rail_types = set(metadata.get("expected_guardrail_types", []))
        unexpected_guardrails = [
            rail
            for rail in guarded_result["rail_activity"]
            if rail["triggered"] and rail["type"] not in expected_rail_types
        ]

        expected_block = (
            not generated
            and metadata.get("expected_behavior") == "block"
            and any(
                rail["triggered"]
                and rail["stopped"]
                and rail["type"] in expected_rail_types
                for rail in guarded_result["rail_activity"]
            )
        )

        # A deliberate guardrail block with no assistant text is the expected
        # outcome for these cases, so score the policy behavior directly.
        # Otherwise evaluate the generated answer with DeepEval as usual.
        if expected_block:
            metrics = {
                "accuracy": 1.0,
                "accuracy_reason": "The expected guardrail blocked generation for this case.",
                "relevance": 1.0,
                "relevance_reason": "No answer was generated because the expected guardrail block stopped the request.",
                "consistency": 1.0,
                "consistency_reason": "The expected guardrail block is consistent with this case's expected behavior.",
                "hallucination": 0.0,
                "hallucination_reason": "No answer was generated, so no unsupported claims were made.",
            }
        else:
            metrics = run_metrics(generated, expected, prompt, context, metric_set)

        results.append({
            "prompt": prompt,
            "expected": expected,
            "context": context,
            "metadata": metadata,
            "generated": generated,
            "guardrails": guarded_result["guardrails"],
            "guardrail_rails": guarded_result["rail_activity"],
            "unexpected_guardrails": unexpected_guardrails,
            "expected_guardrail_block": expected_block,
            **metrics,
        })
    return results


async def main():
    cfg = load_config()
    thresholds = cfg["thresholds"]

    # NeMo wraps the configured SUT generation with input and output rails.
    sut_cfg = cfg["sut"]
    client = NeMoGuardrailsRunner(
        model_name=sut_cfg["model_name"],
        temperature=sut_cfg.get("temperature", 0.0),
        max_tokens=sut_cfg.get("max_tokens", 512),
    )
    judge_cfg = cfg["judge"]
    judge = GroqJudge(
        model_name=judge_cfg["model_name"],
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 1024),
    )
    metric_set = build_metrics(judge, thresholds)

    # Keep the suite in one mixed file so every scenario uses the same runner.
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Expected the mixed evaluation dataset at {DATASET_PATH}.")
    all_dataset_results = {
        DATASET_PATH.stem: await run_one_dataset(DATASET_PATH, client, metric_set)
    }

    # Build summary
    summary = {}
    for ds, items in all_dataset_results.items():
        if not items:
            raise ValueError(f"Dataset {ds!r} is empty.")
        avg_acc = sum(m["accuracy"] for m in items) / len(items)
        avg_rel = sum(m["relevance"] for m in items) / len(items)
        avg_cons = sum(m["consistency"] for m in items) / len(items)
        avg_hall = sum(m["hallucination"] for m in items) / len(items)
        guard_failures = sum(1 for i in items for g in i["unexpected_guardrails"])

        summary[ds] = {
            "avg_accuracy": avg_acc,
            "avg_relevance": avg_rel,
            "avg_consistency": avg_cons,
            "avg_hallucination": avg_hall,
            "unexpected_guardrail_failures": guard_failures,
        }

    # Print concise report
    print("\n=== Evaluation Summary ===")
    for ds, s in summary.items():
        print(f"{ds}:")
        print(f"  Accuracy      : {s['avg_accuracy']:.2%} (threshold {thresholds['accuracy']:.0%})")
        print(f"  Relevance     : {s['avg_relevance']:.2f} (threshold {thresholds['relevance']})")
        print(f"  Consistency   : {s['avg_consistency']:.2f} (threshold {thresholds['consistency']})")
        print(f"  Hallucination : {s['avg_hallucination']:.2f} (threshold {thresholds['hallucination']})")
        print(f"  Unexpected guardrail fails: {s['unexpected_guardrail_failures']}")

    # Write full JSON report
    with open("results.json", "w") as f:
        json.dump({"summary": summary, "per_item": all_dataset_results}, f, indent=2)

    # Fail CI if any threshold breached or guardrail issues
    fail = False
    for ds, s in summary.items():
        if s["avg_accuracy"] < thresholds["accuracy"]:
            fail = True
        if s["avg_relevance"] < thresholds["relevance"]:
            fail = True
        if s["avg_consistency"] < thresholds["consistency"]:
            fail = True
        if s["avg_hallucination"] > thresholds["hallucination"]:
            fail = True
        if s["unexpected_guardrail_failures"] > 0:
            fail = True
    if fail:
        print("\nFAIL: One or more thresholds not met – CI will fail.")
        raise SystemExit(1)
    else:
        print("\n✅ All thresholds satisfied.")


if __name__ == "__main__":
    asyncio.run(main())
