"""Comparison script: HRM-JEPA-LLM vs Baseline LLM.

Tests whether the HRM+JEPA reasoning layer improves LLM outputs
compared to baseline gpt-oss:20b.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.encoders import SimpleTokenizer
from core.llm_integration import ComparisonFramework, create_text_only_pipeline


def create_test_cases() -> list[dict[str, str]]:
    """Create test cases for comparison.

    Returns:
        List of test cases with questions
    """
    return [
        {
            "name": "simple_reasoning",
            "text": "What is 15 + 27?",
            "category": "arithmetic",
        },
        {
            "name": "multi_step_reasoning",
            "text": "If Alice has 3 apples and Bob gives her twice as many, "
            "then Charlie takes 4, how many does Alice have?",
            "category": "math_word_problem",
        },
        {
            "name": "logical_consistency",
            "text": "All cats are mammals. Fluffy is a cat. "
            "Is Fluffy a mammal?",
            "category": "logic",
        },
        {
            "name": "chain_of_thought",
            "text": "A train travels 60 mph for 2 hours, then 40 mph for 3 hours. "
            "What is the average speed?",
            "category": "math_reasoning",
        },
        {
            "name": "common_sense",
            "text": "If it's raining outside, should I bring an umbrella?",
            "category": "common_sense",
        },
    ]


def run_comparison(
    output_dir: Path = Path("comparison_results"),
    model: str = "gpt-oss:20b",
    system_prompt: str | None = None,
) -> None:
    """Run full comparison between HRM-JEPA-LLM and baseline.

    Args:
        output_dir: Directory to save results
        model: Ollama model to use
        system_prompt: Optional system prompt
    """
    print(f"🚀 Starting HRM-JEPA-LLM vs Baseline comparison")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline and baseline
    print("\n📦 Initializing models...")
    pipeline, baseline_llm = create_text_only_pipeline(latent_dim=512, model=model)

    # Create comparison framework
    framework = ComparisonFramework(pipeline, baseline_llm)

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Get test cases
    test_cases = create_test_cases()
    print(f"📝 Running {len(test_cases)} test cases\n")

    # Tokenize test cases
    tokenized_cases = []
    for case in test_cases:
        input_ids, attention_mask = tokenizer.encode(case["text"], max_length=256)
        tokenized_cases.append({
            "text": case["text"],
            "name": case["name"],
            "category": case["category"],
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        })

    # Run comparison
    results = []
    for i, case in enumerate(tokenized_cases, 1):
        print(f"\n[{i}/{len(tokenized_cases)}] {case['name']}")
        print(f"Question: {case['text'][:80]}...")

        try:
            result = framework.compare(
                text=case["text"],
                input_ids=case["input_ids"],
                attention_mask=case["attention_mask"],
                system_prompt=system_prompt,
            )

            # Add metadata
            result["name"] = case["name"]
            result["category"] = case["category"]
            result["timestamp"] = datetime.now().isoformat()

            results.append(result)

            # Print responses
            print(f"\n  HRM-JEPA-LLM: {result['hrm_jepa_llm']['response'][:100]}...")
            print(f"  Baseline:     {result['baseline_llm']['response'][:100]}...")

            if result["hrm_jepa_llm"].get("consistency_score") is not None:
                score = result["hrm_jepa_llm"]["consistency_score"]
                print(f"  Consistency: {score:.3f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "name": case["name"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comparison_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if "error" not in r]
    print(f"Successful: {len(successful)}/{len(results)}")

    if successful:
        avg_hrm_duration = sum(
            r["hrm_jepa_llm"]["duration_ms"] for r in successful
        ) / len(successful)
        avg_baseline_duration = sum(
            r["baseline_llm"]["duration_ms"] for r in successful
        ) / len(successful)

        print(f"\nAverage Duration:")
        print(f"  HRM-JEPA-LLM: {avg_hrm_duration:.1f}ms")
        print(f"  Baseline:     {avg_baseline_duration:.1f}ms")
        print(f"  Overhead:     {avg_hrm_duration - avg_baseline_duration:.1f}ms")

        # Consistency scores
        consistency_scores = [
            r["hrm_jepa_llm"]["consistency_score"]
            for r in successful
            if r["hrm_jepa_llm"].get("consistency_score") is not None
        ]

        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            print(f"\nAverage Consistency: {avg_consistency:.3f}")

    # HRM status
    if successful:
        hrm_status = successful[0]["hrm_status"]
        print(f"\nHRM Status:")
        print(f"  Working Memory Steps: {hrm_status['working_memory']['task_steps']}")
        print(
            f"  Abstract Core Updates: {hrm_status['abstract_core']['update_count']}"
        )
        print(
            f"  Reflection Traces: {hrm_status['reflection']['total_traces']}"
        )


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare HRM-JEPA-LLM vs Baseline LLM"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for LLM",
    )

    args = parser.parse_args()

    run_comparison(
        output_dir=args.output_dir,
        model=args.model,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
