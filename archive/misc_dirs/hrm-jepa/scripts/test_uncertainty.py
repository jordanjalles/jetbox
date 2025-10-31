"""Test uncertainty detection and user question generation.

Runs the HRM-JEPA-LLM pipeline with uncertainty detection enabled
and displays highlighted areas where the model needs user help.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.encoders import SimpleTokenizer
from core.llm_integration import create_text_only_pipeline


def format_user_questions(uncertainty_analysis: dict) -> str:
    """Format user questions for display.

    Args:
        uncertainty_analysis: Uncertainty analysis dict

    Returns:
        Formatted string for display
    """
    if not uncertainty_analysis:
        return "No uncertainty analysis available"

    output = []

    # Overall uncertainty
    overall = uncertainty_analysis["overall_uncertainty"]
    is_uncertain = uncertainty_analysis["is_uncertain"]

    output.append(f"\n{'='*60}")
    output.append(f"UNCERTAINTY ANALYSIS")
    output.append(f"{'='*60}")

    # Uncertainty bar
    bar_length = 40
    filled = int(overall * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    status = "⚠️  UNCERTAIN" if is_uncertain else "✓ CONFIDENT"

    output.append(f"\nOverall Uncertainty: {overall:.2%} {status}")
    output.append(f"[{bar}]")

    # Breakdown
    output.append(f"\nUncertainty Breakdown:")
    breakdown = uncertainty_analysis["uncertainty_breakdown"]
    output.append(f"  Consistency:       {breakdown['consistency']:.2%}")
    output.append(f"  Component Variance: {breakdown['variance']:.2%}")
    output.append(f"  Concept Confidence: {breakdown['concept_confidence']:.2%}")

    # Detected concepts
    output.append(f"\nDetected Concepts (Top 3):")
    for i, concept in enumerate(uncertainty_analysis["detected_concepts"], 1):
        conf_str = f"{concept['confidence']:.1%}"
        conf_bar = "█" * int(concept['confidence'] * 20) + "░" * (20 - int(concept['confidence'] * 20))
        output.append(f"  {i}. {concept['concept'].replace('_', ' '):30s} {conf_str:>6s} [{conf_bar}]")

    # Uncertain concepts
    if uncertainty_analysis["uncertain_concepts"]:
        output.append(f"\n⚠️  Low Confidence Concepts:")
        for concept in uncertainty_analysis["uncertain_concepts"]:
            output.append(f"  - {concept['concept'].replace('_', ' ')} (confidence: {concept['confidence']:.1%})")

    # User questions
    if uncertainty_analysis["user_questions"]:
        output.append(f"\n{'='*60}")
        output.append(f"🙋 QUESTIONS FOR USER (Model needs help!)")
        output.append(f"{'='*60}")

        for i, question in enumerate(uncertainty_analysis["user_questions"], 1):
            priority_icon = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(question["priority"], "⚪")

            output.append(f"\n{priority_icon} Question {i} [{question['type'].replace('_', ' ').title()}]")
            output.append(f"Priority: {question['priority'].upper()}")
            output.append(f"\n{question['question']}")
            output.append(f"\nContext: {question['context']}")
            output.append(f"{'-'*60}")

    return "\n".join(output)


def test_uncertainty(
    test_questions: list[str] | None = None,
    model: str = "gpt-oss:20b",
) -> None:
    """Test uncertainty detection.

    Args:
        test_questions: List of questions to test (None = defaults)
        model: Ollama model to use
    """
    if test_questions is None:
        test_questions = [
            "What is 15 + 27?",  # Simple, low uncertainty expected
            "If Alice has 3 apples and Bob gives her twice as many, then Charlie takes 4, how many does Alice have?",  # Multi-step
            "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?",  # Logic
            "Should I invest in cryptocurrency or stocks?",  # High uncertainty expected
            "What is the meaning of life?",  # Very high uncertainty expected
        ]

    print(f"🔍 Testing Uncertainty Detection")
    print(f"Model: {model}")
    print(f"Test questions: {len(test_questions)}")
    print("-" * 60)

    # Create pipeline with uncertainty enabled
    print("\n📦 Initializing models (with uncertainty detection)...")
    pipeline, _ = create_text_only_pipeline(
        latent_dim=512,
        model=model,
        enable_uncertainty=True,
    )

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*60}")
        print(f"\nQuestion: {question}")

        # Tokenize
        input_ids, attention_mask = tokenizer.encode(question, max_length=256)
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.tensor([attention_mask])

        # Run pipeline
        try:
            result = pipeline.forward(
                text=question,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_hrm_context=True,
            )

            # Display response
            print(f"\nResponse: {result['llm_response'][:200]}...")

            # Display uncertainty analysis
            if result.get("uncertainty_analysis"):
                print(format_user_questions(result["uncertainty_analysis"]))
            else:
                print("\n⚠️  No uncertainty analysis available (detector may not be enabled)")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ Uncertainty testing complete")
    print(f"{'='*60}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test uncertainty detection and user questions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        default=None,
        help="Custom questions to test",
    )

    args = parser.parse_args()

    test_uncertainty(
        test_questions=args.questions,
        model=args.model,
    )


if __name__ == "__main__":
    main()
