"""Synthetic text data generator using gpt-oss:20b via Ollama.

Generates training data for JEPA and HRM training:
- Question-answer pairs
- Chain-of-thought reasoning
- Consistent vs inconsistent examples
- Multi-step problem solving

Uses self-play: LLM generates questions, then answers them.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from ollama import chat


# Question templates for different reasoning types
QUESTION_TEMPLATES = {
    "arithmetic": [
        "What is {a} + {b}?",
        "Calculate {a} - {b}",
        "What is {a} * {b}?",
        "Divide {a} by {b}",
    ],
    "word_problems": [
        "If {name1} has {n1} apples and {name2} gives {pronoun1} {n2} more, how many does {name1} have?",
        "A train travels {speed1} mph for {hours1} hours, then {speed2} mph for {hours2} hours. What is the total distance?",
        "If a book costs ${price} and you have ${money}, how many books can you buy?",
    ],
    "logic": [
        "All {category1} are {category2}. {name} is a {category1}. Is {name} a {category2}?",
        "If {condition1}, then {condition2}. {condition1} is true. What can we conclude?",
        "Either {option1} or {option2}. We know {option1} is false. What is true?",
    ],
    "common_sense": [
        "If it's {weather}, should I {action}?",
        "What happens if you {action1} and then {action2}?",
        "Is it safe to {action}? Why or why not?",
    ],
    "analogy": [
        "{word1} is to {word2} as {word3} is to what?",
        "What do {thing1} and {thing2} have in common?",
        "How is {concept1} similar to {concept2}?",
    ],
}

# Placeholders for templates
PLACEHOLDERS = {
    "a": [10, 15, 27, 42, 56, 73, 89, 100, 125, 250],
    "b": [5, 12, 18, 23, 34, 45, 67, 88, 99, 111],
    "n1": [3, 5, 7, 10, 12],
    "n2": [2, 4, 6, 8, 10],
    "name1": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "name2": ["Friend", "Teacher", "Parent", "Neighbor", "Colleague"],
    "pronoun1": ["him", "her", "them"],
    "speed1": [40, 50, 60, 70, 80],
    "speed2": [30, 40, 50, 60, 70],
    "hours1": [1, 2, 3, 4],
    "hours2": [1, 2, 3, 4],
    "price": [10, 15, 20, 25, 30],
    "money": [50, 75, 100, 150, 200],
    "category1": ["cats", "dogs", "birds", "fish", "reptiles"],
    "category2": ["mammals", "animals", "vertebrates", "pets"],
    "condition1": ["it rains", "the sun shines", "it's cold", "it's hot"],
    "condition2": ["bring umbrella", "wear sunscreen", "wear coat", "drink water"],
    "option1": ["go left", "choose red", "pick A", "say yes"],
    "option2": ["go right", "choose blue", "pick B", "say no"],
    "weather": ["raining", "snowing", "sunny", "windy"],
    "action": ["bring an umbrella", "wear a coat", "go swimming", "fly a kite"],
    "action1": ["heat water", "add sugar", "stir", "pour"],
    "action2": ["boil it", "mix it", "taste it", "serve it"],
    "word1": ["hot", "big", "fast", "light"],
    "word2": ["cold", "small", "slow", "dark"],
    "word3": ["up", "left", "high", "bright"],
    "thing1": ["car", "bicycle", "train"],
    "thing2": ["airplane", "boat", "truck"],
    "concept1": ["learning", "running", "writing"],
    "concept2": ["growing", "racing", "painting"],
    "name": ["Fluffy", "Spot", "Tweety", "Goldie"],
}


def fill_template(template: str) -> str:
    """Fill a template with random placeholders."""
    result = template
    for key, values in PLACEHOLDERS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(random.choice(values)))
    return result


def generate_question(category: str, seed: int | None = None) -> dict[str, str]:
    """Generate a question from a template.

    Args:
        category: Question category
        seed: Random seed for reproducibility

    Returns:
        Dict with question and metadata
    """
    if seed is not None:
        random.seed(seed)

    template = random.choice(QUESTION_TEMPLATES[category])
    question = fill_template(template)

    return {
        "question": question,
        "category": category,
        "template": template,
        "seed": seed,
    }


def generate_answer(
    question: str,
    model: str = "gpt-oss:20b",
    request_explanation: bool = True,
) -> dict[str, str]:
    """Generate answer using LLM.

    Args:
        question: Question to answer
        model: Ollama model to use
        request_explanation: Whether to request step-by-step explanation

    Returns:
        Dict with answer and metadata
    """
    system_prompt = (
        "You are a helpful assistant that provides clear, accurate answers. "
    )

    if request_explanation:
        system_prompt += (
            "When appropriate, show your reasoning step-by-step. "
            "Be concise but complete."
        )

    user_prompt = question

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )

    return {
        "answer": response["message"]["content"],
        "model": model,
        "total_duration_ns": response.get("total_duration", 0),
        "eval_duration_ns": response.get("eval_duration", 0),
    }


def generate_inconsistent_answer(
    question: str,
    correct_answer: str,
    model: str = "gpt-oss:20b",
) -> dict[str, str]:
    """Generate an intentionally inconsistent/wrong answer.

    Args:
        question: Question to answer
        correct_answer: The correct answer (to avoid)
        model: Ollama model to use

    Returns:
        Dict with inconsistent answer
    """
    system_prompt = (
        "You are a flawed assistant that sometimes makes reasoning errors. "
        "Provide an answer that contains a subtle logical flaw or calculation error. "
        "The answer should seem plausible but be incorrect."
    )

    user_prompt = f"Question: {question}\n\nProvide an answer with a reasoning error."

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.7},  # Higher temp for more variation
    )

    return {
        "answer": response["message"]["content"],
        "model": model,
        "is_inconsistent": True,
        "total_duration_ns": response.get("total_duration", 0),
    }


def generate_dataset(
    num_samples: int = 100,
    categories: list[str] | None = None,
    include_inconsistent: bool = True,
    inconsistent_ratio: float = 0.2,
    model: str = "gpt-oss:20b",
    seed: int = 42,
    output_dir: Path = Path("data/text"),
) -> None:
    """Generate synthetic text dataset.

    Args:
        num_samples: Number of samples to generate
        categories: Categories to include (None = all)
        include_inconsistent: Whether to generate inconsistent examples
        inconsistent_ratio: Ratio of inconsistent examples
        model: Ollama model to use
        seed: Random seed
        output_dir: Output directory
    """
    if categories is None:
        categories = list(QUESTION_TEMPLATES.keys())

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🚀 Generating {num_samples} synthetic text samples")
    print(f"Categories: {', '.join(categories)}")
    print(f"Model: {model}")
    print(f"Inconsistent ratio: {inconsistent_ratio if include_inconsistent else 0}")
    print("-" * 60)

    for i in range(num_samples):
        # Select random category
        category = random.choice(categories)

        # Generate question
        question_data = generate_question(category, seed=seed + i)
        question = question_data["question"]

        print(f"\n[{i+1}/{num_samples}] {category}")
        print(f"Q: {question[:80]}...")

        # Generate correct answer
        answer_data = generate_answer(question, model=model)
        correct_answer = answer_data["answer"]

        print(f"A: {correct_answer[:80]}...")

        # Decide if this should be inconsistent
        is_inconsistent = include_inconsistent and random.random() < inconsistent_ratio

        if is_inconsistent:
            # Generate inconsistent answer
            inconsistent_data = generate_inconsistent_answer(
                question, correct_answer, model=model
            )
            sample = {
                "id": f"{timestamp}_{i:04d}",
                "question": question,
                "answer": inconsistent_data["answer"],
                "is_consistent": False,
                "correct_answer": correct_answer,
                "category": category,
                "template": question_data["template"],
                "seed": question_data["seed"],
                "model": model,
                "timestamp": datetime.now().isoformat(),
            }
            print("⚠️  Inconsistent example generated")
        else:
            # Use correct answer
            sample = {
                "id": f"{timestamp}_{i:04d}",
                "question": question,
                "answer": correct_answer,
                "is_consistent": True,
                "category": category,
                "template": question_data["template"],
                "seed": question_data["seed"],
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": answer_data["eval_duration_ns"] / 1e6,
            }

        dataset.append(sample)

    # Save dataset
    output_file = output_dir / f"synthetic_text_{timestamp}.jsonl"

    with open(output_file, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ Dataset saved to: {output_file}")
    print(f"Total samples: {len(dataset)}")

    # Print statistics
    consistent_count = sum(1 for s in dataset if s.get("is_consistent", True))
    inconsistent_count = len(dataset) - consistent_count

    print(f"\nStatistics:")
    print(f"  Consistent: {consistent_count}")
    print(f"  Inconsistent: {inconsistent_count}")
    print(f"  Inconsistent %: {inconsistent_count / len(dataset) * 100:.1f}%")

    # Category breakdown
    category_counts = {}
    for sample in dataset:
        cat = sample["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Save manifest
    manifest = {
        "dataset_file": str(output_file),
        "num_samples": len(dataset),
        "categories": categories,
        "inconsistent_ratio": inconsistent_ratio,
        "model": model,
        "seed": seed,
        "timestamp": timestamp,
        "statistics": {
            "consistent": consistent_count,
            "inconsistent": inconsistent_count,
            "categories": category_counts,
        },
    }

    manifest_file = output_dir / f"manifest_{timestamp}.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📋 Manifest saved to: {manifest_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic text training data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to include",
    )
    parser.add_argument(
        "--no-inconsistent",
        action="store_true",
        help="Disable generation of inconsistent examples",
    )
    parser.add_argument(
        "--inconsistent-ratio",
        type=float,
        default=0.2,
        help="Ratio of inconsistent examples (0.0-1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/text"),
        help="Output directory",
    )

    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        categories=args.categories,
        include_inconsistent=not args.no_inconsistent,
        inconsistent_ratio=args.inconsistent_ratio,
        model=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
