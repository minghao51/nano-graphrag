"""DSPy prompt tuning CLI for optimizing entity extraction prompts.

This module provides a command-line interface for using DSPy to optimize
entity extraction prompts for smaller/cheaper models while maintaining quality.

Usage:
    python -m bench.dspy_tune --dataset musique --model qwen2.5:7b --output prompts/optimized.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def create_dspy_tuner(
    train_examples: List[Any],
    model: str = "gpt-4o-mini",
    max_bootstrapped_demos: int = 4,
) -> str:
    """Create a DSPy tuner for entity extraction prompts.

    Args:
        train_examples: List of training examples for DSPy.
        model: Model to use for tuning.
        max_bootstrapped_demos: Maximum number of few-shot examples.

    Returns:
        Optimized prompt string.

    Raises:
        ImportError: If DSPy is not installed.
    """
    try:
        import dspy
    except ImportError:
        raise ImportError("DSPy is required for prompt tuning. Install with: uv add dspy-ai")

    # Configure DSPy with the specified model
    llm = dspy.LM(model)
    dspy.configure(lm=llm)

    # Define the entity extraction signature
    class EntityExtractionSignature(dspy.Signature):
        """Extract entities and relations from a text chunk as JSON."""

        chunk: str = dspy.InputField(desc="Text chunk to extract entities from")
        entities_json: str = dspy.OutputField(desc="JSON object with entities and relations")

    # Create the DSPy module
    module = dspy.Predict(EntityExtractionSignature)

    # Use BootstrapFewShot to optimize the prompt
    teleprompter = dspy.BootstrapFewShot(
        metric=entity_extraction_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
    )

    # Compile the optimized module
    compiled = teleprompter.compile(module, trainset=train_examples)

    # Extract the optimized prompt
    return compiled.extended_signature.instructions


def entity_extraction_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Metric for evaluating entity extraction quality.

    Args:
        example: Ground truth example.
        prediction: Model prediction.
        trace: Execution trace (optional).

    Returns:
        Score between 0 and 1.
    """
    try:
        # Parse the prediction as JSON
        predicted_entities = json.loads(prediction.entities_json)
        expected_entities = json.loads(example.entities_json)

        # Calculate simple overlap metric
        predicted_names = {e.get("name", e.get("entity_name", "")) for e in predicted_entities}
        expected_names = {e.get("name", e.get("entity_name", "")) for e in expected_entities}

        if not expected_names:
            return 1.0 if not predicted_names else 0.0

        overlap = predicted_names & expected_names
        return len(overlap) / len(expected_names)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return 0.0


def generate_training_examples(
    dataset_path: str,
    num_examples: int = 50,
) -> List[Any]:
    """Generate training examples from dataset.

    Args:
        dataset_path: Path to dataset file.
        num_examples: Number of examples to generate.

    Returns:
        List of DSPy Example objects.
    """
    try:
        import dspy
    except ImportError:
        raise ImportError("DSPy is required for prompt tuning. Install with: uv add dspy-ai")

    examples = []

    # Load dataset (simplified - in practice would load from actual data)
    with open(dataset_path, "r") as f:
        data = json.load(f)

    for i, item in enumerate(data[:num_examples]):
        # Create DSPy Example
        example = dspy.Example(
            chunk=item.get("chunk", item.get("text", "")),
            entities_json=item.get("entities", "{}"),
        )

        examples.append(example)

    return examples


def main():
    """CLI entry point for DSPy prompt tuning."""
    parser = argparse.ArgumentParser(description="Tune entity extraction prompts using DSPy")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSON file with chunk/entities pairs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for tuning (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prompts/optimized_entity_extraction.txt",
        help="Output path for optimized prompt (default: prompts/optimized_entity_extraction.txt)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of training examples to use (default: 50)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Maximum number of few-shot demonstrations (default: 4)",
    )

    args = parser.parse_args()

    # Generate training examples
    print(f"Loading training examples from {args.dataset}...")
    train_examples = generate_training_examples(args.dataset, args.num_examples)
    print(f"Loaded {len(train_examples)} training examples")

    # Tune the prompt
    print(f"Tuning prompt with model {args.model}...")
    optimized_prompt = create_dspy_tuner(
        train_examples,
        model=args.model,
        max_bootstrapped_demos=args.max_demos,
    )

    # Save the optimized prompt
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(optimized_prompt)

    print(f"Optimized prompt saved to {output_path}")
    print(f"Prompt length: {len(optimized_prompt)} characters")


if __name__ == "__main__":
    main()
