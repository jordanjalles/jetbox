"""Uncertainty detection and concept highlighting for user questions.

This module identifies:
1. Areas where the model is uncertain (low confidence)
2. Concepts that are causing mistakes (error patterns)
3. Questions the model should ask the user for clarification

Uncertainty sources:
- Low consistency scores from reflection loop
- High latent variance (disagreement between working memory & abstract core)
- Repeated failures on specific concept types
- Conflicting predictions
"""

from typing import Any

import torch
import torch.nn as nn


class UncertaintyDetector(nn.Module):
    """Detects uncertainty and generates user questions.

    Analyzes HRM outputs to identify:
    - Low-confidence predictions (consistency < threshold)
    - High-variance latents (disagreement between components)
    - Concept-specific error patterns
    """

    def __init__(
        self,
        latent_dim: int = 512,
        consistency_threshold: float = 0.7,
        variance_threshold: float = 1.0,
    ) -> None:
        """Initialize uncertainty detector.

        Args:
            latent_dim: Dimension of latent space
            consistency_threshold: Threshold for flagging low consistency
            variance_threshold: Threshold for flagging high variance
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.consistency_threshold = consistency_threshold
        self.variance_threshold = variance_threshold

        # Concept classifier: maps latent → concept categories
        self.concept_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 4, 32),  # 32 concept categories
        )

        # Concept names (will be learned from data)
        self.concept_names = [
            "arithmetic_addition",
            "arithmetic_subtraction",
            "arithmetic_multiplication",
            "arithmetic_division",
            "logic_deduction",
            "logic_induction",
            "logic_syllogism",
            "logic_contradiction",
            "reasoning_causal",
            "reasoning_temporal",
            "reasoning_spatial",
            "reasoning_analogical",
            "language_grammar",
            "language_semantics",
            "language_pragmatics",
            "language_ambiguity",
            "common_sense_physics",
            "common_sense_social",
            "common_sense_temporal",
            "common_sense_spatial",
            "math_word_problem",
            "math_geometry",
            "math_algebra",
            "math_statistics",
            "knowledge_factual",
            "knowledge_procedural",
            "knowledge_conceptual",
            "knowledge_metacognitive",
            "uncertainty_epistemic",  # Don't know the answer
            "uncertainty_aleatoric",  # Answer is inherently uncertain
            "uncertainty_ambiguous",  # Question is ambiguous
            "uncertainty_conflicting",  # Conflicting information
        ]

    def forward(
        self,
        hrm_output: dict[str, torch.Tensor],
        consistency_score: float,
    ) -> dict[str, Any]:
        """Detect uncertainty and generate user questions.

        Args:
            hrm_output: Output from HRM reasoner
            consistency_score: Consistency score from reflection loop

        Returns:
            Dict with uncertainty analysis and suggested user questions
        """
        enriched_latent = hrm_output["fused"]  # (batch, latent_dim)

        # 1. Consistency-based uncertainty
        is_low_consistency = consistency_score < self.consistency_threshold
        consistency_uncertainty = max(0.0, self.consistency_threshold - consistency_score)

        # 2. Variance-based uncertainty
        # Compare working memory vs abstract core
        wm_latent = hrm_output.get("working_memory", enriched_latent)
        ac_latent = hrm_output.get("abstract_core", enriched_latent)

        # Compute disagreement (L2 distance normalized)
        disagreement = torch.norm(wm_latent - ac_latent, dim=-1).mean().item()
        variance_uncertainty = min(1.0, disagreement / self.variance_threshold)

        # 3. Concept classification
        concept_logits = self.concept_classifier(enriched_latent)  # (batch, 32)
        concept_probs = torch.softmax(concept_logits, dim=-1)  # (batch, 32)

        # Get top 3 concepts
        top_k = 3
        top_probs, top_indices = torch.topk(concept_probs, top_k, dim=-1)

        detected_concepts = []
        for i in range(top_k):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            detected_concepts.append({
                "concept": self.concept_names[idx],
                "confidence": prob,
            })

        # 4. Overall uncertainty score (weighted combination)
        overall_uncertainty = (
            0.4 * consistency_uncertainty +
            0.3 * variance_uncertainty +
            0.3 * (1.0 - detected_concepts[0]["confidence"])  # Low concept confidence
        )

        # 5. Generate user questions based on uncertainty type
        user_questions = []

        if is_low_consistency:
            user_questions.append({
                "type": "consistency_check",
                "priority": "high",
                "question": "I'm not confident about my reasoning on this. Can you verify if my answer makes sense?",
                "context": f"Consistency score: {consistency_score:.3f}",
            })

        if variance_uncertainty > 0.5:
            user_questions.append({
                "type": "component_disagreement",
                "priority": "medium",
                "question": "My fast intuition and deep knowledge disagree on this. Which approach seems more appropriate?",
                "context": f"Disagreement level: {variance_uncertainty:.3f}",
            })

        # Concept-specific questions
        uncertain_concepts = [c for c in detected_concepts if c["confidence"] < 0.5]
        if uncertain_concepts:
            concept_list = ", ".join([c["concept"].replace("_", " ") for c in uncertain_concepts])
            user_questions.append({
                "type": "concept_uncertainty",
                "priority": "medium",
                "question": f"I'm uncertain about these concepts: {concept_list}. Can you provide more context or examples?",
                "context": f"Low confidence on: {len(uncertain_concepts)} concepts",
            })

        # High-level uncertainty (epistemic)
        if overall_uncertainty > 0.7:
            user_questions.append({
                "type": "epistemic_uncertainty",
                "priority": "high",
                "question": "I don't have enough information to answer confidently. What additional context would help?",
                "context": f"Overall uncertainty: {overall_uncertainty:.3f}",
            })

        return {
            "overall_uncertainty": overall_uncertainty,
            "consistency_uncertainty": consistency_uncertainty,
            "variance_uncertainty": variance_uncertainty,
            "is_uncertain": overall_uncertainty > 0.5,
            "detected_concepts": detected_concepts,
            "uncertain_concepts": uncertain_concepts,
            "user_questions": user_questions,
            "uncertainty_breakdown": {
                "consistency": consistency_uncertainty,
                "variance": variance_uncertainty,
                "concept_confidence": 1.0 - detected_concepts[0]["confidence"],
            },
        }


class ErrorPatternTracker:
    """Tracks error patterns to identify problematic concepts.

    Maintains a running history of:
    - Which concepts have high error rates
    - Which concept combinations cause failures
    - Temporal patterns (recent vs historical errors)
    """

    def __init__(
        self,
        max_history: int = 1000,
        error_threshold: float = 0.3,
    ) -> None:
        """Initialize error tracker.

        Args:
            max_history: Maximum number of examples to track
            error_threshold: Error rate threshold for flagging concepts
        """
        self.max_history = max_history
        self.error_threshold = error_threshold
        self.history: list[dict[str, Any]] = []

    def record(
        self,
        concepts: list[str],
        is_error: bool,
        consistency_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an example.

        Args:
            concepts: Detected concepts
            is_error: Whether this was an error
            consistency_score: Consistency score
            metadata: Additional metadata
        """
        record = {
            "concepts": concepts,
            "is_error": is_error,
            "consistency_score": consistency_score,
            "metadata": metadata or {},
        }

        self.history.append(record)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def analyze_patterns(self) -> dict[str, Any]:
        """Analyze error patterns across history.

        Returns:
            Dict with error pattern analysis
        """
        if not self.history:
            return {
                "total_examples": 0,
                "error_rate": 0.0,
                "problematic_concepts": [],
                "concept_error_rates": {},
            }

        # Overall statistics
        total_examples = len(self.history)
        total_errors = sum(1 for h in self.history if h["is_error"])
        error_rate = total_errors / total_examples

        # Per-concept error rates
        concept_counts = {}
        concept_errors = {}

        for record in self.history:
            for concept in record["concepts"]:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
                if record["is_error"]:
                    concept_errors[concept] = concept_errors.get(concept, 0) + 1

        concept_error_rates = {
            concept: concept_errors.get(concept, 0) / count
            for concept, count in concept_counts.items()
        }

        # Identify problematic concepts
        problematic_concepts = [
            {
                "concept": concept,
                "error_rate": error_rate,
                "count": concept_counts[concept],
                "errors": concept_errors.get(concept, 0),
            }
            for concept, error_rate in concept_error_rates.items()
            if error_rate > self.error_threshold
        ]

        # Sort by error rate
        problematic_concepts.sort(key=lambda x: x["error_rate"], reverse=True)

        # Concept co-occurrence analysis (which concepts fail together?)
        co_occurrence_errors = {}
        for record in self.history:
            if record["is_error"] and len(record["concepts"]) > 1:
                concepts = tuple(sorted(record["concepts"]))
                co_occurrence_errors[concepts] = co_occurrence_errors.get(concepts, 0) + 1

        problematic_combinations = [
            {"concepts": list(concepts), "error_count": count}
            for concepts, count in co_occurrence_errors.items()
            if count > 2  # At least 3 failures together
        ]
        problematic_combinations.sort(key=lambda x: x["error_count"], reverse=True)

        return {
            "total_examples": total_examples,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "concept_error_rates": concept_error_rates,
            "problematic_concepts": problematic_concepts[:10],  # Top 10
            "problematic_combinations": problematic_combinations[:5],  # Top 5
        }

    def get_user_feedback_suggestions(self) -> list[dict[str, Any]]:
        """Generate suggestions for user feedback based on error patterns.

        Returns:
            List of suggestions for user questions/corrections
        """
        analysis = self.analyze_patterns()

        if not analysis["problematic_concepts"]:
            return []

        suggestions = []

        # High error rate concepts need training
        for concept_info in analysis["problematic_concepts"][:3]:  # Top 3
            suggestions.append({
                "type": "concept_training_needed",
                "priority": "high",
                "concept": concept_info["concept"],
                "error_rate": concept_info["error_rate"],
                "suggestion": f"I frequently make mistakes with '{concept_info['concept'].replace('_', ' ')}' "
                              f"(error rate: {concept_info['error_rate']:.1%}). "
                              f"Can you provide more examples or corrections for this concept?",
            })

        # Concept combinations that fail together
        for combo_info in analysis["problematic_combinations"][:2]:  # Top 2
            concept_list = ", ".join([c.replace("_", " ") for c in combo_info["concepts"]])
            suggestions.append({
                "type": "concept_combination_issue",
                "priority": "medium",
                "concepts": combo_info["concepts"],
                "error_count": combo_info["error_count"],
                "suggestion": f"I struggle when combining these concepts: {concept_list}. "
                              f"This has caused {combo_info['error_count']} mistakes. "
                              f"Can you help me understand how these concepts interact?",
            })

        return suggestions


def create_uncertainty_system(
    latent_dim: int = 512,
    consistency_threshold: float = 0.7,
) -> tuple[UncertaintyDetector, ErrorPatternTracker]:
    """Create uncertainty detection system.

    Args:
        latent_dim: Latent dimension
        consistency_threshold: Threshold for flagging uncertainty

    Returns:
        Tuple of (detector, tracker)
    """
    detector = UncertaintyDetector(
        latent_dim=latent_dim,
        consistency_threshold=consistency_threshold,
    )

    tracker = ErrorPatternTracker(
        max_history=1000,
        error_threshold=0.3,
    )

    return detector, tracker
