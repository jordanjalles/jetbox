"""LLM Integration - wraps gpt-oss:20b as outer reasoning layer.

The HRM+JEPA system processes text into latent representations, then
an outer LLM (gpt-oss:20b via Ollama) reasons over those representations
to produce final outputs.

Architecture:
    Text → JEPA(text) → latent → HRM(latent) → enriched_latent
    enriched_latent → LLM(context + latent) → final_output
"""

import os
from typing import Any

import torch
from ollama import chat


class LLMWrapper:
    """Wrapper for gpt-oss:20b via Ollama."""

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        temperature: float = 0.2,
    ) -> None:
        """Initialize LLM wrapper.

        Args:
            model: Ollama model name
            temperature: Sampling temperature
        """
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response and metadata
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        )

        return {
            "content": response["message"]["content"],
            "model": response.get("model", self.model),
            "total_duration": response.get("total_duration", 0),
            "load_duration": response.get("load_duration", 0),
            "prompt_eval_duration": response.get("prompt_eval_duration", 0),
            "eval_duration": response.get("eval_duration", 0),
        }


class HRMJEPALLMPipeline:
    """Complete pipeline: Text → JEPA → HRM → LLM.

    This integrates the HRM+JEPA reasoning system with an outer LLM
    for final text generation.
    """

    def __init__(
        self,
        jepa_core: torch.nn.Module,
        hrm_reasoner: torch.nn.Module,
        llm_wrapper: LLMWrapper,
    ) -> None:
        """Initialize pipeline.

        Args:
            jepa_core: JEPA text encoder
            hrm_reasoner: HRM reasoning module
            llm_wrapper: LLM wrapper (gpt-oss:20b)
        """
        self.jepa = jepa_core
        self.hrm = hrm_reasoner
        self.llm = llm_wrapper

        # Set to eval mode
        self.jepa.eval()
        self.hrm.eval()

    def _latent_to_prompt_context(self, latent: torch.Tensor) -> str:
        """Convert latent representation to text context for LLM.

        Args:
            latent: Latent vector (batch=1, latent_dim)

        Returns:
            String representation for LLM context
        """
        # Simple approach: summarize latent statistics
        latent_np = latent.detach().cpu().numpy()[0]

        # Compute statistics
        mean = float(latent_np.mean())
        std = float(latent_np.std())
        norm = float((latent_np**2).sum() ** 0.5)

        context = (
            f"[Reasoning Context]\n"
            f"The system has processed your input through hierarchical reasoning.\n"
            f"Internal representation statistics:\n"
            f"- Mean activation: {mean:.3f}\n"
            f"- Std activation: {std:.3f}\n"
            f"- Representation norm: {norm:.3f}\n"
            f"\nBased on this internal reasoning state, provide your response:\n"
        )

        return context

    def forward(
        self,
        text: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        system_prompt: str | None = None,
        use_hrm_context: bool = True,
    ) -> dict[str, Any]:
        """Full forward pass: Text → JEPA → HRM → LLM.

        Args:
            text: Original text input
            input_ids: Tokenized input
            attention_mask: Attention mask
            system_prompt: Optional system prompt for LLM
            use_hrm_context: Whether to include HRM reasoning in LLM prompt

        Returns:
            Dictionary with outputs from each stage
        """
        with torch.no_grad():
            # Stage 1: JEPA encoding
            jepa_outputs = self.jepa(input_ids=input_ids, attention_mask=attention_mask)
            text_latent = jepa_outputs["text_latent"]

            # Stage 2: HRM reasoning
            hrm_outputs = self.hrm(
                text_latent,
                use_working_memory=True,
                use_abstract_core=True,
                record_trace=True,
            )
            enriched_latent = hrm_outputs["fused"]

            # Stage 3: Prepare LLM prompt
            if use_hrm_context:
                latent_context = self._latent_to_prompt_context(enriched_latent)
                full_prompt = f"{latent_context}\n{text}"
            else:
                full_prompt = text

            # Stage 4: LLM generation
            llm_output = self.llm.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
            )

        return {
            "text_latent": text_latent,
            "hrm_output": hrm_outputs,
            "enriched_latent": enriched_latent,
            "llm_response": llm_output["content"],
            "llm_metadata": {
                "total_duration": llm_output["total_duration"],
                "eval_duration": llm_output["eval_duration"],
            },
            "consistency_score": hrm_outputs.get("consistency_score"),
            "is_inconsistent": hrm_outputs.get("is_inconsistent"),
        }


class ComparisonFramework:
    """Framework for comparing HRM-JEPA-LLM vs baseline LLM."""

    def __init__(
        self,
        hrm_jepa_llm_pipeline: HRMJEPALLMPipeline,
        baseline_llm: LLMWrapper,
    ) -> None:
        """Initialize comparison framework.

        Args:
            hrm_jepa_llm_pipeline: Full HRM-JEPA-LLM pipeline
            baseline_llm: Baseline LLM (same model, no HRM-JEPA)
        """
        self.pipeline = hrm_jepa_llm_pipeline
        self.baseline = baseline_llm

    def compare(
        self,
        text: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Compare HRM-JEPA-LLM vs baseline on same input.

        Args:
            text: Input text
            input_ids: Tokenized input
            attention_mask: Attention mask
            system_prompt: System prompt

        Returns:
            Comparison results
        """
        # Run HRM-JEPA-LLM pipeline
        pipeline_output = self.pipeline.forward(
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            system_prompt=system_prompt,
            use_hrm_context=True,
        )

        # Run baseline (no HRM-JEPA context)
        baseline_output = self.baseline.generate(
            prompt=text,
            system_prompt=system_prompt,
        )

        return {
            "input": text,
            "hrm_jepa_llm": {
                "response": pipeline_output["llm_response"],
                "consistency_score": pipeline_output.get("consistency_score"),
                "is_inconsistent": pipeline_output.get("is_inconsistent"),
                "duration_ms": pipeline_output["llm_metadata"]["eval_duration"] / 1e6,
            },
            "baseline_llm": {
                "response": baseline_output["content"],
                "duration_ms": baseline_output["eval_duration"] / 1e6,
            },
            "hrm_status": self.pipeline.hrm.get_status(),
        }

    def batch_compare(
        self,
        test_cases: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run comparison on batch of test cases.

        Args:
            test_cases: List of test cases with 'text', 'input_ids', 'attention_mask'
            system_prompt: System prompt for all cases

        Returns:
            List of comparison results
        """
        results = []

        for case in test_cases:
            result = self.compare(
                text=case["text"],
                input_ids=case["input_ids"],
                attention_mask=case.get("attention_mask"),
                system_prompt=system_prompt,
            )
            results.append(result)

        return results


def create_text_only_pipeline(
    latent_dim: int = 512,
    model: str = "gpt-oss:20b",
) -> tuple[HRMJEPALLMPipeline, LLMWrapper]:
    """Create text-only HRM-JEPA-LLM pipeline.

    Args:
        latent_dim: Latent dimension
        model: Ollama model name

    Returns:
        Tuple of (pipeline, baseline_llm) for comparison
    """
    from core.encoders import create_text_transformer_lite
    from core.hrm import create_hrm_lite
    from core.jepa_core import JEPACore

    # Create text-only JEPA (no vision encoder)
    text_encoder = create_text_transformer_lite(latent_dim=latent_dim)
    jepa_core = JEPACore(
        vision_encoder=None,  # No vision for text-only
        text_encoder=text_encoder,
        latent_dim=latent_dim,
    )

    # Create HRM
    hrm_reasoner = create_hrm_lite(latent_dim=latent_dim)

    # Create LLM wrappers
    llm_wrapper = LLMWrapper(model=model)
    baseline_llm = LLMWrapper(model=model)

    # Create pipeline
    pipeline = HRMJEPALLMPipeline(jepa_core, hrm_reasoner, llm_wrapper)

    return pipeline, baseline_llm
