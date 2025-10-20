# Text-Only + LLM Integration Update

**Date:** 2025-10-20
**Status:** ✅ Text-only scope + gpt-oss:20b integration complete

## Scope Change Summary

Based on user feedback: "let's scope to exclusively text for now. keep in mind local storage is limited. The HRM+JEPA needs to be integrated with a baseline LLM as the outer layer - gpt-oss - and compare their outputs (HRM-JEPA-LLM vs LLM) to judge improvements/loss"

### Key Changes

1. **Text-Only Mode:** Vision components remain in codebase but are unused
2. **LLM Integration:** gpt-oss:20b via Ollama as outer reasoning layer
3. **Comparison Framework:** A/B testing HRM-JEPA-LLM vs baseline LLM
4. **Storage Management:** Compression & cleanup for limited disk space

## Architecture

```
Text Input
    │
    ▼
┌─────────────────────────────────┐
│   JEPA Text Encoder             │
│   (Transformer)                 │
│   → text_latent (512-dim)       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│            HRM Reasoning Head               │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │   Working    │      │  Abstract Core  │ │
│  │   Memory     │      │  (gated update) │ │
│  │  (fast LoRA) │      │    (slow)       │ │
│  └──────┬───────┘      └────────┬────────┘ │
│         │                       │          │
│         └───────────┬───────────┘          │
│                     │                      │
│         enriched_latent (512-dim)         │
└─────────────────────┼──────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  LLM Outer Layer       │
         │  (gpt-oss:20b)         │
         │  + latent context      │
         │  → final response      │
         └────────────────────────┘
```

**Flow:** Text → JEPA → HRM → LLM(context + latent) → Output

## New Files Created

### 1. core/llm_integration.py (336 lines)

**Purpose:** Integrate HRM+JEPA with gpt-oss:20b and provide comparison framework

**Classes:**

```python
class LLMWrapper:
    """Wrapper for gpt-oss:20b via Ollama."""
    - generate(prompt, system_prompt, max_tokens) -> dict
    - Uses Ollama chat API
    - Returns response + timing metadata

class HRMJEPALLMPipeline:
    """Complete pipeline: Text → JEPA → HRM → LLM."""
    - forward(text, input_ids, attention_mask, system_prompt, use_hrm_context) -> dict
    - Stage 1: JEPA text encoding
    - Stage 2: HRM reasoning (working memory + abstract core + reflection)
    - Stage 3: Convert enriched latent to text context for LLM
    - Stage 4: LLM generation with HRM context
    - Returns: text_latent, hrm_output, enriched_latent, llm_response, consistency_score

class ComparisonFramework:
    """Framework for comparing HRM-JEPA-LLM vs baseline LLM."""
    - compare(text, input_ids, attention_mask, system_prompt) -> dict
    - Runs both pipelines on same input
    - Returns side-by-side results with timing and consistency scores
    - batch_compare() for running multiple test cases

def create_text_only_pipeline(latent_dim=512, model="gpt-oss:20b"):
    """Factory function to create text-only pipeline + baseline LLM."""
    - Creates JEPA with vision_encoder=None
    - Creates HRM reasoner
    - Returns (pipeline, baseline_llm) for comparison
```

**Latent-to-Context Strategy:**

The `_latent_to_prompt_context()` method converts the HRM enriched latent (512-dim vector) into a text context for the LLM. Current approach:

```
[Reasoning Context]
The system has processed your input through hierarchical reasoning.
Internal representation statistics:
- Mean activation: 0.123
- Std activation: 0.456
- Representation norm: 12.345

Based on this internal reasoning state, provide your response:
```

This is a simple baseline. Future improvements could:
- Train a latent-to-text decoder
- Use clustering to identify semantic regions
- Map latent dimensions to natural language concepts

### 2. core/jepa_core.py (MODIFIED)

**Changes:**

```python
def __init__(
    self,
    vision_encoder: VisionViT | None,  # NOW OPTIONAL
    text_encoder: TextTransformer,
    latent_dim: int = 512,
    ...
):
    self.text_only = vision_encoder is None  # NEW FLAG

def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
    if self.text_only:  # NEW CHECK
        raise ValueError("Vision encoder not available in text-only mode")
    # ... rest of method
```

### 3. tools/compact_storage.py (229 lines)

**Purpose:** Manage limited disk space with compression and cleanup strategies

**Functions:**

```python
def quantize_checkpoint(checkpoint_path, output_path, quantize_level="int8"):
    """Quantize checkpoint to int8 for 75% size reduction.

    - Loads checkpoint
    - Converts float32 tensors to int8 with scale factor
    - Saves quantized checkpoint
    - Returns size reduction stats
    """

def compress_logs(log_dir, keep_recent=10):
    """Compress old log files with gzip.

    - Keeps N most recent logs uncompressed
    - Compresses older logs with gzip
    - Deletes original uncompressed files
    - Returns compression stats
    """

def cleanup_old_checkpoints(checkpoint_dir, keep_best=3, keep_recent=5, metric_key="loss"):
    """Clean up old checkpoints, keeping only best and most recent.

    - Loads metadata from all checkpoints
    - Sorts by metric (lower is better)
    - Sorts by modification time
    - Keeps union of best N and recent N
    - Deletes everything else
    - Returns space freed stats
    """

def estimate_storage_usage(project_root):
    """Estimate storage usage by component.

    - Calculates size of checkpoints/
    - Calculates size of logs/
    - Calculates size of data/
    - Returns breakdown in MB
    """

def suggest_cleanup_actions(project_root, storage_limit_mb=1000):
    """Suggest cleanup actions based on current usage.

    - Checks if over limit
    - Suggests quantization if checkpoints > 500MB
    - Suggests compression if logs > 100MB
    - Returns list of action strings
    """
```

**Storage Strategies:**

1. **int8 quantization:** 75% size reduction for checkpoints
2. **gzip compression:** ~70% reduction for text logs
3. **Selective retention:** Keep only best + recent checkpoints
4. **Trace buffer limit:** Max 1000 entries in reflection traces

### 4. scripts/compare_hrm_jepa_llm.py (265 lines)

**Purpose:** Run comparison tests between HRM-JEPA-LLM and baseline

**Test Cases:**

```python
[
    {"name": "simple_reasoning", "text": "What is 15 + 27?", "category": "arithmetic"},
    {"name": "multi_step_reasoning", "text": "If Alice has 3 apples and Bob gives her twice as many, then Charlie takes 4, how many does Alice have?", "category": "math_word_problem"},
    {"name": "logical_consistency", "text": "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?", "category": "logic"},
    {"name": "chain_of_thought", "text": "A train travels 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?", "category": "math_reasoning"},
    {"name": "common_sense", "text": "If it's raining outside, should I bring an umbrella?", "category": "common_sense"},
]
```

**Outputs:**

- JSON file with timestamp: `comparison_results/comparison_20251020_123456.json`
- Each result includes:
  - Input text
  - HRM-JEPA-LLM response + consistency score + duration
  - Baseline LLM response + duration
  - HRM status (working memory steps, abstract core updates, reflection traces)
- Summary statistics:
  - Success rate
  - Average duration for each pipeline
  - Overhead of HRM-JEPA layer
  - Average consistency score

## Usage

### Run Comparison

```bash
cd /workspace/hrm-jepa

# Activate conda environment
conda activate hrm-jepa

# Run comparison (requires Ollama with gpt-oss:20b)
python scripts/compare_hrm_jepa_llm.py --model gpt-oss:20b --output-dir comparison_results

# View results
cat comparison_results/comparison_*.json
```

### Storage Management

```bash
# Check usage
python -c "from tools.compact_storage import suggest_cleanup_actions; print('\n'.join(suggest_cleanup_actions('.', storage_limit_mb=1000)))"

# Quantize a checkpoint
python -c "from tools.compact_storage import quantize_checkpoint; stats = quantize_checkpoint('checkpoints/model.pth', 'checkpoints/model_int8.pth'); print(f\"Reduced by {stats['reduction_pct']:.1f}%\")"
```

## Testing Status

**New components NOT yet tested** (require PyTorch + Ollama):
- core/llm_integration.py
- scripts/compare_hrm_jepa_llm.py
- tools/compact_storage.py

**Existing tests still pass:**
- tests/unit/test_smoke.py (5 tests) ✅
- tests/unit/test_imports.py (3 tests) ✅

**Pending tests** (require conda environment):
- tests/unit/test_encoders.py (8 tests)
- tests/unit/test_jepa_core.py (11 tests)
- tests/unit/test_hrm.py (14 tests)

## Dependencies

**New dependencies:**
- ollama (Python package) - Already in environment.yml
- ollama (binary) - User must install from https://ollama.ai
- gpt-oss:20b model - User must pull with `ollama pull gpt-oss:20b`

**No other changes to dependencies.**

## Performance Expectations

### HRM-JEPA-LLM Pipeline

**Total latency breakdown:**
1. JEPA encoding: ~10-20ms (text transformer)
2. HRM reasoning: ~30-50ms (working memory + abstract core + reflection)
3. LLM generation: ~2000-5000ms (depends on gpt-oss:20b and sequence length)

**Total:** ~2-5 seconds per query

### Baseline LLM

**Total latency:**
- LLM generation: ~2000-5000ms (same as above)

**Overhead:** +40-70ms for HRM-JEPA layer (1-3% of total latency)

### Storage Targets

- Checkpoints: <500MB (use quantization if exceeds)
- Logs: <100MB (use compression if exceeds)
- Data: <400MB (synthetic text only)
- **Total:** <1000MB (1GB)

## Next Steps

1. **Test the comparison framework**
   - Requires conda environment with PyTorch
   - Requires Ollama running with gpt-oss:20b
   - Run `python scripts/compare_hrm_jepa_llm.py`

2. **Analyze results**
   - Does HRM-JEPA improve response quality?
   - What is the actual latency overhead?
   - Are consistency scores useful?

3. **Iterate on latent-to-context conversion**
   - Current approach is simple statistics
   - Could train a decoder
   - Could map dimensions to concepts

4. **Generate synthetic text data** (M1 milestone)
   - Self-play conversations
   - Chain-of-thought reasoning
   - Instruction tuning data

5. **Train JEPA on synthetic data** (M2 milestone)
   - Contrastive learning on text pairs
   - Latent prediction objectives

6. **Fine-tune HRM on reasoning tasks** (M3 milestone)
   - Working memory adaptation
   - Abstract core updates (with human approval)

## Implementation Notes

**Text-Only JEPA:**
- Vision encoder is still in codebase (`core/encoders/vision_vit.py`)
- Not loaded in text-only mode
- Can be re-enabled for future multimodal expansion
- JEPACore checks `self.text_only` flag before vision operations

**LLM Context Injection:**
- Enriched latent is converted to text statistics
- Statistics are prepended to user prompt
- LLM sees both original text + reasoning context
- `use_hrm_context=False` flag allows baseline comparison

**Comparison Framework:**
- Same tokenizer used for both pipelines
- Same system prompt (if provided)
- Side-by-side results in single JSON file
- Can batch process multiple test cases

**Storage Management:**
- All functions are pure (no side effects unless explicitly requested)
- Quantization creates new file, doesn't modify original
- Compression creates .gz files, then deletes originals
- Cleanup uses union of best + recent to avoid data loss

## Commit Summary

**Files modified:**
- core/jepa_core.py (added text-only mode support)
- README.md (updated for text-only + LLM integration)

**Files created:**
- core/llm_integration.py (336 lines)
- tools/compact_storage.py (229 lines)
- scripts/compare_hrm_jepa_llm.py (265 lines)
- TEXT_ONLY_LLM_UPDATE.md (this file)

**Total new code:** ~830 lines

**Git commit message:**
```
Add text-only mode + gpt-oss:20b LLM integration

- Scope to text-only processing (vision encoder optional)
- Integrate gpt-oss:20b via Ollama as outer reasoning layer
- Add comparison framework (HRM-JEPA-LLM vs baseline)
- Add storage management tools (quantization, compression, cleanup)
- Update documentation for text-only architecture

This addresses limited storage constraints and enables
measurement of HRM-JEPA improvement over baseline LLM.

Flow: Text → JEPA → HRM → LLM(context + latent) → Output

New files:
- core/llm_integration.py (LLM wrapper, pipeline, comparison)
- tools/compact_storage.py (compression & cleanup utilities)
- scripts/compare_hrm_jepa_llm.py (comparison test script)
- TEXT_ONLY_LLM_UPDATE.md (this update summary)

Modified files:
- core/jepa_core.py (text-only mode support)
- README.md (updated architecture & usage)
```

---

**Status:** Ready to commit and push to GitHub
