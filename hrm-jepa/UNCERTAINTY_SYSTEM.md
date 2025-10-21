# Uncertainty Detection & User Question System

**Date:** 2025-10-20
**Status:** ✅ Implemented and tested

## Overview

The HRM-JEPA system now includes comprehensive uncertainty detection that:
1. **Identifies areas where the model is uncertain**
2. **Extracts concepts causing mistakes**
3. **Generates specific questions for user feedback**
4. **Tracks error patterns over time**

This enables the model to **proactively ask for help** instead of silently making mistakes.

## Architecture

```
Text → JEPA → HRM → [Uncertainty Detector] → User Questions
                ↓
        Enriched Latent
                ↓
             LLM → Response + Uncertainty Analysis
```

## Components

### 1. UncertaintyDetector (core/uncertainty.py)

**Purpose:** Analyzes HRM outputs to detect uncertainty and generate user questions

**Inputs:**
- `hrm_output`: Dict with working memory, abstract core, and fused latents
- `consistency_score`: Score from reflection loop (0-1)

**Outputs:**
- `overall_uncertainty`: Combined uncertainty score (0-1)
- `detected_concepts`: Top 3 concepts with confidence scores
- `uncertain_concepts`: Concepts below 50% confidence
- `user_questions`: List of specific questions for the user

**Uncertainty Sources:**

1. **Consistency-based** (40% weight)
   - From HRM reflection loop
   - Low consistency = uncertain reasoning
   - Threshold: 0.7

2. **Variance-based** (30% weight)
   - Disagreement between working memory & abstract core
   - High disagreement = components pulling different directions
   - Measured as L2 distance between latents

3. **Concept confidence** (30% weight)
   - Neural classifier: latent → 32 concept categories
   - Low confidence on primary concept = uncertain domain
   - Softmax over concept logits

**Concept Categories (32 total):**
- Arithmetic (add, subtract, multiply, divide)
- Logic (deduction, induction, syllogism, contradiction)
- Reasoning (causal, temporal, spatial, analogical)
- Language (grammar, semantics, pragmatics, ambiguity)
- Common sense (physics, social, temporal, spatial)
- Math (word problems, geometry, algebra, statistics)
- Knowledge (factual, procedural, conceptual, metacognitive)
- Uncertainty types (epistemic, aleatoric, ambiguous, conflicting)

### 2. ErrorPatternTracker (core/uncertainty.py)

**Purpose:** Tracks error patterns to identify problematic concepts over time

**Features:**
- Circular buffer (max 1000 examples)
- Per-concept error rates
- Concept co-occurrence analysis (which concepts fail together)
- Temporal tracking (recent vs historical errors)

**Methods:**
- `record(concepts, is_error, consistency_score, metadata)` - Record an example
- `analyze_patterns()` - Analyze error rates and problematic concepts
- `get_user_feedback_suggestions()` - Generate feedback requests

**Outputs:**
- Problematic concepts (error rate > 30%)
- Problematic combinations (concepts that fail together)
- Suggestions for user training/corrections

### 3. User Question Generation

**Question Types:**

1. **Consistency Check** (Priority: HIGH)
   ```
   "I'm not confident about my reasoning on this. Can you verify if my answer makes sense?"
   Context: Consistency score: 0.45
   ```
   - Triggered when: consistency < 0.7
   - User action: Verify answer, provide correction

2. **Component Disagreement** (Priority: MEDIUM)
   ```
   "My fast intuition and deep knowledge disagree on this. Which approach seems more appropriate?"
   Context: Disagreement level: 0.85
   ```
   - Triggered when: variance > 0.5
   - User action: Choose which component is right, explain why

3. **Concept Uncertainty** (Priority: MEDIUM)
   ```
   "I'm uncertain about these concepts: arithmetic division, logic deduction. Can you provide more context or examples?"
   Context: Low confidence on: 2 concepts
   ```
   - Triggered when: concept confidence < 0.5
   - User action: Provide examples, explanations, corrections

4. **Epistemic Uncertainty** (Priority: HIGH)
   ```
   "I don't have enough information to answer confidently. What additional context would help?"
   Context: Overall uncertainty: 0.75
   ```
   - Triggered when: overall uncertainty > 0.7
   - User action: Provide more context, rephrase question, clarify

## Integration

### In LLM Pipeline

```python
from core.llm_integration import create_text_only_pipeline

# Create pipeline with uncertainty enabled
pipeline, baseline = create_text_only_pipeline(
    latent_dim=512,
    model="gpt-oss:20b",
    enable_uncertainty=True,  # Enable uncertainty detection
)

# Run inference
result = pipeline.forward(
    text="What is 15 + 27?",
    input_ids=input_ids,
    attention_mask=attention_mask,
)

# Access uncertainty analysis
if result["uncertainty_analysis"]:
    uncertainty = result["uncertainty_analysis"]

    print(f"Overall uncertainty: {uncertainty['overall_uncertainty']:.2%}")

    # Display user questions
    for question in uncertainty["user_questions"]:
        print(f"{question['priority']}: {question['question']}")
```

### In Training Loop

```python
from core.uncertainty import ErrorPatternTracker

# Create tracker
tracker = ErrorPatternTracker(max_history=1000, error_threshold=0.3)

# During training/evaluation
for example in dataset:
    result = pipeline.forward(...)

    # Extract concepts
    concepts = [c["concept"] for c in result["uncertainty_analysis"]["detected_concepts"]]

    # Determine if error (compare to ground truth)
    is_error = result["llm_response"] != example["ground_truth"]

    # Record
    tracker.record(
        concepts=concepts,
        is_error=is_error,
        consistency_score=result["consistency_score"],
    )

# Analyze patterns
patterns = tracker.analyze_patterns()
print(f"Problematic concepts: {patterns['problematic_concepts']}")

# Get feedback suggestions
suggestions = tracker.get_user_feedback_suggestions()
for s in suggestions:
    print(f"{s['priority']}: {s['suggestion']}")
```

## Test Results

**Test run:** `python scripts/test_uncertainty.py --model gpt-oss:20b`

**Findings:**

1. **All questions flagged as uncertain** (~67% uncertainty)
   - Consistency: ~21% (low, as expected - untrained)
   - Variance: 100% (working memory & abstract core fully disagree)
   - Concept confidence: ~3% (untrained classifier)

2. **Concept detection is random** (untrained)
   - All concepts get similar low scores (~3.4%)
   - This is expected - concept classifier needs training

3. **User questions are generated correctly**
   - Consistency check (consistency < 0.7)
   - Component disagreement (variance = 1.0)
   - Concept uncertainty (all concepts < 0.5)

4. **System is functional** but needs training
   - Infrastructure works end-to-end
   - Uncertainty detection pipeline is active
   - Question generation logic is sound

## Training Requirements

### 1. Concept Classifier Training

**Data needed:**
- Text samples with ground-truth concept labels
- Format: `{"text": "...", "concepts": ["arithmetic_addition", "math_word_problem"]}`
- Size: ~1000+ labeled examples

**Training objective:**
- Multi-label classification (one sample can have multiple concepts)
- Cross-entropy loss over concept categories
- Accuracy target: 70%+ on test set

**Training script:**
```python
# Pseudocode
for epoch in epochs:
    for batch in dataloader:
        latent = jepa.encode_text(batch["text"])
        concept_logits = uncertainty_detector.concept_classifier(latent)
        loss = multi_label_cross_entropy(concept_logits, batch["concept_labels"])
        loss.backward()
```

### 2. Error Pattern Tracking

**Data needed:**
- Examples with ground-truth answers
- Error annotations (is_error=True/False)
- Concept labels

**Process:**
1. Run inference on evaluation set
2. Compare predictions to ground truth
3. Record errors with detected concepts
4. Analyze patterns to find problematic concepts

**Metrics:**
- Per-concept error rate
- Concept co-occurrence in errors
- Error rate trend over time

### 3. Uncertainty Calibration

**Goal:** Calibrate uncertainty scores to match actual error rates

**Method:**
1. Collect predictions with uncertainty scores
2. Bin by uncertainty level (0-10%, 10-20%, ..., 90-100%)
3. Measure actual error rate in each bin
4. Calibration is good if: predicted uncertainty ≈ actual error rate

**Example:**
- Bin: 60-70% uncertainty
- Actual error rate: 65%
- Calibration: Good! (within 10%)

## UI Integration (Future)

### Question Highlighting in Web UI

```html
<div class="response-container">
  <div class="llm-response">
    {{ response_text }}
  </div>

  <div class="uncertainty-panel" v-if="uncertainty.is_uncertain">
    <h3>⚠️ Model Uncertainty Detected</h3>

    <div class="uncertainty-meter">
      <div class="meter-fill" :style="{width: uncertainty.overall_uncertainty * 100 + '%'}"></div>
      <span>{{ (uncertainty.overall_uncertainty * 100).toFixed(1) }}% uncertain</span>
    </div>

    <div class="user-questions">
      <h4>🙋 The model needs your help:</h4>
      <div v-for="question in uncertainty.user_questions" :key="question.type">
        <div class="question-card" :class="question.priority">
          <div class="priority-badge">{{ question.priority }}</div>
          <p>{{ question.question }}</p>
          <small>{{ question.context }}</small>

          <!-- User feedback buttons -->
          <div class="feedback-buttons">
            <button @click="provideCorrection">Provide Correction</button>
            <button @click="addExample">Add Example</button>
            <button @click="rephrase">Rephrase Question</button>
          </div>
        </div>
      </div>
    </div>

    <div class="concepts">
      <h4>Detected Concepts:</h4>
      <div v-for="concept in uncertainty.detected_concepts" class="concept-chip">
        <span>{{ concept.concept.replace('_', ' ') }}</span>
        <span class="confidence">{{ (concept.confidence * 100).toFixed(1) }}%</span>
      </div>
    </div>
  </div>
</div>
```

### Error Pattern Dashboard

```html
<div class="error-patterns">
  <h2>Error Pattern Analysis</h2>

  <div class="stats-cards">
    <div class="card">
      <h3>Total Examples</h3>
      <p>{{ patterns.total_examples }}</p>
    </div>
    <div class="card">
      <h3>Error Rate</h3>
      <p>{{ (patterns.error_rate * 100).toFixed(1) }}%</p>
    </div>
  </div>

  <div class="problematic-concepts">
    <h3>⚠️ Concepts Causing Mistakes</h3>
    <table>
      <tr>
        <th>Concept</th>
        <th>Error Rate</th>
        <th>Examples</th>
        <th>Actions</th>
      </tr>
      <tr v-for="concept in patterns.problematic_concepts">
        <td>{{ concept.concept.replace('_', ' ') }}</td>
        <td>
          <div class="error-bar" :style="{width: concept.error_rate * 100 + '%'}"></div>
          {{ (concept.error_rate * 100).toFixed(1) }}%
        </td>
        <td>{{ concept.count }} ({{ concept.errors }} errors)</td>
        <td>
          <button @click="trainConcept(concept)">Add Training Data</button>
        </td>
      </tr>
    </table>
  </div>

  <div class="feedback-suggestions">
    <h3>💡 Suggested User Feedback</h3>
    <div v-for="suggestion in suggestions" class="suggestion-card">
      <div class="priority">{{ suggestion.priority }}</div>
      <p>{{ suggestion.suggestion }}</p>
      <button @click="provideFeedback(suggestion)">Provide Feedback</button>
    </div>
  </div>
</div>
```

## Benefits

### For Users

1. **Transparency:** Know when the model is uncertain
2. **Trust:** Model admits when it doesn't know
3. **Efficiency:** Model asks specific questions instead of generic "tell me more"
4. **Learning:** User sees which concepts need training

### For Model

1. **Targeted improvement:** Identify specific weak areas
2. **Active learning:** Request labels for uncertain examples
3. **Concept tracking:** Understand performance by concept type
4. **Error prevention:** Flag uncertain predictions before they cause harm

## Next Steps

1. **Label synthetic data with concepts** (100+ samples)
2. **Train concept classifier** (target: 70%+ accuracy)
3. **Collect error patterns** on evaluation set
4. **Calibrate uncertainty scores** (match predicted to actual error rate)
5. **Implement web UI** for question highlighting
6. **Add user feedback loop** (corrections → retraining)

## Example Output

```
============================================================
UNCERTAINTY ANALYSIS
============================================================

Overall Uncertainty: 67.42% ⚠️  UNCERTAIN
[██████████████████████████░░░░░░░░░░░░░░]

Uncertainty Breakdown:
  Consistency:       21.11%
  Component Variance: 100.00%
  Concept Confidence: 96.59%

Detected Concepts (Top 3):
  1. math statistics                  3.4% [░░░░░░░░░░░░░░░░░░░░]
  2. language pragmatics              3.4% [░░░░░░░░░░░░░░░░░░░░]
  3. arithmetic addition              3.4% [░░░░░░░░░░░░░░░░░░░░]

⚠️  Low Confidence Concepts:
  - math statistics (confidence: 3.4%)
  - language pragmatics (confidence: 3.4%)
  - arithmetic addition (confidence: 3.4%)

============================================================
🙋 QUESTIONS FOR USER (Model needs help!)
============================================================

🔴 Question 1 [Consistency Check]
Priority: HIGH

I'm not confident about my reasoning on this. Can you verify if my answer makes sense?

Context: Consistency score: 0.489
------------------------------------------------------------

🟡 Question 2 [Component Disagreement]
Priority: MEDIUM

My fast intuition and deep knowledge disagree on this. Which approach seems more appropriate?

Context: Disagreement level: 1.000
------------------------------------------------------------

🟡 Question 3 [Concept Uncertainty]
Priority: MEDIUM

I'm uncertain about these concepts: math statistics, language pragmatics, arithmetic addition. Can you provide more context or examples?

Context: Low confidence on: 3 concepts
------------------------------------------------------------
```

---

**Files:**
- `core/uncertainty.py` (UncertaintyDetector + ErrorPatternTracker)
- `scripts/test_uncertainty.py` (test script with visualization)
- `core/llm_integration.py` (updated with uncertainty integration)

**Total new code:** ~600 lines
