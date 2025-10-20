# System Overview: HRM+JEPA

## Vision

Build a local-first multimodal AI system that combines:
- **Hierarchical Reasoning Model (HRM)** for transparent, layered reasoning
- **Joint Embedding Predictive Architecture (JEPA)** as the predictive latent core

The system operates entirely locally (Windows + RTX 3090), trains on synthetic data only, and requires human approval for deep model updates.

## Architecture Layers

### 1. JEPA Latent Core (Foundation)

**Purpose:** Learn unified multimodal representations

**Components:**
- **Vision Encoder:** ViT-lite (Vision Transformer) - encodes images to latents
- **Text Encoder:** Transformer - encodes text to latents
- **Joint Latent Space:** Shared representation z where text and vision align
- **Predictive Objective:** Predict masked/future latents (no pixel/token reconstruction)

**Key Property:** Self-supervised learning from synthetic data

**Output:** Dense latent vectors z that capture semantic meaning across modalities

### 2. HRM Reasoning Head (Control)

**Purpose:** Hierarchical reasoning and decision-making over latents

**Components:**

**a) Working Memory (Fast Layer)**
- Lightweight adapters over JEPA latents (LoRA/IA3-style)
- Task-specific context (what are we doing right now?)
- Updates rapidly during task execution
- High plasticity, short-term retention

**b) Abstract Core (Slow Layer)**
- Deep reasoning weights
- Long-term knowledge and patterns
- Updates only with human approval
- Low plasticity, long-term retention
- Gated by reflection loop

**c) Reflection Loop**
- Monitors consistency between working memory and abstract core
- Detects contradictions or uncertainty
- Generates update proposals with rationale
- Stores thought traces for human review

**Key Property:** Explicit separation of fast/slow thinking (inspired by Kahneman's System 1/2)

### 3. Human-Approval Gate (Safety)

**Purpose:** Ensure deep changes are transparent and reversible

**Workflow:**
1. Reflection loop detects need for abstract core update
2. System generates proposal JSON:
   - Rationale (why is this change needed?)
   - Evidence (what data supports it?)
   - Expected effects (what will change?)
   - Revert plan (how to undo it?)
3. User reviews in UI (diffs, consistency scores, thought traces)
4. User approves or denies
5. If approved, checkpoint old state, apply update
6. Monitor for regressions

**Key Property:** No "black box" updates - every deep change is explained

## Data Flow

```
┌──────────────────────────────────────────────────────────┐
│  Input: Text + Image                                     │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  JEPA Encoders                                         │
│  ┌──────────┐              ┌──────────┐              │
│  │   ViT    │              │   Text   │              │
│  │  Vision  │              │Transformer│             │
│  └────┬─────┘              └────┬─────┘              │
│       │                         │                     │
│       └───────────┬─────────────┘                     │
│                   │                                   │
│              Joint Latent z                           │
│           (semantic vector)                           │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  HRM Reasoning                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Working Memory (fast adapters)                 │  │
│  │  - Task context                                 │  │
│  │  - Recent interactions                          │  │
│  │  - Temporary hypotheses                         │  │
│  └─────────────────┬───────────────────────────────┘  │
│                    │                                   │
│  ┌─────────────────▼───────────────────────────────┐  │
│  │  Reflection Loop                                │  │
│  │  - Consistency check                            │  │
│  │  - Uncertainty estimation                       │  │
│  │  - Update proposal generation                   │  │
│  └─────────────────┬───────────────────────────────┘  │
│                    │                                   │
│                    │ (if proposal needed)              │
│                    ▼                                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Abstract Core (slow weights)                   │  │
│  │  - Long-term knowledge                          │  │
│  │  - Stable reasoning patterns                    │  │
│  │  - [Gated by human approval]                    │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Output: Reasoning trace + Action                      │
└────────────────────────────────────────────────────────┘
```

## Training Strategy

### Phase 1: JEPA Pre-training (Unsupervised)

**Goal:** Learn good multimodal representations

**Data:** Synthetic text-image pairs
- Text: Generated via self-play, instruction trees
- Images: ComfyUI/Stable Diffusion with synthetic prompts
- Alignment: Programmatically generated (caption ↔ image)

**Objective:** Predict masked/future latents
- Mask random patches of image latents
- Mask random tokens of text latents
- Predict masked content from context
- No pixel/token reconstruction (only latent prediction)

**Success Metric:** Low prediction error on held-out synthetic data

### Phase 2: HRM Working Memory (Supervised)

**Goal:** Learn task-specific reasoning

**Data:** Synthetic task traces
- Instruction → reasoning steps → answer
- Generated via self-play or templates

**Objective:** Fine-tune working memory adapters
- Freeze JEPA encoders
- Train lightweight heads over latents
- Predict next reasoning step given context

**Success Metric:** Accuracy on synthetic reasoning tasks

### Phase 3: HRM Abstract Core (Human-in-Loop)

**Goal:** Build stable long-term knowledge

**Data:** User interactions + reflection traces

**Objective:** Selective updates to deep weights
- Working memory proposes updates
- Reflection loop validates consistency
- Human reviews and approves
- Abstract core updated only on approval

**Success Metric:** User approval rate, consistency scores

## Crash Recovery

**Philosophy:** Expect crashes; design for fast resume

**State Persistence:**
- JEPA checkpoints → `checkpoints/jepa/epoch_NNN.pth`
- HRM checkpoints → `checkpoints/hrm/step_MMMMM.pth`
- Reflection traces → `logs/reflections/trace_TIMESTAMP.jsonl`
- Thought traces → `logs/thoughts/YYYYMMDD_HHMMSS.json`

**Resume Logic:**
1. On startup, check for latest checkpoint
2. Load JEPA state, HRM state, reflection history
3. Continue from last saved step
4. Idempotent operations (can re-run safely)

## Safety Properties

### Local-Only Processing

- ✅ All computation on RTX 3090 (no cloud)
- ✅ No network calls in model code
- ✅ Data stays on disk
- ✅ Works offline

### Synthetic-Only Data

- ✅ No web scraping
- ✅ No external datasets
- ✅ Full provenance tracking (seeds, params, generation method)
- ✅ Deterministic reproduction

### Human-Approval Gates

- ✅ No "black box" updates
- ✅ Every abstract core change has explanation
- ✅ One-click rollback
- ✅ Thought traces inspectable

### Deterministic Behavior

- ✅ Fixed seeds for reproducibility
- ✅ `torch.backends.cudnn.deterministic = True`
- ✅ Manifest versioning for data

## Performance Targets (RTX 3090)

| Component | Target | Notes |
|-----------|--------|-------|
| JEPA training step | ≤ 100ms | Batch=32, resolution=224 |
| HRM inference | ≤ 50ms | Single example, working memory only |
| Peak VRAM | ≤ 22GB | Leave 2GB headroom |
| Checkpoint save | ≤ 5s | Full model state |
| Reflection step | ≤ 200ms | Consistency check + scoring |

## Milestones Recap

- **M0:** Repo skeleton + tooling ✅ (current)
- **M1:** Synthetic data fabric (text + image generators)
- **M2:** JEPA core (encoders + joint latent + training loop)
- **M3:** HRM reasoning (working memory + abstract core + reflection)
- **M4:** Web UI (FastAPI + Gradio with approval workflow)
- **M5:** Eval harness (synthetic tasks, consistency checks)
- **M6+:** Optimization (VRAM tuning, throughput, curriculum learning)

## Next Steps

1. Execute TICKET-001: Synthetic text generator
2. Execute TICKET-002: Image synth harness
3. Execute TICKET-003: JEPA encoder stubs
4. Validate each milestone before moving forward

**Core principle:** Incremental, verifiable progress. Each ticket delivers a working, tested artifact.
