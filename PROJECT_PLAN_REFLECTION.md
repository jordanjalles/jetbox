# Reflection on Project Plan: HRM + JEPA Local Multimodal AI System

## Executive Summary

The PROJECT_PLAN.md outlines an ambitious vision for building a local multimodal AI system combining Hierarchical Reasoning Model (HRM) with Joint Embedding Predictive Architecture (JEPA). This reflection analyzes the plan's relationship to the current Jetbox agent implementation and provides recommendations for moving forward.

## Current State: Jetbox Agent (Where We Are)

### What We Have Built

**Core Components:**
- `agent.py` - Basic agent with flat context and backward-chaining planning
- `agent_enhanced.py` - Enhanced agent with hierarchical context manager
- `agent_quality.py` - Optimized agent with LLM warm-up (1.7x speedup)
- `context_manager.py` - Hierarchical context (Goal→Task→Subtask→Action)
- `llm_warmup.py` - LLM warm-up and keep-alive (9.2s savings on first call)

**Key Achievements:**
- ✅ Crash-resilient local-first design with state persistence
- ✅ Hierarchical context management with loop detection
- ✅ Performance optimizations (probe caching, parallel execution, LLM warm-up)
- ✅ Production-quality code generation with gpt-oss:20b
- ✅ Comprehensive profiling and benchmarking tools

**Design Philosophy Alignment:**
Our current Jetbox agent already embodies several principles from the new project plan:
- **Local-first:** All processing runs locally with Ollama
- **Crash-resilient:** State persists to `.agent_context/state.json`
- **Transparent reasoning:** Hierarchical task trees show explicit reasoning steps
- **Incremental verification:** Each subtask completion is verified via probes

### What We Don't Have Yet

**Missing from New Plan:**
- ❌ Vision/multimodal capabilities (text-only currently)
- ❌ JEPA latent space architecture
- ❌ Explicit "working memory" vs "abstract core" separation
- ❌ Human-approval gate for deep model updates
- ❌ Synthetic data generation pipeline
- ❌ FastAPI + Gradio UI
- ❌ Formal eval harness with synthetic tasks

## Project Plan Analysis: HRM + JEPA Vision (Where We're Going)

### Strengths of the Plan

**1. Clear Separation of Concerns**
- Master Orchestrator (Claude) vs Local Agent (RTX 3090 execution)
- Working memory (fast adapters) vs Abstract core (slow, gated updates)
- This mirrors our current hierarchical context manager design

**2. Incremental Milestones**
- M0→M6+ progression ensures steady, testable progress
- Each milestone has clear deliverables and acceptance criteria
- Matches our agent's "short timebox" philosophy

**3. Safety-First Design**
- Local-only processing (no external data pulls)
- Synthetic-only data policy
- Human-approval gate for abstract core updates
- Aligns with our current "safe commands only" approach (SAFE_BIN whitelist)

**4. Rigorous Testing Culture**
- pytest + ruff + mypy + pre-commit
- ≥80% coverage target
- Deterministic seeds for reproducibility
- We already have this foundation in place

### Potential Challenges

**1. Scope Expansion**
The new plan is significantly larger than current Jetbox:
- Current: ~2,000 LOC Python agent
- Proposed: Full multimodal training pipeline + inference + UI

**Risk:** Scope creep could derail focused agent development
**Mitigation:** Keep Jetbox as the "master orchestrator" and delegate larger ML tasks to separate repos

**2. Hardware Constraints**
- RTX 3090 (24GB VRAM) is powerful but limited for:
  - Large vision transformers (ViT)
  - Training joint latent spaces
  - Simultaneous model updates + inference

**Risk:** VRAM pressure could limit model size and batch size
**Mitigation:** Use gradient accumulation, bf16, smaller ViTs, and staged training

**3. Synthetic Data Quality**
Plan assumes synthetic data can match real-world performance:
- Text: Self-play can work well (proven in many LLM setups)
- Vision: Synthetic images from SD/ComfyUI may have domain gaps

**Risk:** Model performance ceiling limited by synthetic data quality
**Mitigation:** Start with simple tasks where synthetic works (geometric reasoning, text-image alignment), expand gradually

**4. Complexity of JEPA**
JEPA (Joint Embedding Predictive Architecture) is relatively new:
- Requires careful tuning of masking strategies
- Latent space quality depends on encoder architecture
- Prediction target design is non-trivial

**Risk:** Could take multiple iterations to get working well
**Mitigation:** Start with well-studied components (ViT from timm, proven text encoders), focus on engineering first

## Recommended Path Forward: Hybrid Approach

### Phase 1: Enhance Jetbox as Master Orchestrator (2-4 weeks)

**Goal:** Make Jetbox the "Claude" role from the project plan

**Enhancements:**
1. **Ticket System:**
   - Add `tickets/` directory for TICKET-NNN.yaml files
   - Each ticket has: goal, constraints, acceptance tests, DoD
   - Context manager tracks ticket execution state

2. **Delegation Protocol:**
   - New tool: `delegate_to_local(ticket_id, spec)`
   - Monitors execution via log files
   - Collects artifacts (diffs, test results, metrics)
   - Provides review feedback

3. **Safety Gates:**
   - Whitelist expansion detection (warns before adding new tools)
   - Network call detector (fails if any subprocess tries external access)
   - Deep change proposals (require user approval)

4. **Improved Reflection:**
   - Current: Loop detection on actions
   - Enhanced: Consistency scoring across task hierarchy
   - Proposal format for agent improvements (mirrors HRM abstract core updates)

**Deliverable:** `agent_orchestrator.py` - Enhanced Jetbox that can manage ticket-based workflows

### Phase 2: Bootstrap M0 (Repo Skeleton) (1-2 weeks)

**Goal:** Create the new HRM+JEPA project structure using Jetbox orchestrator

**Approach:**
1. Jetbox creates TICKET-000 based on PROJECT_PLAN.md M0 spec
2. Jetbox generates files for new repo:
   - `pyproject.toml` with dependencies
   - `environment.yml` for Conda
   - Pre-commit config
   - Basic `tests/unit/test_smoke.py`
3. Jetbox validates the new repo structure (pytest, ruff, mypy)

**Deliverable:** New `hrm-jepa/` project directory with passing CI

### Phase 3: Incremental M1-M6 Execution (3-6 months)

**Goal:** Execute milestones M1→M6 with Jetbox orchestrating

**Process per milestone:**
1. Jetbox creates tickets from PROJECT_PLAN.md milestone spec
2. For each ticket:
   - Jetbox generates implementation spec
   - User reviews spec
   - Jetbox executes (or delegates to local agent if implemented)
   - Jetbox validates outputs (tests, metrics, DoD)
3. Jetbox compiles milestone report (deliverables checklist)

**Key Decision Points:**
- **M2 (JEPA Core):** May need to switch from Jetbox orchestration to manual coding for complex ML
- **M3 (HRM):** This is where we connect back to Jetbox's own hierarchical context manager
- **M4 (UI):** Jetbox can generate FastAPI/Gradio scaffolding
- **M5 (Eval):** Jetbox excels at test harness generation

## Synergies Between Jetbox and HRM+JEPA

### 1. Hierarchical Context Manager → HRM Working Memory

Our current `context_manager.py` is a prototype of HRM working memory:
- **Current:** Goal→Task→Subtask→Action hierarchy for coding tasks
- **HRM:** High-level reasoning → Task decomposition → Action selection

**Bridge:** Extract `context_manager.py` into a reusable module that can work over:
- Code tasks (current)
- JEPA latent space reasoning (future)

### 2. Loop Detection → Reflection Loop

Our current loop detection (repeated actions, alternating patterns) is a basic reflection mechanism:
- **Current:** Block looping actions, advance to next subtask
- **HRM:** Detect inconsistencies in reasoning, propose updates to abstract core

**Bridge:** Generalize loop detection to "consistency scoring" that works on:
- Tool call sequences (current)
- Reasoning traces over latents (future)

### 3. Probe-Verify-Act → JEPA Prediction

Our current agent workflow:
1. Probe state (files, tests, linting)
2. Verify against expected state
3. Act to close the gap

This mirrors JEPA's predict-then-check paradigm:
1. Predict future latent state
2. Encode actual observation
3. Update to close prediction error

**Bridge:** Formalize "probe" as a general prediction-verification interface

### 4. LLM Warm-up → Model Keep-Alive

Our `llm_warmup.py` keeps gpt-oss:20b warm to reduce latency:
- Pre-warm on startup (155ms vs 9,376ms cold)
- Keep-alive thread maintains model in memory

**Bridge:** Apply same strategy to JEPA/HRM models:
- Keep encoders warm for fast inference
- Lazy-load abstract core only when needed

## Risks and Mitigations

### Risk 1: Jetbox Becomes Too Complex

**Problem:** Adding orchestration + delegation + tickets makes Jetbox harder to understand

**Mitigation:**
- Keep `agent_quality.py` as-is for simple coding tasks
- Create separate `agent_orchestrator.py` for ticket-based workflows
- Clear documentation on when to use each

### Risk 2: HRM+JEPA Diverges from Jetbox Philosophy

**Problem:** ML training pipelines have different patterns than coding agents

**Mitigation:**
- Apply same principles: local-first, crash-resilient, verify-first
- Use Jetbox context manager for training job orchestration
- Maintain "short timebox" discipline even for long training runs (checkpoint frequently)

### Risk 3: Synthetic Data Limits Performance

**Problem:** Models trained purely on synthetic data may not generalize

**Mitigation:**
- Start with tasks where synthetic works well (geometric reasoning, code generation)
- Use Jetbox itself to generate diverse synthetic examples
- Plan for eventual human feedback loop (already designed in reflection policy)

### Risk 4: VRAM Pressure on RTX 3090

**Problem:** 24GB may not be enough for training + inference simultaneously

**Mitigation:**
- Use gradient accumulation (effective batch size > memory batch size)
- Train encoders separately, then freeze for HRM training
- Offload abstract core to CPU when not in use (keep working memory on GPU)

## Recommendations

### Immediate Next Steps (This Week)

1. **Complete Jetbox Documentation Cleanup** ✅
   - Remove unreliable model references
   - Focus on gpt-oss:20b quality
   - Commit and push

2. **Create Ticket System Prototype**
   - Design TICKET-NNN.yaml schema
   - Add `tickets/` directory with examples
   - Implement ticket parser in context manager

3. **Draft TICKET-000 for HRM+JEPA M0**
   - Use PROJECT_PLAN.md M0 spec
   - Specify exact file structure and content
   - Include acceptance tests

### Short-Term Goals (Next 2-4 Weeks)

1. **Enhance Jetbox Orchestrator**
   - Implement ticket tracking in context manager
   - Add delegation interface (placeholder for now)
   - Create review workflow (compare outputs vs DoD)

2. **Bootstrap M0 with Jetbox**
   - Run Jetbox to generate HRM+JEPA repo skeleton
   - Validate all files and tests pass
   - Document the process for future milestones

3. **Reflect on M0 Experience**
   - What worked well with Jetbox orchestration?
   - What needed manual intervention?
   - How to improve for M1?

### Medium-Term Goals (Next 2-3 Months)

1. **Execute M1 (Synthetic Data Fabric)**
   - Text synthesizer (Jetbox can help generate this)
   - Image synth harness (may need manual setup of ComfyUI)
   - Data versioning and manifest system

2. **Execute M2 (JEPA Core - Minimal)**
   - This is the critical milestone
   - May need to switch from Jetbox orchestration to hands-on coding
   - Focus on getting one working example end-to-end

3. **Connect M3 (HRM) to Jetbox Context Manager**
   - Extract hierarchical reasoning logic
   - Apply to both code tasks and JEPA latents
   - Unified reflection loop

### Long-Term Vision (6+ Months)

1. **Self-Improving Agent**
   - Jetbox orchestrates its own improvements
   - Proposes updates to context manager
   - User reviews and approves deep changes

2. **Multimodal Jetbox**
   - Can reason over code + images + (eventually) audio
   - Uses HRM+JEPA for latent reasoning
   - Maintains same local-first, crash-resilient design

3. **Community Ecosystem**
   - Jetbox orchestrator as standalone tool
   - HRM+JEPA as pluggable backend
   - Clear interfaces for extension

## Conclusion

The PROJECT_PLAN.md for HRM+JEPA is ambitious but achievable. The key insight is that **Jetbox is already a prototype of the master orchestrator role** described in the plan. Rather than building HRM+JEPA separately, we should:

1. **Enhance Jetbox to be the orchestrator** it's meant to be (ticket system, delegation, review)
2. **Use enhanced Jetbox to bootstrap the HRM+JEPA project** (starting with M0)
3. **Extract common patterns** from both projects (hierarchical context, reflection, crash recovery)
4. **Eventually merge** HRM working memory into Jetbox's context manager for multimodal reasoning

This approach:
- ✅ Leverages existing Jetbox investment
- ✅ Provides concrete testing ground for orchestration ideas
- ✅ Maintains incremental, verifiable progress
- ✅ Keeps local-first, crash-resilient philosophy
- ✅ Creates reusable components (context manager, reflection loop, synthetic data)

The next immediate step is to create a ticket system prototype and draft TICKET-000 for M0. This will validate the orchestration approach before committing to larger milestones.

---

**Author:** Claude Code
**Date:** 2025-10-20
**Status:** Proposal for User Review
