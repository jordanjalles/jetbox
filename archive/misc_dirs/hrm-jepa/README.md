# HRM+JEPA: Local Text Reasoning System

A local-first text reasoning system combining **Hierarchical Reasoning Model (HRM)** for layered reasoning with **Joint Embedding Predictive Architecture (JEPA)** as the predictive latent core, integrated with **gpt-oss:20b** as an outer LLM layer.

## Overview

This project implements a transparent, reflective AI system with:
- **Text-only processing** (vision planned for future expansion)
- **LLM integration** (gpt-oss:20b via Ollama as outer reasoning layer)
- **Local processing only** (Windows 10/11 + RTX 3090, 24GB VRAM)
- **Synthetic data training** (no external data dependencies)
- **Human-approval gates** for deep model updates
- **Crash-resilient design** with state persistence
- **Compact storage** (limited disk space via quantization & compression)

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

Flow: Text → JEPA → HRM → LLM(context + latent) → Output
```

## Quick Start

### Prerequisites

- **Hardware:** RTX 3090 (24GB VRAM) or similar
- **OS:** Windows 10/11 with WSL2 or native Python
- **Software:**
  - Conda or Miniconda
  - CUDA 12.1 drivers
  - Ollama (for gpt-oss:20b)
  - Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jordanjalles/hrm-jepa.git
   cd hrm-jepa
   ```

2. **Install Ollama and pull gpt-oss:20b:**
   ```bash
   # Install Ollama from https://ollama.ai
   # Then pull the model:
   ollama pull gpt-oss:20b
   ```

3. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate hrm-jepa
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Verify installation:**
   ```bash
   pytest tests/unit/ -q
   ruff check .
   black --check .
   ```

## Project Structure

```
hrm-jepa/
├─ core/                    # Core model implementations
│  ├─ jepa_core.py         # JEPA architecture (text-only mode)
│  ├─ llm_integration.py   # gpt-oss:20b integration & comparison
│  ├─ encoders/            # Encoders
│  │  ├─ vision_vit.py     # Vision Transformer (unused in text-only)
│  │  └─ text_transformer.py
│  ├─ objectives/          # Training objectives
│  │  └─ jepa_objectives.py
│  └─ hrm/                 # HRM reasoning components
│     ├─ hrm_reasoner.py
│     ├─ working_memory.py
│     ├─ abstract_core.py
│     └─ reflection_loop.py
├─ data/                    # Data storage
│  ├─ text/                # Synthetic text data
│  └─ manifests/           # Data manifests with provenance
├─ scripts/                 # Training and generation scripts
│  ├─ compare_hrm_jepa_llm.py  # HRM-JEPA-LLM vs baseline comparison
│  ├─ generate_synthetic_text.py
│  ├─ train_jepa.py
│  ├─ train_hrm.py
│  └─ eval_suite.py
├─ ui/                      # Web interface
│  ├─ api.py               # FastAPI backend
│  ├─ webui.py             # Gradio frontend
│  └─ feedback.py          # Human feedback collection
├─ configs/                 # Configuration files
│  ├─ env_windows.yaml
│  ├─ jepa_config.yaml
│  ├─ hrm_config.yaml
│  ├─ reflection_rules.yaml
│  └─ data_config.yaml
├─ tests/                   # Test suite
│  ├─ unit/
│  └─ integration/
├─ docs/                    # Documentation
│  ├─ SYSTEM_OVERVIEW.md
│  ├─ ARCH_HRM_JEPA.md
│  └─ REFLECTION_POLICY.md
└─ tools/                   # Utilities
   ├─ compact_storage.py   # Compression & cleanup for limited disk
   ├─ profiling.py
   ├─ seed_utils.py
   └─ checkpointing.py
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest -q

# Unit tests only
pytest tests/unit/ -q

# With coverage
pytest --cov=core --cov=ui --cov-report=html
```

### Code Quality

```bash
# Lint
ruff check .

# Auto-fix issues
ruff check --fix .

# Format
black .

# Type check
mypy core/ --strict
```

### LLM Comparison

```bash
# Compare HRM-JEPA-LLM vs baseline gpt-oss:20b
python scripts/compare_hrm_jepa_llm.py --model gpt-oss:20b

# View results
cat comparison_results/comparison_*.json
```

This will run test cases through both pipelines:
- **HRM-JEPA-LLM:** Text → JEPA → HRM → LLM with enriched context
- **Baseline LLM:** Direct gpt-oss:20b with no reasoning layer

Results include response quality, consistency scores, and latency measurements.

### Training

```bash
# Generate synthetic data
python scripts/generate_synthetic_text.py --cfg configs/data_config.yaml

# Train JEPA
python scripts/train_jepa.py --cfg configs/jepa_config.yaml

# Train HRM
python scripts/train_hrm.py --cfg configs/hrm_config.yaml
```

### Storage Management

```bash
# Check storage usage
python -c "from tools.compact_storage import estimate_storage_usage, suggest_cleanup_actions; print(suggest_cleanup_actions('.'))"

# Quantize checkpoints (75% size reduction)
python -c "from tools.compact_storage import quantize_checkpoint; quantize_checkpoint('checkpoints/model.pth', 'checkpoints/model_int8.pth')"

# Compress old logs
python -c "from tools.compact_storage import compress_logs; compress_logs('logs/', keep_recent=10)"
```

### Running the UI

```bash
# Start FastAPI backend
python ui/api.py

# Start Gradio frontend (separate terminal)
python ui/webui.py
```

## Design Principles

### Local-First & Crash-Resilient

- All processing runs locally (no cloud dependencies)
- State persists to disk at every checkpoint
- Can resume from any interruption
- Idempotent operations for safe retries

### Synthetic-Only Data

- **Text:** Self-play conversations, instruction trees, chain-of-thought
- **Images:** ComfyUI/Stable Diffusion with synthetic prompts
- **Manifests:** Full provenance (seed, params, generation method)
- **No web scraping** or external datasets

### Transparent Reasoning

- Working memory: Fast adapters for task context
- Abstract core: Slow, deliberate updates (human-approved)
- Reflection loop: Stores thought traces and consistency scores
- All reasoning steps are inspectable

### Human-in-the-Loop Updates

- Abstract core changes require explicit approval
- Update proposals show: rationale, evidence, expected effects, revert plan
- UI displays diffs and provides one-click rollback

## Milestones

- [x] **M0:** Repo Skeleton & Tooling *(current)*
- [ ] **M1:** Synthetic Data Fabric
- [ ] **M2:** JEPA Core (minimal)
- [ ] **M3:** HRM Reasoning Head
- [ ] **M4:** Local Web UI & API
- [ ] **M5:** Evaluation & Safety
- [ ] **M6+:** Optimization & Extensions

## Performance Targets

- **Training step:** ≤ 100ms (RTX 3090, batch=32, resolution=224)
- **Peak VRAM:** ≤ 22GB (leave 2GB headroom)
- **Inference latency:** ≤ 50ms for single example
- **Test coverage:** ≥ 80% for core modules

## Safety & Constraints

- ✅ **Local-only:** No network calls in model code
- ✅ **Synthetic-only:** All data generated locally
- ✅ **Deterministic:** Fixed seeds for reproducibility
- ✅ **Gated updates:** Human approval for abstract core changes
- ✅ **Offline-first:** Works without internet connection

## Contributing

This is a personal research project. Contributions are not currently accepted, but feel free to fork and experiment.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built using the Jetbox agent orchestrator system, which implements the master orchestrator role for ticket-based development.

**Philosophy:** Transparent, reflective AI with human oversight. Fast-adapting working memory; slow, deliberate abstract reasoning with explicit approval gates.
