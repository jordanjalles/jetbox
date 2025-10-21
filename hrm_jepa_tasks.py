#!/usr/bin/env python3
"""HRM-JEPA development tasks for overnight autonomous work."""

from pathlib import Path

HRM_JEPA_TASKS = [
    {
        "id": "HRM-1",
        "category": "core",
        "name": "Uncertainty Quantification Enhancement",
        "goal": "Improve uncertainty detection in HRM-JEPA pipeline",
        "task": """Enhance the uncertainty quantification system in hrm-jepa/:
1. Review scripts/test_uncertainty.py and understand current implementation
2. Add variance tracking to JEPA encoder predictions (modify src/jepa/core.py if needed)
3. Create uncertainty_metrics.py with calibration metrics (ECE, Brier score)
4. Add tests to tests/unit/test_uncertainty.py
5. Run experiments and save results to results/uncertainty_analysis.json
6. Document findings in results/uncertainty_report.md""",
        "working_dir": "hrm-jepa",
        "expected_outputs": [
            "src/uncertainty_metrics.py",
            "tests/unit/test_uncertainty_metrics.py",
            "results/uncertainty_analysis.json",
            "results/uncertainty_report.md"
        ],
        "timeout": 1800,  # 30 min
    },
    {
        "id": "HRM-2",
        "category": "data",
        "name": "Training Data Quality Enhancement",
        "goal": "Improve synthetic data generation for JEPA training",
        "task": """Enhance training data generation:
1. Review current data generation in src/data/synthetic.py
2. Analyze data distribution (create analysis script)
3. Add data augmentation strategies (paraphrasing, entity swapping, etc.)
4. Generate 10,000 new diverse training samples
5. Validate sample quality with automated metrics
6. Save enhanced dataset to data/synthetic_v2.json
7. Document improvements in data/generation_report.md""",
        "working_dir": "hrm-jepa",
        "expected_outputs": [
            "src/data/augmentation.py",
            "data/synthetic_v2.json",
            "data/quality_metrics.json",
            "data/generation_report.md"
        ],
        "timeout": 1800,
    },
    {
        "id": "HRM-3",
        "category": "visualization",
        "name": "Latent Space Visualization",
        "goal": "Create tools to visualize JEPA latent representations",
        "task": """Build latent space visualization tools:
1. Create scripts/visualize_latents.py using matplotlib/seaborn
2. Implement PCA and t-SNE projection of JEPA embeddings
3. Generate visualizations for 100 sample inputs
4. Add clustering analysis (k-means on latent space)
5. Create interactive plots if possible
6. Save plots to results/latent_viz/
7. Document latent space structure in results/latent_analysis.md""",
        "working_dir": "hrm-jepa",
        "expected_outputs": [
            "scripts/visualize_latents.py",
            "results/latent_viz/pca_projection.png",
            "results/latent_viz/tsne_projection.png",
            "results/latent_viz/clusters.png",
            "results/latent_analysis.md"
        ],
        "timeout": 1200,  # 20 min
    },
    {
        "id": "HRM-4",
        "category": "testing",
        "name": "End-to-End Pipeline Test",
        "goal": "Create comprehensive integration test for HRM-JEPA",
        "task": """Build end-to-end integration tests:
1. Create tests/integration/test_pipeline.py
2. Design 10+ test scenarios covering edge cases
3. Test with gpt-oss:20b integration (if Ollama available)
4. Measure latency, throughput, and quality metrics
5. Add benchmarking suite
6. Document test results in tests/integration/results.md
7. Create CI-friendly test runner""",
        "working_dir": "hrm-jepa",
        "expected_outputs": [
            "tests/integration/test_pipeline.py",
            "tests/integration/test_scenarios.py",
            "tests/integration/benchmarks.py",
            "tests/integration/results.md"
        ],
        "timeout": 1800,
    },
    {
        "id": "HRM-5",
        "category": "performance",
        "name": "Performance Optimization",
        "goal": "Profile and optimize HRM-JEPA inference",
        "task": """Optimize HRM-JEPA pipeline performance:
1. Create scripts/profile_pipeline.py using cProfile and torch.profiler
2. Profile current pipeline and identify bottlenecks
3. Optimize critical paths (batching, caching, etc.)
4. Add batch processing support to JEPA encoder
5. Implement model quantization if beneficial
6. Benchmark before/after performance
7. Document optimizations in results/performance_report.md""",
        "working_dir": "hrm-jepa",
        "expected_outputs": [
            "scripts/profile_pipeline.py",
            "src/optimizations.py",
            "results/profile_before.txt",
            "results/profile_after.txt",
            "results/performance_report.md"
        ],
        "timeout": 1800,
    },
    {
        "id": "HRM-6",
        "category": "reliability",
        "name": "State Persistence Enhancement",
        "goal": "Improve crash resilience of HRM state",
        "task": """Enhance state management and crash recovery:
1. Review current checkpoint system in src/hrm/state.py
2. Add incremental state saving (every N steps)
3. Implement state validation and corruption detection
4. Add recovery tests simulating crashes
5. Create state migration utilities
6. Document state format and recovery procedure
7. Add tests to tests/unit/test_state_recovery.py""",
        "working_dir": "hrm-jepa",
        "expected_outputs": [
            "src/hrm/state_manager.py",
            "src/hrm/state_validator.py",
            "tests/unit/test_state_recovery.py",
            "docs/state_format.md"
        ],
        "timeout": 1200,
    },
]


def get_hrm_tasks_by_category(category: str) -> list[dict]:
    """Get HRM-JEPA tasks by category."""
    return [t for t in HRM_JEPA_TASKS if t["category"] == category]


def get_all_hrm_tasks() -> list[dict]:
    """Get all HRM-JEPA tasks."""
    return HRM_JEPA_TASKS
