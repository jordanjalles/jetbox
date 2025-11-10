# L4-L7 Context Inspection Evaluation Report

**Date**: 2025-11-08 06:21:36

## Summary

- **Total runs**: 5
- **Successful**: 0 (0.0%)
- **Failed**: 5 (100.0%)

## Detailed Results

| Task | Run | Success | Duration | Files | Validation | Inspection |
|------|-----|---------|----------|-------|------------|------------|
| L5 - blog_system | 1 | ✗ | 1.5s | ✗ | ✗ | [View](evaluation_results/context_analysis_20251108_062130/failed_runs/L5_blog_system_run1_inspection) |
| L5 - todo_app | 1 | ✗ | 0.6s | ✗ | ✗ | [View](evaluation_results/context_analysis_20251108_062130/failed_runs/L5_todo_app_run1_inspection) |
| L5 - inventory_system | 1 | ✗ | 0.8s | ✗ | ✗ | [View](evaluation_results/context_analysis_20251108_062130/failed_runs/L5_inventory_system_run1_inspection) |
| L5 - url_shortener | 1 | ✗ | 0.8s | ✗ | ✗ | [View](evaluation_results/context_analysis_20251108_062130/failed_runs/L5_url_shortener_run1_inspection) |
| L5 - email_validator_service | 1 | ✗ | 0.9s | ✗ | ✗ | [View](evaluation_results/context_analysis_20251108_062130/failed_runs/L5_email_validator_service_run1_inspection) |

## Context Inspection Directories

- **Successful runs**: `evaluation_results/context_analysis_20251108_062130/successful_runs`
- **Failed runs**: `evaluation_results/context_analysis_20251108_062130/failed_runs`

## Next Steps

1. Analyze failed run context inspections
2. Look for patterns in context growth
3. Check for repeated errors in agent behavior
4. Generate context inspection reports with:
   ```bash
   python tools/analyze_context.py <inspection_dir>
   ```
