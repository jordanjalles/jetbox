# L4-L7 Context Inspection Evaluation Report

**Date**: 2025-11-07 23:06:00

## Summary

- **Total runs**: 40
- **Successful**: 0 (0.0%)
- **Failed**: 40 (100.0%)

## Detailed Results

| Task | Run | Success | Duration | Files | Validation | Inspection |
|------|-----|---------|----------|-------|------------|------------|
| L4 - rest_api_mock | 1 | ✗ | 4.9s | ✗ | ✗ | N/A |
| L4 - rest_api_mock | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - sqlite_manager | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - sqlite_manager | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - async_downloader | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - async_downloader | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - test_framework_basic | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - test_framework_basic | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - command_parser | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - command_parser | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - config_loader | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L4 - config_loader | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - blog_system | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - blog_system | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - todo_app | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - todo_app | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - inventory_system | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - inventory_system | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - url_shortener | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - url_shortener | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - email_validator_service | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L5 - email_validator_service | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - observer_pattern | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - observer_pattern | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - factory_pattern | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - factory_pattern | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - dependency_injection | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - dependency_injection | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - plugin_system | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - plugin_system | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - event_bus | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L6 - event_bus | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - rate_limiter | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - rate_limiter | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - connection_pool | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - connection_pool | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - circuit_breaker | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - circuit_breaker | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - distributed_cache | 1 | ✗ | 0.1s | ✗ | ✗ | N/A |
| L7 - distributed_cache | 2 | ✗ | 0.1s | ✗ | ✗ | N/A |

## Context Inspection Directories

- **Successful runs**: `evaluation_results/context_analysis_20251107_230551/successful_runs`
- **Failed runs**: `evaluation_results/context_analysis_20251107_230551/failed_runs`

## Next Steps

1. Analyze failed run context inspections
2. Look for patterns in context growth
3. Check for repeated errors in agent behavior
4. Generate context inspection reports with:
   ```bash
   python tools/analyze_context.py <inspection_dir>
   ```
