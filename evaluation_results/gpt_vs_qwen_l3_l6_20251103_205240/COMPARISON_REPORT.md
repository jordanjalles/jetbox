# GPT-OSS vs Qwen3:8b Comparison Report

**Date**: 2025-11-03T22:40:13.844109
**Total Time**: 6453.2s (107.6m)
**Models**: gpt-oss:20b (baseline) vs qwen3:8b (challenger)
**Levels**: L3-L6 (5 tasks per level)
**Total Tests**: 40

## Overall Performance

| Model | Success | Failed | Timeout | Unknown | Error | Success Rate | Avg Time (successful) |
|-------|---------|--------|---------|---------|-------|--------------|----------------------|
| gpt-oss:20b | 5/20 | 0 | 6 | 9 | 0 | 25.0% | 140.6s |
| qwen3:8b | 10/20 | 0 | 7 | 3 | 0 | 50.0% | 77.7s |

## Winner

**🏆 qwen3:8b wins** with 50.0% success rate (+25.0% vs baseline)

**Speed**: qwen3:8b is 1.81x faster

## Performance by Level

| Level | gpt-oss:20b | qwen3:8b | Winner |
|-------|-------------|----------|--------|
| L3 | 0/5 (0%) | 1/5 (20%) | qwen3:8b 🏆 |
| L4 | 1/5 (20%) | 2/5 (40%) | qwen3:8b 🏆 |
| L5 | 3/5 (60%) | 3/5 (60%) | Tie |
| L6 | 1/5 (20%) | 4/5 (80%) | qwen3:8b 🏆 |

## Detailed Results by Level

### L3 Results

| Run | Goal | gpt-oss:20b | qwen3:8b |
|-----|------|-------------|----------|
| 1 | Create a Python package 'mathx' with add, subtract, multiply... | ❓ 19.1s | ✅ 21.7s |
| 2 | Create a 'string_utils' package with functions: reverse(s), ... | ❓ 27.8s | ⏱️ 180.0s |
| 3 | Build a 'data_structures' package with Stack and Queue class... | ❓ 42.7s | ❓ 172.9s |
| 4 | Create a 'validators' package with email, phone, url, and pa... | ❓ 39.2s | ⏱️ 180.0s |
| 5 | Build a 'converters' package with temperature, distance, wei... | ❓ 34.4s | ❓ 119.6s |

### L4 Results

| Run | Goal | gpt-oss:20b | qwen3:8b |
|-----|------|-------------|----------|
| 1 | Create a 'requests_wrapper' package that wraps HTTP requests... | ✅ 28.1s | ✅ 17.1s |
| 2 | Build a 'json_validator' package that validates JSON schemas... | ❓ 42.6s | ❓ 117.3s |
| 3 | Create a 'file_processor' package that reads/writes CSV, JSO... | ❓ 46.8s | ⏱️ 240.0s |
| 4 | Build a 'cache_manager' package with TTL-based in-memory cac... | ❓ 43.3s | ⏱️ 240.0s |
| 5 | Create a 'logger_wrapper' package with multiple output forma... | ❓ 46.8s | ✅ 54.7s |

### L5 Results

| Run | Goal | gpt-oss:20b | qwen3:8b |
|-----|------|-------------|----------|
| 1 | Create a Flask REST API with CRUD endpoints for a User model... | ⏱️ 300.0s | ⏱️ 300.0s |
| 2 | Build a Flask API for a Book library with Book model (id, ti... | ✅ 221.0s | ✅ 104.9s |
| 3 | Create a Flask REST API for a Product catalog (id, name, pri... | ⏱️ 300.0s | ✅ 158.8s |
| 4 | Build a Flask API for managing Tasks (id, title, description... | ✅ 143.8s | ✅ 89.7s |
| 5 | Create a Flask REST API for Student records (id, name, grade... | ✅ 144.9s | ⏱️ 300.0s |

### L6 Results

| Run | Goal | gpt-oss:20b | qwen3:8b |
|-----|------|-------------|----------|
| 1 | Build a Flask blog API with User and Post models. Include us... | ⏱️ 420.0s | ✅ 84.5s |
| 2 | Create a Flask e-commerce API with Product and Order models.... | ✅ 165.0s | ✅ 73.9s |
| 3 | Build a Flask todo API with User and Task models. Include au... | ⏱️ 420.0s | ✅ 82.0s |
| 4 | Create a Flask notes API with User and Note models. Include ... | ⏱️ 420.0s | ⏱️ 420.0s |
| 5 | Build a Flask inventory API with User and Item models. Inclu... | ⏱️ 420.0s | ✅ 89.6s |

## Key Findings

- **Total Successes**: gpt-oss:20b (5), qwen3:8b (10)
- gpt-oss:20b had 6 timeouts
- qwen3:8b had 7 timeouts
- gpt-oss:20b had 9 UNKNOWN statuses (completion detection issue)
- qwen3:8b had 3 UNKNOWN statuses

