# Model Benchmark Summary: gpt-oss:20b vs qwen3:14b

**Date**: 2025-10-28
**Benchmark File**: simple_benchmark_20251028_071248.json
**Tasks**: L4-L6 coding tasks (Calculator, Refactor, Simple API)
**Iterations**: 5 per task per model
**Total Runs**: 30 (15 per model)

---

## Executive Summary

**Key Finding**: gpt-oss:20b is **2.31x faster** than qwen3:14b with **identical pass rates** (86.7%)

Both models demonstrated comparable task completion quality, but gpt-oss:20b showed significantly better speed and efficiency.

---

## Overall Performance

| Metric | gpt-oss:20b | qwen3:14b | Winner |
|--------|-------------|-----------|---------|
| **Pass Rate** | 86.7% (13/15) | 86.7% (13/15) | Tie |
| **Avg Duration** | 53.1s | 122.5s | gpt-oss:20b (2.31x faster) |
| **Avg Rounds** | 12.1 | 8.4 | qwen3:14b (more efficient) |
| **Min Duration** | 19.2s | 45.3s | gpt-oss:20b |
| **Max Duration** | 160.9s | 327.7s | gpt-oss:20b |

### Interpretation

- **Quality**: Both models achieve identical pass rates, indicating comparable code generation quality
- **Speed**: gpt-oss:20b completes tasks more than twice as fast on average
- **Efficiency**: qwen3:14b uses fewer rounds (8.4 vs 12.1), suggesting more concentrated reasoning per round, but this doesn't translate to faster overall completion
- **Consistency**: gpt-oss:20b has tighter duration range (19-161s) vs qwen3:14b (45-328s)

---

## Performance by Difficulty Level

### Level 4: Calculator with Tests

**Task**: Create calculator.py with add, subtract, multiply, divide functions. Write comprehensive tests. Run tests to verify.

| Metric | gpt-oss:20b | qwen3:14b |
|--------|-------------|-----------|
| **Pass Rate** | 60% (3/5) | 100% (5/5) |
| **Avg Duration** | 53.3s | 63.9s |
| **Avg Rounds** | 11.4 | 7.8 |

**Analysis**:
- qwen3:14b shows **better reliability** on L4 tasks (100% vs 60% pass rate)
- gpt-oss:20b slightly faster when successful (53.3s vs 63.9s)
- qwen3:14b more consistent across all 5 iterations

### Level 5: Refactor to Classes

**Task**: Create todo_list.py with TodoList class (add_task, complete_task, list_tasks methods). Include tests.

| Metric | gpt-oss:20b | qwen3:14b |
|--------|-------------|-----------|
| **Pass Rate** | 100% (5/5) | 80% (4/5) |
| **Avg Duration** | 51.3s | 95.2s |
| **Avg Rounds** | 11.8 | 6.6 |

**Analysis**:
- gpt-oss:20b achieves **perfect reliability** on L5 tasks
- gpt-oss:20b nearly **2x faster** (51.3s vs 95.2s)
- qwen3:14b had one significant outlier (193.6s failure on iteration 5)

### Level 6: Simple API

**Task**: Create simple REST API with Flask (GET/POST endpoints, error handling, tests).

| Metric | gpt-oss:20b | qwen3:14b |
|--------|-------------|-----------|
| **Pass Rate** | 100% (5/5) | 80% (4/5) |
| **Avg Duration** | 54.8s | 208.3s |
| **Avg Rounds** | 12.8 | 10.8 |

**Analysis**:
- gpt-oss:20b shows **superior performance** on complex L6 tasks
- gpt-oss:20b nearly **4x faster** (54.8s vs 208.3s)
- qwen3:14b struggles with API tasks (one failure + high durations: 223s, 328s, 209s, 165s, 117s)
- Performance gap widens significantly at higher complexity levels

---

## Detailed Results by Iteration

### gpt-oss:20b Results

| Iteration | Task | Level | Status | Duration | Rounds |
|-----------|------|-------|--------|----------|--------|
| 1 | Calculator | L4 | ✓ PASS | 62.5s | 9 |
| 1 | Refactor | L5 | ✓ PASS | 26.1s | 7 |
| 1 | API | L6 | ✓ PASS | 51.8s | 13 |
| 2 | Calculator | L4 | ✗ FAIL | 24.5s | 5 |
| 2 | Refactor | L5 | ✓ PASS | 160.9s | 27 |
| 2 | API | L6 | ✓ PASS | 37.4s | 9 |
| 3 | Calculator | L4 | ✓ PASS | 19.2s | 9 |
| 3 | Refactor | L5 | ✓ PASS | 21.5s | 8 |
| 3 | API | L6 | ✓ PASS | 48.2s | 10 |
| 4 | Calculator | L4 | ✓ PASS | 112.1s | 23 |
| 4 | Refactor | L5 | ✓ PASS | 24.3s | 9 |
| 4 | API | L6 | ✓ PASS | 38.6s | 9 |
| 5 | Calculator | L4 | ✗ FAIL | 47.9s | 11 |
| 5 | Refactor | L5 | ✓ PASS | 23.5s | 8 |
| 5 | API | L6 | ✓ PASS | 98.2s | 24 |

**Notable outliers**:
- Refactor L5 iteration 2: 160.9s (27 rounds) - significant outlier, likely hit complex edge case
- Calculator L4 iteration 4: 112.1s (23 rounds) - struggled but eventually passed
- API L6 iteration 5: 98.2s (24 rounds) - took longer than typical

### qwen3:14b Results

| Iteration | Task | Level | Status | Duration | Rounds |
|-----------|------|-------|--------|----------|--------|
| 1 | Calculator | L4 | ✓ PASS | 71.2s | 9 |
| 1 | Refactor | L5 | ✓ PASS | 63.0s | 7 |
| 1 | API | L6 | ✗ FAIL | 222.8s | 11 |
| 2 | Calculator | L4 | ✓ PASS | 60.8s | 7 |
| 2 | Refactor | L5 | ✓ PASS | 84.0s | 7 |
| 2 | API | L6 | ✓ PASS | 327.7s | 12 |
| 3 | Calculator | L4 | ✓ PASS | 80.3s | 9 |
| 3 | Refactor | L5 | ✓ PASS | 73.0s | 5 |
| 3 | API | L6 | ✓ PASS | 208.9s | 10 |
| 4 | Calculator | L4 | ✓ PASS | 45.3s | 7 |
| 4 | Refactor | L5 | ✓ PASS | 62.4s | 7 |
| 4 | API | L6 | ✓ PASS | 165.0s | 9 |
| 5 | Calculator | L4 | ✓ PASS | 62.2s | 7 |
| 5 | Refactor | L5 | ✗ FAIL | 193.6s | 7 |
| 5 | API | L6 | ✓ PASS | 116.9s | 12 |

**Notable outliers**:
- API L6 iteration 2: 327.7s (12 rounds) - massive slowdown, still passed
- API L6 iteration 1: 222.8s (11 rounds) - failed after long duration
- Refactor L5 iteration 5: 193.6s (7 rounds) - failed after long duration

---

## Speed Analysis

### Duration Distribution

**gpt-oss:20b**:
- Fastest: 19.2s (Calculator L4, iteration 3)
- Slowest: 160.9s (Refactor L5, iteration 2)
- Median: ~38s
- Most tasks complete in 20-60s range

**qwen3:14b**:
- Fastest: 45.3s (Calculator L4, iteration 4)
- Slowest: 327.7s (API L6, iteration 2)
- Median: ~71s
- High variance: 45-328s range
- API tasks consistently slow (116-328s)

### Speed by Task Type

| Task | gpt-oss:20b Avg | qwen3:14b Avg | Speedup |
|------|-----------------|---------------|---------|
| Calculator (L4) | 53.3s | 63.9s | 1.20x |
| Refactor (L5) | 51.3s | 95.2s | 1.85x |
| API (L6) | 54.8s | 208.3s | 3.80x |

**Trend**: Performance gap widens with task complexity
- Simple tasks (L4): gpt-oss:20b 20% faster
- Medium tasks (L5): gpt-oss:20b 85% faster
- Complex tasks (L6): gpt-oss:20b 280% faster

---

## Reliability Analysis

### Failure Patterns

**gpt-oss:20b failures** (2/15):
- Calculator L4, iteration 2: 24.5s, 5 rounds - early failure
- Calculator L4, iteration 5: 47.9s, 11 rounds - struggled longer before failing
- Both failures on L4 Calculator task specifically

**qwen3:14b failures** (2/15):
- API L6, iteration 1: 222.8s, 11 rounds - long struggle before failure
- Refactor L5, iteration 5: 193.6s, 7 rounds - long struggle before failure
- Both failures after extended durations, suggesting timeout or persistence issues

### Success Rate by Task

| Task | gpt-oss:20b | qwen3:14b | More Reliable |
|------|-------------|-----------|---------------|
| Calculator (L4) | 60% (3/5) | 100% (5/5) | qwen3:14b |
| Refactor (L5) | 100% (5/5) | 80% (4/5) | gpt-oss:20b |
| API (L6) | 100% (5/5) | 80% (4/5) | gpt-oss:20b |

**Interpretation**:
- qwen3:14b more reliable on simple tasks (L4)
- gpt-oss:20b more reliable on medium/complex tasks (L5, L6)
- gpt-oss:20b has perfect record on harder tasks despite being faster

---

## Efficiency Analysis: Rounds per Task

**Lower rounds = more efficient reasoning per round**

| Task | gpt-oss:20b Avg Rounds | qwen3:14b Avg Rounds |
|------|------------------------|----------------------|
| Calculator (L4) | 11.4 | 7.8 |
| Refactor (L5) | 11.8 | 6.6 |
| API (L6) | 12.8 | 10.8 |
| **Overall** | **12.1** | **8.4** |

**Analysis**:
- qwen3:14b uses **30% fewer rounds** overall (8.4 vs 12.1)
- qwen3:14b does more work per round (longer LLM thinking time)
- Despite fewer rounds, qwen3:14b is 2.31x slower overall
- This suggests qwen3:14b has slower per-round execution time

### Time per Round

| Model | Avg Duration | Avg Rounds | Avg Time/Round |
|-------|--------------|------------|----------------|
| gpt-oss:20b | 53.1s | 12.1 | 4.4s/round |
| qwen3:14b | 122.5s | 8.4 | 14.6s/round |

**Key Finding**: qwen3:14b takes **3.3x longer per round** (14.6s vs 4.4s)

This explains why fewer rounds don't translate to faster completion.

---

## Recommendations

### When to Use gpt-oss:20b

✅ **Best for**:
- Time-sensitive tasks requiring fast iteration
- Medium to complex coding tasks (L5-L6)
- Production workflows where speed matters
- Tasks requiring API development or integration
- Projects with tight deadlines

**Strengths**:
- 2.31x faster overall
- 3.3x faster per round
- Perfect reliability on L5-L6 tasks
- More consistent performance (tighter duration range)

### When to Use qwen3:14b

✅ **Best for**:
- Simple, well-defined tasks (L4)
- Projects where compute cost matters more than speed
- Tasks where reliability on basic operations is critical

**Strengths**:
- 100% success rate on L4 tasks
- Fewer rounds needed (30% reduction)
- Potentially more thoughtful per-round reasoning

**Weaknesses**:
- 2.31x slower overall
- Struggles with complex tasks (API development)
- High variance in completion times
- 3.3x slower per LLM round

---

## Cost-Benefit Analysis

Assuming Ollama local inference (no API costs), the main cost is **developer time**:

**Scenario**: Developer working on 10 L4-L6 tasks per day

| Model | Avg Task Time | Daily Total | Weekly Total |
|-------|---------------|-------------|--------------|
| gpt-oss:20b | 53.1s | 8m 51s | 1h 2m |
| qwen3:14b | 122.5s | 20m 25s | 2h 23m |
| **Time Saved** | **69.4s/task** | **11m 34s/day** | **1h 21m/week** |

**Over a month**: ~5.7 hours saved with gpt-oss:20b

---

## Conclusion

**Winner: gpt-oss:20b** for general coding agent tasks

While both models achieve identical overall pass rates (86.7%), gpt-oss:20b demonstrates clear advantages:

1. **Speed**: 2.31x faster overall, 3.3x faster per round
2. **Reliability on complex tasks**: Perfect 100% pass rate on L5-L6
3. **Consistency**: Tighter performance variance
4. **Scalability**: Performance advantage grows with task complexity

**Use qwen3:14b only when**: Task is simple (L4) and time is not a constraint.

**Recommended default**: gpt-oss:20b for the Jetbox coding agent
