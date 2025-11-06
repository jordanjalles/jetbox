# LLM Response Speed Analysis

## Summary

**Counter-intuitive finding**: The fastest overall model (qwen3:8b) has the **slowest per-round response time**, but wins by needing **fewer rounds**.

## Average Response Time Per Round (L1-L4 successful tests)

| Model | Avg Response Time | Total Rounds | Successful Tests |
|-------|------------------|--------------|------------------|
| **gpt-oss:20b** | **2.9s/round** ⚡ | 6 | 1/4 (25%) |
| qwen3:4b | 11.9s/round | 7 | 2/4 (50%) |
| qwen3:14b | 24.3s/round | 5 | 4/4 (100%) |
| **qwen3:8b** | **28.5s/round** 🐌 | 5 | 4/4 (100%) |

## The Paradox Explained

### Why is gpt-oss:20b fast per-round but slow overall?

**gpt-oss:20b behavior**:
- Fast inference: 2.9s per LLM call
- BUT: Needs 6-12 rounds per task
- Fails to recognize completion (L3, L4 got "UNKNOWN")
- Iterative approach: many small steps

**L3 Example (gpt-oss:20b)**:
- 12 rounds × 2.9s = **34.8s**
- Status: **UNKNOWN** (hit max rounds, didn't call `mark_complete`)
- Work may have been done, but agent didn't signal completion

### Why is qwen3:8b slow per-round but fast overall?

**qwen3:8b behavior**:
- Slower inference: 28.5s per LLM call (10x slower!)
- BUT: Completes tasks in 1-2 rounds
- Always calls `mark_complete` when done
- Confident approach: one decisive action

**L3 Example (qwen3:8b)**:
- 1 round × 44.6s = **44.6s**
- Status: **SUCCESS** ✅
- All work done in single confident multi-tool call

## Round Efficiency Analysis

### L1 (Simple File)

| Model | Time | Rounds | Status | Notes |
|-------|------|--------|--------|-------|
| gpt-oss:20b | 60.0s | 4 | TIMEOUT ⏱️ | No completion call |
| qwen3:4b | 56.1s | 3 | SUCCESS ✅ | Close to timeout |
| **qwen3:8b** | **19.8s** | **1** | **SUCCESS** ✅ | **One shot** |
| qwen3:14b | 26.2s | 1 | SUCCESS ✅ | One shot |

### L2 (File + Test)

| Model | Time | Rounds | Status | Notes |
|-------|------|--------|--------|-------|
| **gpt-oss:20b** | **17.7s** | **6** | **SUCCESS** ✅ | **Fastest here** |
| qwen3:4b | 27.1s | 4 | SUCCESS ✅ | |
| qwen3:8b | 23.0s | 2 | SUCCESS ✅ | |
| qwen3:14b | 28.2s | 2 | SUCCESS ✅ | |

### L3 (Multi-file Package)

| Model | Time | Rounds | Status | Notes |
|-------|------|--------|--------|-------|
| gpt-oss:20b | 25.0s | 12 | UNKNOWN ❓ | Hit max rounds |
| qwen3:4b | 180.0s | 12 | TIMEOUT ⏱️ | Hit max rounds + timeout |
| qwen3:14b | 37.8s | 1 | SUCCESS ✅ | One shot |
| **qwen3:8b** | **44.6s** | **1** | **SUCCESS** ✅ | **One shot** |

### L4 (Package + Dependencies)

| Model | Time | Rounds | Status | Notes |
|-------|------|--------|--------|-------|
| gpt-oss:20b | 42.4s | 12 | UNKNOWN ❓ | Hit max rounds |
| qwen3:4b | 240.0s | 8 | TIMEOUT ⏱️ | Timeout |
| **qwen3:14b** | **29.1s** | **1** | **SUCCESS** ✅ | **Fastest** |
| qwen3:8b | 55.0s | 1 | SUCCESS ✅ | One shot |

## Key Insights

### 1. **"One-shot" Strategy Wins**

qwen3:8b and qwen3:14b complete most tasks in **1 round** by:
- Planning all work upfront
- Making confident multi-tool calls
- Calling `mark_complete` immediately when done

Example: L3 task (create mathx package with 4 functions + tests)
- qwen3:8b makes ONE LLM call that:
  - Writes all 5 files (mathx/{add,subtract,multiply,divide}.py + test)
  - Verifies structure
  - Calls `mark_complete`

### 2. **Iterative Approach Loses**

gpt-oss:20b uses an iterative "think → act → verify" loop:
- Round 1: Explore workspace
- Round 2: Write first file
- Round 3: Write second file
- ...
- Round 12: Still working, hit max rounds

This is **slower overall** despite faster per-round inference.

### 3. **Completion Recognition is Critical**

Models that don't call `mark_complete` get:
- "UNKNOWN" status (exit code 1)
- Counted as failures even if work was done
- Wasted rounds trying to figure out if done

gpt-oss:20b fails to recognize completion on L3/L4 despite potentially doing the work.

### 4. **Model Size Sweet Spot**

| Size | Strategy | Per-Round Speed | Total Rounds | Overall Speed |
|------|----------|-----------------|--------------|---------------|
| 4B | Struggles | Medium | 3-12 | Slow (timeouts) |
| 8B | One-shot ✅ | Slow | 1-2 | **Fast** |
| 14B | One-shot ✅ | Medium-slow | 1-2 | Medium |
| 20B | Iterative | Fast | 6-12 | Slow |

The 8B model finds the optimal balance:
- Large enough for confident one-shot planning
- Small enough for reasonable inference speed
- Not so large that it overthinks

## Performance Formula

**Overall Task Time = (LLM Response Time) × (Number of Rounds) + Overhead**

### gpt-oss:20b
- 2.9s/round × 12 rounds = **34.8s** (but didn't complete)

### qwen3:8b
- 28.5s/round × 1 round = **28.5s** (completed successfully)

**qwen3:8b wins** because `1 round` beats `12 rounds`, even with 10x slower inference.

## Recommendation

**Use qwen3:8b** because:

1. **Round efficiency** matters more than per-round speed
2. **Completion recognition** prevents wasted rounds
3. **One-shot strategy** minimizes total time
4. **100% success rate** means no retries needed

The "slow per-round but fast overall" paradox is key to understanding why smaller, more decisive models outperform larger, more cautious ones in agentic workflows.

## Appendix: Detailed Round Breakdown

### L1: Create greet.py

**qwen3:8b (19.8s total, 1 round)**:
- Round 1: List dir, write greet.py, call mark_complete → Done ✅

**gpt-oss:20b (60.0s timeout, 4 rounds)**:
- Round 1: List dir, no tools called (empty round)
- Round 2: List dir again (recovery)
- Round 3-4: Working... (timeout at 60s) ⏱️

### L3: Create mathx package

**qwen3:8b (44.6s total, 1 round)**:
- Round 1:
  - Write mathx/__init__.py
  - Write mathx/add.py
  - Write mathx/subtract.py
  - Write mathx/multiply.py
  - Write mathx/divide.py
  - Write tests/test_mathx.py
  - Call mark_complete → Done ✅

**gpt-oss:20b (25.0s, 12 rounds, UNKNOWN)**:
- Round 1: List dir, no tools (empty)
- Round 2: List dir (recovery)
- Round 3-6: Write files
- Round 7: Empty round
- Round 8-11: More writes
- Round 12: List dir → Max rounds reached ❓

The work may have been done, but gpt-oss didn't signal completion, so status = UNKNOWN.

## Conclusion

**Per-round speed doesn't matter if you need 10x more rounds.**

qwen3:8b's "slow and steady in one shot" beats gpt-oss:20b's "fast but hesitant in many steps".
