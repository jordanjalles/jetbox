# Context Inspection Analysis Report

Generated from: .context_inspection/test_data

## Executive Summary

- **Agents analyzed**: 1
- **Behaviors detected**: 5
- **Token waste from duplication**: 1,358
- **Recommendations**: 2

### Top 3 Recommendations

1. **[MEDIUM]** Agent 'task_executor' has steady context growth (Save ~250 tokens)
2. **[LOW]** Reduce context duplication (1,358 wasted tokens) (Save ~1,358 tokens)

## Duplication Analysis

- **Exact duplicate system prompts**: 1
- **Exact duplicate messages**: 5
- **Exact duplicate tools**: 4
- **Fuzzy duplicate system prompts**: 6
- **Fuzzy duplicate messages**: 0

## Growth Analysis

### Agent: task_executor

- **Rounds**: 4
- **Initial context**: 450 chars
- **Final context**: 1,200 chars
- **Growth rate**: 166.7%
- **Pattern**: linear
- **Rounds to token limit**: 676
- **Growth causes**:
  - Message history growing (2 -> 8)
  - Tool count increased (3 -> 4)

## Behavior Token Contributions

| Behavior | System Prompt | Tools | Total |
|----------|---------------|-------|-------|
| LoopDetectionBehavior | 53 | 131 | 184 |
| FileToolsBehavior | 33 | 81 | 114 |
| CommandToolsBehavior | 33 | 81 | 114 |
| DelegationBehavior | 20 | 50 | 70 |
| StatusDisplayBehavior | 20 | 50 | 70 |

## Detailed Recommendations

### 1. [MEDIUM] Agent 'task_executor' has steady context growth

**Category**: growth

Context growing linearly at 166.7%

**Suggested fixes**:
- Review context strategy configuration
- Consider message history limits

**Potential savings**: ~250 tokens
**Difficulty**: low

### 2. [LOW] Reduce context duplication (1,358 wasted tokens)

**Category**: duplication

Significant duplication detected in system prompts and messages

**Suggested fixes**:
- Implement context deduplication in context strategies
- Cache system prompts instead of rebuilding each round
- Use message compression for repeated content

**Potential savings**: ~1,358 tokens
**Difficulty**: medium
