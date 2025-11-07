# Context Inspection Report

**Generated**: 2025-11-07 18:45:50
**Tool**: Jetbox Context Inspector
**Purpose**: Identify context inefficiencies and optimization opportunities

## Executive Summary

### Key Metrics

| Metric | Value |
|--------|-------|
| Scenarios Analyzed | 3 |
| Total Rounds | 61 |
| Avg Context Size | 41.7K tokens |
| Max Context Size | 110.0K tokens |
| Total Duplication Found | 17.7K tokens |

### Top Recommendations
1. 🔴 **Cache tool definitions across rounds** - Save 8.0K tokens
2. 🔴 **Deduplicate system prompt instructions** - Save 3.5K tokens
3. 🟡 **Compress historical context** - Save 5.0K tokens


## Per-Scenario Analysis

### Scenario: simple

**Metrics**:

| Metric | Value |
|--------|-------|
| Total Rounds | 8 |
| Avg Context Size | 15.0K |
| Max Context Size | 20.0K |
| Avg System Prompt | 8.0K |
| Avg Tool Definitions | 3.0K |
| Duplication Found | 1.2K |

**Context Growth**:

```
                                                     ███████
                                             ████████       
                                      ███████               
                       ███████████████                      
                                                            
               ████████                                     
        ███████                                             
████████                                                    
```

Growth Rate: **linear**

**Top Behaviors by Token Usage**:

```
FileToolsBehavior           │ ██████████████████████████████ 5,000
HierarchicalContextBehavior │ █████████████████████ 3,500
LoopDetectionBehavior       │ ████████████ 2,000
```

### Scenario: medium

**Metrics**:

| Metric | Value |
|--------|-------|
| Total Rounds | 18 |
| Avg Context Size | 35.0K |
| Max Context Size | 50.0K |
| Avg System Prompt | 12.0K |
| Avg Tool Definitions | 5.0K |
| Duplication Found | 4.5K |

**Context Growth**:

```
                                                       █████
                                   ████████████████████     
                              █████                         
                    ██████████                              
               █████                                        
          █████                                             
     █████                                                  
█████                                                       
```

Growth Rate: **linear**

**Top Behaviors by Token Usage**:

```
FileToolsBehavior           │ ██████████████████████████████ 8,000
HierarchicalContextBehavior │ ██████████████████████ 6,000
CommandToolsBehavior        │ ███████████████ 4,000
WorkspaceTaskNotesBehavior  │ ███████████ 3,000
LoopDetectionBehavior       │ █████████ 2,500
```

### Scenario: complex

**Metrics**:

| Metric | Value |
|--------|-------|
| Total Rounds | 35 |
| Avg Context Size | 75.0K |
| Max Context Size | 110.0K |
| Avg System Prompt | 18.0K |
| Avg Tool Definitions | 8.0K |
| Duplication Found | 12.0K |

**Context Growth**:

```
                                                        ████
                                               █████████    
                                 ██████████████             
                            █████                           
                   █████████                                
              █████                                         
          ████                                              
██████████                                                  
```

Growth Rate: **slightly exponential**

**Top Behaviors by Token Usage**:

```
DelegationBehavior          │ ██████████████████████████████ 15,000
FileToolsBehavior           │ ████████████████████████ 12,000
HierarchicalContextBehavior │ ████████████████████ 10,000
StatusDisplayBehavior       │ ██████████████ 7,000
CommandToolsBehavior        │ ████████████ 6,000
```


## Duplication Deep Dive

### Exact Duplicates

| Content (truncated) | Count | Locations | Token Waste |
|---------------------|-------|-----------|-------------|
| def write_file(path: str, content: str) -> dict:... | 8 | r2, r4, r6 | 150 |
| Tool definitions are loaded from behaviors... | 12 | r1, r3, r5 | 240 |

### Fuzzy Duplicates (>80% similarity)

| Content (truncated) | Similarity | Locations | Token Waste |
|---------------------|------------|-----------|-------------|
| The agent should complete the current task... | 87% | r5, r8 | 95 |



## Behavior Contribution Matrix

| Behavior | System Prompt | Tool Defs | Context Inject | Total | ROI Score |
|----------|---------------|-----------|----------------|-------|----------|
| DelegationBehavior | 2.0K | 8.0K | 5.0K | 15.0K | 0.65 |
| FileToolsBehavior | 800 | 3.2K | 1.0K | 5.0K | 0.85 |
| CommandToolsBehavior | 500 | 2.5K | 1.0K | 4.0K | 0.80 |
| HierarchicalContextBehavior | 1.5K | 0 | 2.0K | 3.5K | 0.78 |
| LoopDetectionBehavior | 400 | 800 | 800 | 2.0K | 0.92 |

**ROI Score**: Value provided / tokens consumed (higher is better)


## Prioritized Recommendations

### 🔴 HIGH Priority

#### 1. Cache tool definitions across rounds

Tool definitions are being serialized in every context snapshot. Implement caching to reference definitions by hash instead of including full text.

**Impact**: Save 8.0K tokens per run

**Difficulty**: Medium

**Location**: `base_agent.py:425`

**Implementation**:
```python
# Add tool definition cache
self._tool_def_cache = {}

def _get_tool_defs(self):
    cache_key = hash(tuple(sorted(self.tools)))
    if cache_key not in self._tool_def_cache:
        self._tool_def_cache[cache_key] = self._serialize_tools()
    return self._tool_def_cache[cache_key]
```

#### 2. Deduplicate system prompt instructions

Multiple behaviors inject similar instructions into the system prompt. Create a shared instruction registry to avoid duplication.

**Impact**: Save 3.5K tokens per run

**Difficulty**: High

**Location**: `base_agent.py:280`

### 🟡 MEDIUM Priority

#### 1. Compress historical context

Message history grows linearly. Implement rolling summarization for messages older than 5 rounds.

**Impact**: Save 5.0K tokens per run

**Difficulty**: Medium

**Location**: `behaviors/hierarchical_context.py:156`

#### 2. Optimize DelegationBehavior tool definitions

DelegationBehavior has the lowest ROI (0.65). Tool definitions are verbose. Simplify descriptions and parameter schemas.

**Impact**: Save 4.0K tokens per run

**Difficulty**: Low

**Location**: `behaviors/delegation.py:78`

### 🟢 LOW Priority

#### 1. Implement context pruning strategy

Automatically remove low-value context elements when approaching token limits (e.g., old error messages, redundant status updates).

**Impact**: Save 2.0K tokens per run

**Difficulty**: High

**Location**: `base_agent.py:520`



## Comparative Analysis

### Context Size by Scenario

```
simple  │ ███████ 20,000
medium  │ ██████████████████ 50,000
complex │ ████████████████████████████████████████ 110,000
```

### Growth Rate Comparison

| Scenario | Growth Rate | Final Context | Duplication % |
|----------|-------------|---------------|---------------|
| simple | linear | 20.0K | 6.0% |
| medium | linear | 50.0K | 9.0% |
| complex | slightly exponential | 110.0K | 10.9% |

### Key Insights

- ⚠️ **High complexity multiplication**: 5.5x context growth from simple to complex scenarios
- ⚠️ **High duplication**: Average 5.9K duplicated tokens per scenario


---

## How to Use This Report

1. **Start with HIGH priority recommendations** - Highest impact, quickest wins
2. **Review behavior contribution matrix** - Identify low-ROI behaviors to optimize
3. **Investigate exact duplicates** - Often easy to fix with caching or deduplication
4. **Monitor growth rate** - Exponential growth needs immediate attention
5. **Compare scenarios** - Understand how complexity affects context usage

## Next Steps

1. Implement HIGH priority recommendations
2. Re-run context inspection to validate improvements
3. Set up monitoring for context size trends
4. Document optimization guidelines for future behaviors

---

*Generated by Jetbox Context Inspector*