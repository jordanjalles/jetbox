# Context Strategy Refactoring: Decoupling for Testability

**Date:** 2025-10-29
**Goal:** Decouple context management strategies from agents for independent testing and benchmarking

## Problem Statement

Previously, context management logic was tightly coupled to agent implementations:
- `TaskExecutorAgent` had `build_context()` method calling function-based strategies
- No easy way to swap strategies for testing
- Difficult to compare strategies objectively
- Strategy behavior (like message clearing) hardcoded in agent logic

## Solution: Strategy Pattern

Implemented proper Strategy pattern with:
1. **Abstract base class** (`ContextStrategy`)
2. **Concrete implementations** (Hierarchical, AppendUntilFull)
3. **Pluggable strategies** (agents accept strategy parameter)
4. **Standardized interface** (all strategies implement same methods)

## Architecture

### Class Hierarchy

```
ContextStrategy (ABC)
├── build_context() → list[dict]
├── should_clear_on_transition() → bool
├── get_name() → str
└── estimate_context_size() → int

HierarchicalStrategy(ContextStrategy)
├── history_keep: int = 12
├── Clears messages on subtask transitions
└── Keeps last N message exchanges only

AppendUntilFullStrategy(ContextStrategy)
├── max_tokens: int = 8000
├── recent_keep: int = 20
├── Does NOT clear on transitions
└── Compacts when near 80% of token limit
```

### Usage

```python
# Default behavior (Hierarchical)
agent = TaskExecutorAgent(
    workspace=workspace,
    goal="Create hello.py",
)

# Explicit hierarchical
agent = TaskExecutorAgent(
    workspace=workspace,
    goal="Create hello.py",
    context_strategy=HierarchicalStrategy(history_keep=10),
)

# Append-until-full strategy
agent = TaskExecutorAgent(
    workspace=workspace,
    goal="Create hello.py",
    context_strategy=AppendUntilFullStrategy(max_tokens=10000, recent_keep=30),
)
```

## Implementation Details

### 1. Abstract Base Class

**File:** `context_strategies.py:18-76`

```python
class ContextStrategy(ABC):
    """Abstract base class for context management strategies."""

    @abstractmethod
    def build_context(
        self,
        context_manager: ContextManager,
        messages: list[dict[str, Any]],
        system_prompt: str,
        config: Any,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Build context for LLM based on strategy."""
        pass

    @abstractmethod
    def should_clear_on_transition(self) -> bool:
        """Whether messages should be cleared on subtask transitions."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name for reporting."""
        pass

    @abstractmethod
    def estimate_context_size(self, context: list[dict[str, Any]]) -> int:
        """Estimate context size in tokens."""
        pass
```

### 2. Hierarchical Strategy

**File:** `context_strategies.py:79-223`

**Features:**
- Keeps only recent N message exchanges
- Clears messages on subtask transitions
- Includes loop detection warnings
- Includes filesystem probe state
- Includes jetbox notes (if available)

**Best for:**
- Task-focused execution
- Agents with clear subtask boundaries
- Scenarios where focus is critical
- Minimizing token usage

**Configuration:**
```python
HierarchicalStrategy(history_keep=12)
```

### 3. Append-Until-Full Strategy

**File:** `context_strategies.py:226-308`

**Features:**
- Appends all messages initially
- Compacts when near 80% of token limit
- Keeps recent N messages during compaction
- Summarizes older messages
- Does NOT clear on subtask transitions

**Best for:**
- Conversational agents (Orchestrator)
- Long-running sessions with context continuity
- Tasks requiring reference to earlier work
- Multi-turn reasoning

**Configuration:**
```python
AppendUntilFullStrategy(max_tokens=8000, recent_keep=20)
```

### 4. TaskExecutorAgent Integration

**File:** `task_executor_agent.py:42,72,163,356-360`

**Changes:**

1. **Accept strategy in __init__:**
```python
def __init__(
    self,
    workspace: Path,
    goal: str | None = None,
    context_strategy: ContextStrategy | None = None,
):
    # Default to hierarchical if not specified
    self.context_strategy = context_strategy or HierarchicalStrategy()
```

2. **Use strategy in build_context:**
```python
def build_context(self) -> list[dict[str, Any]]:
    return self.context_strategy.build_context(
        context_manager=self.context_manager,
        messages=self.state.messages,
        system_prompt=self.get_system_prompt(),
        config=self.config,
        probe_state_func=self._probe_state if hasattr(self, '_probe_state') else None,
        workspace=self.workspace_manager.workspace_dir if self.workspace_manager else None,
    )
```

3. **Respect strategy's clearing policy:**
```python
if isinstance(actual_result, dict) and actual_result.get("status") in ["subtask_advanced", "task_advanced"]:
    if self.context_strategy.should_clear_on_transition():
        old_count = len(self.state.messages)
        self.clear_messages()
        messages.clear()
        print(f"[context_isolation] Cleared {old_count} messages (strategy: {self.context_strategy.get_name()})")
```

### 5. Backward Compatibility

**File:** `context_strategies.py:311-413`

Old function-based APIs still work, now delegating to new classes:

```python
def build_hierarchical_context(...) -> list[dict]:
    """Backward compatibility wrapper."""
    strategy = HierarchicalStrategy()
    return strategy.build_context(...)

def build_append_context(...) -> list[dict]:
    """Backward compatibility wrapper."""
    strategy = AppendUntilFullStrategy(max_tokens, recent_keep)
    return strategy.build_context(...)
```

**Result:** All existing code continues to work without changes.

## Benchmark Test

**File:** `tests/test_context_strategy_benchmark.py`

### Test Suite

Compares both strategies across 5 tasks of increasing complexity:

1. **simple_file** - Single hello.py file
2. **two_functions** - math_utils.py with 2 functions
3. **with_tests** - calculator.py with tests
4. **multi_file_web** - HTML/CSS/JS web page
5. **package_structure** - Python package with __init__.py

### Metrics Collected

For each task × strategy combination:
- ✅ **Success rate** - Did it complete?
- ⏱️ **Rounds to completion** - How many LLM calls?
- 🕐 **Wall time** - Total execution time
- 📊 **Context tokens** - Final context size
- 💬 **Total messages** - Message count at end

### Running the Benchmark

```bash
PYTHONPATH=. python tests/test_context_strategy_benchmark.py
```

**Output:**
- Per-task results with immediate feedback
- Strategy comparison table
- Winner analysis (fewer rounds, less context)
- JSON results file: `context_strategy_benchmark_results.json`

## Benefits

### 1. Testability
- Can now compare strategies objectively
- Benchmark suite provides quantitative data
- Easy to add new strategies for testing

### 2. Flexibility
- Agents can use different strategies
- Strategy parameters are configurable
- Can switch strategies at runtime (future)

### 3. Clarity
- Strategy behavior is explicit
- Clear interface (abstract methods)
- Self-documenting code (get_name, should_clear_on_transition)

### 4. Maintainability
- Changes to one strategy don't affect others
- New strategies don't require agent changes
- Backward compatibility preserved

### 5. Optimization
- Can tune strategy parameters per use case
- Benchmark data guides optimization decisions
- Easy to A/B test strategy changes

## Expected Results

### Hierarchical Strategy

**Advantages:**
- ✅ Lower context tokens (only keeps last N messages)
- ✅ Better focus on current subtask
- ✅ Faster LLM calls (smaller context)
- ✅ Clearer subtask boundaries

**Disadvantages:**
- ❌ May forget important earlier context
- ❌ Can't reference work from 10+ subtasks ago
- ❌ Requires jetbox notes for long-term memory

### Append-Until-Full Strategy

**Advantages:**
- ✅ Preserves full conversation history
- ✅ Can reference any earlier work
- ✅ Better for multi-turn reasoning
- ✅ No context loss

**Disadvantages:**
- ❌ Higher token usage (keeps all messages)
- ❌ Slower LLM calls (larger context)
- ❌ May include irrelevant old messages
- ❌ Requires compaction logic

## Future Enhancements

### 1. Additional Strategies

```python
class SlidingWindowStrategy(ContextStrategy):
    """Keep last N messages, but slide window instead of clearing."""
    pass

class ImportanceBasedStrategy(ContextStrategy):
    """Keep messages based on importance scoring."""
    pass

class HybridStrategy(ContextStrategy):
    """Hierarchical for subtasks, append for tasks."""
    pass
```

### 2. Dynamic Strategy Switching

```python
agent = TaskExecutorAgent(workspace, goal)

# Start with hierarchical
agent.context_strategy = HierarchicalStrategy()

# Switch to append for long reasoning
if complex_reasoning_detected():
    agent.context_strategy = AppendUntilFullStrategy()
```

### 3. Strategy Auto-Selection

```python
def auto_select_strategy(task_complexity, goal_type):
    if task_complexity < 3 and goal_type == "code":
        return HierarchicalStrategy()
    elif goal_type == "conversation":
        return AppendUntilFullStrategy()
    else:
        return HybridStrategy()
```

### 4. Strategy Tuning Based on Model

```python
# Smaller models need tighter context
if model == "qwen2.5-coder:3b":
    return HierarchicalStrategy(history_keep=8)

# Larger models can handle more context
elif model == "qwen2.5-coder:14b":
    return HierarchicalStrategy(history_keep=16)
```

## Migration Guide

### For Existing Code

No changes required! Backward compatibility wrappers handle old function calls.

### For New Code

**Before:**
```python
from context_strategies import build_hierarchical_context

context = build_hierarchical_context(
    context_manager=cm,
    messages=msgs,
    system_prompt=prompt,
    config=cfg,
)
```

**After:**
```python
from context_strategies import HierarchicalStrategy

strategy = HierarchicalStrategy(history_keep=12)
context = strategy.build_context(
    context_manager=cm,
    messages=msgs,
    system_prompt=prompt,
    config=cfg,
)
```

### For TaskExecutorAgent

**Before:**
```python
agent = TaskExecutorAgent(workspace, goal)
# Always used hierarchical strategy
```

**After:**
```python
# Default (hierarchical)
agent = TaskExecutorAgent(workspace, goal)

# Or explicit
agent = TaskExecutorAgent(
    workspace=workspace,
    goal=goal,
    context_strategy=AppendUntilFullStrategy(),
)
```

## Testing

### Unit Tests

Each strategy can be tested independently:

```python
def test_hierarchical_clears_on_transition():
    strategy = HierarchicalStrategy()
    assert strategy.should_clear_on_transition() == True

def test_append_does_not_clear():
    strategy = AppendUntilFullStrategy()
    assert strategy.should_clear_on_transition() == False
```

### Integration Tests

Full agent execution with each strategy:

```python
def test_agent_with_hierarchical():
    agent = TaskExecutorAgent(
        workspace=tmp,
        goal="Create hello.py",
        context_strategy=HierarchicalStrategy(),
    )
    result = agent.run()
    assert result["status"] == "success"
```

### Benchmark Tests

Comparative analysis across strategies:

```python
python tests/test_context_strategy_benchmark.py
```

## Conclusion

This refactoring:
- ✅ Decouples context management from agents
- ✅ Enables objective strategy comparison
- ✅ Maintains backward compatibility
- ✅ Provides clear extension points
- ✅ Improves code maintainability

The benchmark suite will provide data-driven insights into which strategy works best for different task types, enabling optimization based on real performance metrics.

---

**Status:** Implementation complete, benchmarks running
**Next:** Analyze benchmark results and update recommendations
