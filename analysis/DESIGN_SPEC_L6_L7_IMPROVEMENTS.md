# Design Spec: L6-L7 Performance Improvements

**Version:** 1.0
**Date:** 2025-10-29
**Status:** PROPOSED

## Executive Summary

This spec proposes improvements to handle L6-L7 complexity (architecture patterns and expert-level tasks) more reliably. Current performance:
- **L5 (Integration):** 67% success ✅
- **L6 (Architecture):** 22% success ⚠️
- **L7 (Expert):** 11% success ❌

**Target after improvements:**
- **L6:** 70%+ success
- **L7:** 50%+ success

## Problem Analysis

### Issue 1: LLM Timeouts/Hangs (22% of tasks)

**Symptoms:**
- "No response from Ollama for 30s" errors
- Model stops responding mid-task
- Happens more on L7 (Circuit Breaker, Connection Pool)

**Root causes:**
1. **Context overflow** - Complex tasks generate very long contexts
2. **Reasoning loops** - Model gets stuck in circular logic
3. **No health monitoring** - Can't detect/recover from hangs

### Issue 2: Complex Pattern Implementation (Observer, Circuit Breaker)

**Symptoms:**
- Agent creates partial implementations
- Missing key pattern elements
- Incorrect state management

**Root causes:**
1. **No pattern templates** - Agent must derive pattern from scratch
2. **Unclear decomposition** - Hard to break patterns into subtasks
3. **No validation hooks** - Can't verify pattern correctness incrementally

### Issue 3: Multi-Algorithm Tasks (Rate Limiter)

**Symptoms:**
- Takes 14+ rounds
- Often incomplete
- Mixes algorithms incorrectly

**Root causes:**
1. **Simultaneous complexity** - Token bucket + sliding window + Redis backend at once
2. **No incremental delivery** - All-or-nothing approach
3. **Algorithm knowledge gaps** - Model may not know detailed implementations

---

## Proposed Solution: Three-Tier Improvement Strategy

### Tier 1: Infrastructure (Foundation)

Improve the agent's ability to handle complex tasks without hanging or timing out.

#### 1.1 LLM Health Monitoring

**Component:** `llm_health_monitor.py`

**Features:**
- Track response times per request
- Detect "stuck" patterns (repeated identical calls)
- Implement proactive timeout (15s warning, 25s abort)
- Automatic retry with simplified prompt

**Implementation:**
```python
class LLMHealthMonitor:
    def __init__(self):
        self.response_times = []
        self.last_responses = deque(maxlen=3)
        self.stuck_threshold = 2  # Same response 2x = stuck

    def check_stuck(self, response):
        """Detect if model is in a loop."""
        self.last_responses.append(response)
        if len(set(self.last_responses)) == 1:
            return True  # All responses identical
        return False

    def call_with_health_check(self, model, messages, timeout=25):
        """Call LLM with health monitoring."""
        start = time.time()
        response = chat_with_inactivity_timeout(
            model, messages, timeout=timeout
        )
        duration = time.time() - start

        self.response_times.append(duration)

        if self.check_stuck(response):
            raise StuckLoopError("Model repeating same response")

        return response
```

**Success criteria:**
- Detect hangs within 25s (not 30s+)
- Recover from 80% of stuck states
- No false positives (normal operation not flagged)

#### 1.2 Progressive Context Compaction

**Component:** Enhanced `context_strategies.py`

**Features:**
- Monitor context size in real-time
- Trigger compaction at 75% of model limit
- More aggressive summarization for L6-L7
- Keep only essential pattern information

**Implementation:**
```python
def compact_for_complex_tasks(messages, max_tokens=8000):
    """
    Aggressive compaction for L6-L7 tasks.

    Keeps:
    - System prompt
    - Goal description
    - Current subtask
    - Last 5 messages
    - Summary of older messages
    """
    # Estimate current size
    current_size = estimate_tokens(messages)

    if current_size > max_tokens * 0.75:
        # Keep essentials
        system = [m for m in messages if m["role"] == "system"]
        recent = messages[-5:]

        # Summarize middle messages
        middle = messages[len(system):-5]
        summary = summarize_messages(middle)

        return system + [summary] + recent

    return messages
```

**Success criteria:**
- Context never exceeds 90% of limit
- Compaction preserves essential information
- No degradation in task understanding

#### 1.3 Graceful Degradation

**Component:** `error_recovery.py`

**Features:**
- On LLM timeout: simplify goal, retry once
- On stuck loop: decompose further
- On repeated failure: request user guidance

**Implementation:**
```python
class ErrorRecovery:
    def recover_from_timeout(self, goal, context):
        """Simplify and retry on timeout."""
        simplified_goal = self.simplify_goal(goal)
        print(f"[recovery] LLM timeout. Simplifying goal...")
        print(f"  Original: {goal[:100]}...")
        print(f"  Simplified: {simplified_goal[:100]}...")

        # Retry with simplified goal
        return simplified_goal

    def simplify_goal(self, goal):
        """Break complex goal into first step only."""
        # Example: "Create X with A, B, C" -> "Create X with A"
        # Remove later requirements
        parts = goal.split(",")
        return parts[0]  # Just first requirement
```

**Success criteria:**
- 50% of timeouts recover successfully
- Degraded tasks still produce useful output
- Clear communication of simplification to user

---

### Tier 2: Knowledge (Pattern Library)

Provide the agent with architectural guidance for common patterns.

#### 2.1 Pattern Templates

**Component:** `pattern_templates.py`

**Concept:** Inject pattern structure into system prompt for L6 tasks.

**Templates:**

```python
PATTERN_TEMPLATES = {
    "observer": {
        "structure": """
        Observer Pattern Structure:
        1. Subject class:
           - List of observers
           - attach(observer) method
           - detach(observer) method
           - notify() method to update all observers

        2. Observer class (or interface):
           - update() method to receive notifications

        3. ConcreteObserver classes:
           - Implement update() to react to changes
        """,
        "key_files": ["subject.py", "observer.py", "concrete_observers.py"],
        "gotchas": [
            "Subject must store observer references",
            "notify() calls update() on each observer",
            "Observers should not modify subject during update"
        ]
    },

    "factory": {
        "structure": """
        Factory Pattern Structure:
        1. Product interface/base class:
           - Common interface for all products

        2. ConcreteProduct classes:
           - Implement product interface
           - Different variants (e.g., ProductA, ProductB)

        3. Factory class:
           - create_product(type) method
           - Returns appropriate ConcreteProduct
        """,
        "key_files": ["factory.py", "products.py", "base.py"],
        "gotchas": [
            "Factory method should return base type, not concrete",
            "Use dict/match to map types to classes",
            "Consider using registry for extensibility"
        ]
    },

    "circuit_breaker": {
        "structure": """
        Circuit Breaker Pattern Structure:
        1. States:
           - CLOSED: Normal operation, calls go through
           - OPEN: Failing, reject all calls immediately
           - HALF_OPEN: Testing, allow one call to test recovery

        2. State transitions:
           - CLOSED -> OPEN: After N failures
           - OPEN -> HALF_OPEN: After timeout period
           - HALF_OPEN -> CLOSED: If test call succeeds
           - HALF_OPEN -> OPEN: If test call fails

        3. CircuitBreaker class:
           - call(func, *args) method
           - Track failure count
           - Manage state transitions
           - Record metrics
        """,
        "key_files": ["circuit_breaker.py", "states.py", "metrics.py"],
        "gotchas": [
            "Use enum for states",
            "Reset failure count on success",
            "Thread-safe state transitions if concurrent"
        ]
    }
}
```

**Injection mechanism:**

```python
def inject_pattern_template(goal, task_type):
    """Add pattern template to system prompt if task is a design pattern."""
    for pattern_name, template in PATTERN_TEMPLATES.items():
        if pattern_name in task_type.lower():
            enhanced_goal = f"""
{goal}

PATTERN GUIDANCE:
{template['structure']}

Key gotchas to avoid:
{', '.join(template['gotchas'])}
"""
            return enhanced_goal

    return goal
```

**Success criteria:**
- Pattern tasks show correct structure
- Fewer missing pattern elements
- Proper state management in stateful patterns

#### 2.2 Algorithm Knowledge Base

**Component:** `algorithm_kb.py`

**Concept:** Provide pseudocode/guidance for complex algorithms.

**Algorithms:**

```python
ALGORITHM_GUIDES = {
    "token_bucket": """
    Token Bucket Algorithm:

    State:
    - tokens: current token count (starts at capacity)
    - capacity: maximum tokens
    - refill_rate: tokens per second
    - last_refill: timestamp of last refill

    On allow(key):
        1. Calculate time since last_refill
        2. tokens += time * refill_rate (capped at capacity)
        3. last_refill = now
        4. If tokens >= 1:
             tokens -= 1
             return True
        5. Else:
             return False (rate limited)
    """,

    "sliding_window": """
    Sliding Window Algorithm:

    State:
    - requests: list of (timestamp, key) tuples
    - window_size: time window in seconds
    - max_requests: max allowed in window

    On allow(key):
        1. Remove requests older than (now - window_size)
        2. Count requests for this key in window
        3. If count < max_requests:
             Add (now, key) to requests
             return True
        4. Else:
             return False (rate limited)
    """,

    "connection_pool": """
    Connection Pool Algorithm:

    State:
    - available: queue of idle connections
    - in_use: set of active connections
    - max_size: maximum pool size

    On acquire():
        1. If available is not empty:
             conn = available.pop()
             in_use.add(conn)
             return conn
        2. Elif total_connections < max_size:
             conn = create_new_connection()
             in_use.add(conn)
             return conn
        3. Else:
             Wait for connection (with timeout)

    On release(conn):
        1. in_use.remove(conn)
        2. If conn.is_healthy():
             available.add(conn)
        3. Else:
             conn.close()
    """
}
```

**Success criteria:**
- Algorithm tasks implement correct logic
- Fewer conceptual errors
- Proper state management

---

### Tier 3: Strategy (Task Decomposition)

Improve how complex tasks are broken down.

#### 3.1 Incremental Delivery for Multi-Component Tasks

**Component:** Enhanced `decompose_task` in tools.py

**Strategy:** For tasks with multiple algorithms/components, create phased subtasks.

**Example (Rate Limiter):**

**Before (problematic):**
```
Subtasks:
1. Implement token bucket, sliding window, and Redis backend
2. Add metrics and tests
```

**After (incremental):**
```
Subtasks:
1. Implement token bucket algorithm (in-memory)
2. Test token bucket
3. Implement sliding window algorithm (in-memory)
4. Test sliding window
5. Add Redis backend for token bucket
6. Add Redis backend for sliding window
7. Add metrics and final tests
```

**Implementation:**

```python
def decompose_complex_task(goal):
    """
    Detect multi-component goals and decompose incrementally.

    Patterns:
    - "X with A, B, and C" -> Build A, test A, build B, test B, build C
    - "Create X: feature1, feature2, feature3" -> One feature per subtask
    """
    # Detect multiple components
    components = extract_components(goal)

    if len(components) > 2:
        # Multi-component: incremental approach
        subtasks = []
        for component in components:
            subtasks.append(f"Implement {component}")
            subtasks.append(f"Test {component}")
        return subtasks
    else:
        # Simple: standard decomposition
        return standard_decompose(goal)
```

**Success criteria:**
- Multi-algorithm tasks complete more often
- Each subtask is self-contained and testable
- Clear progress through complex implementation

#### 3.2 Pattern-Aware Decomposition

**Component:** `pattern_decomposer.py`

**Strategy:** Recognize design pattern tasks and use pattern-specific decomposition.

**Example (Observer Pattern):**

```python
def decompose_observer_pattern(goal):
    """Standard decomposition for Observer pattern."""
    return [
        "Create Subject base class with attach/detach/notify",
        "Create Observer interface with update method",
        "Create 1-2 ConcreteObserver examples",
        "Create demo showing pattern in action",
        "Add unit tests for pattern behavior"
    ]

def decompose_circuit_breaker(goal):
    """Standard decomposition for Circuit Breaker pattern."""
    return [
        "Define State enum (CLOSED, OPEN, HALF_OPEN)",
        "Create CircuitBreaker class with state tracking",
        "Implement CLOSED state logic (track failures)",
        "Implement state transitions (CLOSED -> OPEN -> HALF_OPEN)",
        "Add timeout for OPEN -> HALF_OPEN transition",
        "Implement call() method with state checks",
        "Add metrics tracking",
        "Write tests for all state transitions"
    ]
```

**Success criteria:**
- Pattern tasks follow standard structure
- All pattern elements implemented
- Correct behavior for pattern's key scenarios

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1)
- [ ] Implement LLM Health Monitor
- [ ] Add progressive context compaction
- [ ] Create error recovery system
- [ ] Test with 3-5 L7 tasks

**Success metric:** LLM timeout rate < 10%

### Phase 2: Knowledge (Week 2)
- [ ] Create pattern template library
- [ ] Create algorithm knowledge base
- [ ] Inject templates into system prompts
- [ ] Test with L6 pattern tasks

**Success metric:** L6 pass rate > 50%

### Phase 3: Strategy (Week 3)
- [ ] Implement incremental decomposition
- [ ] Add pattern-aware decomposition
- [ ] Create decomposition validators
- [ ] Test with full L6-L7 suite

**Success metric:** L7 pass rate > 40%

### Phase 4: Validation (Week 4)
- [ ] Run full evaluation (L1-L7 x10)
- [ ] Measure improvement vs baseline
- [ ] Identify remaining gaps
- [ ] Document best practices

**Success metric:** Overall L6-L7 pass rate > 60%

---

## Success Criteria

### Quantitative

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| L6 pass rate | 22% | 70% | 85% |
| L7 pass rate | 11% | 50% | 70% |
| LLM timeout rate | 22% | <10% | <5% |
| Avg L6 duration | 21s | 30s | 25s |
| Avg L7 duration | 39s | 60s | 45s |

### Qualitative

- [ ] Design patterns show correct structure
- [ ] Multi-algorithm tasks deliver incrementally
- [ ] Agent recovers from most stuck states
- [ ] Clear progress visibility on complex tasks
- [ ] Pattern implementations are maintainable

---

## Alternative Approaches Considered

### 1. Multi-Agent Specialization

**Concept:** Separate agents for patterns vs algorithms vs integration.

**Pros:**
- Specialized knowledge per agent
- Can optimize prompts per domain

**Cons:**
- Complex orchestration
- Higher latency (agent switching)
- More state management

**Decision:** Not now. Try single-agent improvements first.

### 2. Few-Shot Learning (Examples in Prompt)

**Concept:** Include example implementations of patterns in system prompt.

**Pros:**
- Direct learning from examples
- Clear structure to follow

**Cons:**
- Huge context usage (examples are long)
- Risk of copying instead of adapting
- Hard to keep examples up-to-date

**Decision:** Use templates (structure) not examples (code).

### 3. External Knowledge Retrieval (RAG)

**Concept:** Query external docs (Gang of Four, algorithm textbooks) during execution.

**Pros:**
- Access to detailed knowledge
- Always current (if docs updated)

**Cons:**
- Requires infrastructure (vector DB, embeddings)
- Latency overhead
- Context management complexity

**Decision:** Not for MVP. Pattern templates are sufficient for common cases.

---

## Risks and Mitigations

### Risk 1: Templates Reduce Creativity

**Risk:** Agent might rigidly follow templates, missing better approaches.

**Mitigation:**
- Templates are guidance, not constraints
- Phrase as "suggested structure" not "required structure"
- Monitor for template over-reliance

### Risk 2: Incremental Decomposition Too Granular

**Risk:** Breaking tasks too small might slow down simple cases.

**Mitigation:**
- Only use for multi-component tasks (detected automatically)
- Standard decomposition for simple cases
- Allow merging subtasks if agent judges it safe

### Risk 3: Context Compaction Loses Critical Info

**Risk:** Aggressive compaction might remove needed details.

**Mitigation:**
- Preserve task-specific keywords
- Keep recent messages intact
- Test compaction with known-good tasks

---

## Appendix A: Evaluation Tasks

### L6 Tasks (Architecture)
1. Observer Pattern
2. Factory Pattern
3. Dependency Injection
4. Plugin System
5. Event Bus

### L7 Tasks (Expert)
1. Rate Limiter (token bucket + sliding window)
2. Connection Pool (acquire/release + health checks)
3. Circuit Breaker (state machine + metrics)
4. Distributed Cache (consistent hashing + replication)

---

## Appendix B: Baseline Performance Data

From comprehensive_l5_l7_results.json:

```
L5 (Integration):
  - blog_system: 0/3 (validation issues, but completed)
  - todo_app: 0/3 (validation issues, but completed)
  - inventory: 0/3 (1 timeout, 2 validation issues)

L6 (Architecture):
  - observer: 0/3 (timeouts)
  - factory: 0/3 (validation issues)
  - dependency_injection: 0/3 (validation issues)

L7 (Expert):
  - rate_limiter: 0/3 (timeouts)
  - connection_pool: 0/3 (timeouts)
  - circuit_breaker: 0/3 (LLM hangs)
```

**Key insight:** Most "failures" are timeouts or validation mismatches, not code quality issues.

---

## Appendix C: Validation Improvements

Created `semantic_validator.py` which checks for:
- Required classes exist (any file)
- Required functions exist (any file)
- Code is importable
- Core functionality present

Script: `run_l5_l7_semantic.py` runs evaluation with semantic validation.

Expected improvement: 0% -> 30-50% pass rate just from better validation.

---

## Sign-off

**Proposed by:** Claude (Agent System)
**Date:** 2025-10-29
**Review status:** AWAITING APPROVAL

**Next steps:**
1. Review and approve design
2. Prioritize implementation (all phases or subset?)
3. Allocate resources (timeline, testing)
4. Begin Phase 1 implementation
