# Timeout Fix V2 - Correct Design

## The Fundamental Mistake in V1

**What we did wrong:**
- Timed out the entire `decompose_goal()` LLM call at 120 seconds
- This cuts off **legitimate thinking** for complex tasks
- The timeout should detect **Ollama being dead/hung**, not **Ollama thinking hard**

## The Right Approach

### What We Should Timeout

**BAD** (Current V1):
```python
# Timeout the entire LLM call
resp = chat_with_timeout(model=MODEL, messages=[...], timeout_seconds=120)
```
Problem: Kills the call even when Ollama is actively processing.

**GOOD** (V2):
```python
# Only timeout if Ollama stops making progress (no response activity)
resp = chat_with_activity_timeout(model=MODEL, messages=[...],
                                  inactivity_timeout=30)
```
Benefit: Allows long thinking, but detects when Ollama is actually hung.

## How HTTP Streaming Works with Ollama

The Ollama Python client supports **streaming** responses:

```python
from ollama import chat

# Streaming mode - tokens arrive incrementally
for chunk in chat(model=MODEL, messages=[...], stream=True):
    # chunk arrives every time Ollama generates tokens
    # If Ollama is alive, chunks arrive continuously
    # If Ollama is hung, no chunks arrive
```

**Key insight**: If we haven't received a chunk in 30 seconds, Ollama is probably dead/hung.

## Proposed Fix

### Option 1: Inactivity Timeout (RECOMMENDED)

```python
def chat_with_inactivity_timeout(model, messages, options, inactivity_timeout=30):
    """
    Call Ollama with streaming, timeout only if no activity for N seconds.

    This allows complex tasks to take as long as they need,
    but detects when Ollama has actually stopped responding.
    """
    import time
    from queue import Queue, Empty
    from threading import Thread

    result_queue = Queue()
    last_activity = time.time()

    def stream_chat():
        try:
            full_response = ""
            for chunk in chat(model=model, messages=messages,
                            options=options, stream=True):
                result_queue.put(('chunk', chunk))
                full_response += chunk.get('message', {}).get('content', '')

            result_queue.put(('done', full_response))
        except Exception as e:
            result_queue.put(('error', e))

    # Start streaming in background
    thread = Thread(target=stream_chat, daemon=True)
    thread.start()

    # Monitor for inactivity
    while True:
        try:
            msg_type, data = result_queue.get(timeout=inactivity_timeout)

            if msg_type == 'chunk':
                last_activity = time.time()
                continue
            elif msg_type == 'done':
                return {'message': {'content': data}}
            elif msg_type == 'error':
                raise data

        except Empty:
            # No activity for inactivity_timeout seconds
            raise TimeoutError(
                f"No response from Ollama for {inactivity_timeout}s - likely hung"
            )
```

### Option 2: Remove Timeout Entirely

Just accept that if Ollama is dead, the test will timeout naturally at the test framework level (240-360s). This is actually fine - the test framework already handles this.

### Option 3: Much Longer Timeout

Change from 120s → 600s (10 minutes). This allows complex decomposition while still eventually catching true hangs.

## Comparison

| Approach | Allows Complex Tasks | Detects Hung Ollama | Implementation |
|----------|---------------------|---------------------|----------------|
| **V1 (Current)** | ❌ NO - kills at 120s | ✅ YES | Simple |
| **V2 (Inactivity)** | ✅ YES - unlimited | ✅ YES - within 30s | Complex |
| **Remove Timeout** | ✅ YES - unlimited | ⚠️ Eventually (test timeout) | Trivial |
| **Longer Timeout** | ⚠️ Maybe (depends on task) | ✅ YES - within 600s | Trivial |

## Recommended Solution

**Use inactivity timeout with streaming**: This is the "right" solution because:

1. ✅ **Allows complex tasks** - No arbitrary time limit on thinking
2. ✅ **Detects hung Ollama fast** - 30s of no activity = problem
3. ✅ **Distinguishes "thinking" from "hung"** - Activity = healthy, silence = problem

## Implementation

Replace `chat_with_timeout()` with `chat_with_inactivity_timeout()` in agent.py.

The key difference:
- **V1**: "You have 120 seconds total to finish"
- **V2**: "You can take as long as you want, but if you go silent for 30 seconds, you're hung"

This matches how HTTP clients work - they don't timeout based on total duration, they timeout based on **inactivity**.
