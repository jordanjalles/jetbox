#!/usr/bin/env python3
"""LLM warm-up and keep-alive implementation for reducing first-call latency."""
import time
import threading
from typing import Any, Optional
from ollama import chat

class LLMWarmer:
    """
    Keep LLM model warm to reduce first-call latency.

    Strategy:
    1. Send periodic keep-alive requests (minimal tokens)
    2. Pre-warm model on agent startup
    3. Maintain model in memory between tool calls
    """

    def __init__(self, model: str = "gpt-oss:20b"):
        self.model = model
        self.last_call_time = 0
        self.keep_alive_interval = 30  # Send keep-alive every 30s
        self.warmed_up = False
        self._stop_keepalive = False
        self._keepalive_thread: Optional[threading.Thread] = None

    def warmup(self) -> dict[str, float]:
        """
        Warm up the model with a minimal request.
        Returns timing metrics.
        """
        print(f"🔥 Warming up {self.model}...")

        # Measure cold start
        cold_start = time.perf_counter()
        try:
            resp = chat(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 1, "temperature": 0.1},
            )
            cold_time = (time.perf_counter() - cold_start) * 1000
            print(f"   Cold start: {cold_time:.0f}ms")
        except Exception as e:
            print(f"   Error during warm-up: {e}")
            return {"cold_start": 0, "warm_start": 0}

        # Measure warm start (should be faster)
        time.sleep(0.5)
        warm_start = time.perf_counter()
        try:
            resp = chat(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 1, "temperature": 0.1},
            )
            warm_time = (time.perf_counter() - warm_start) * 1000
            print(f"   Warm start: {warm_time:.0f}ms")
            print(f"   Improvement: {cold_time - warm_time:.0f}ms ({((cold_time - warm_time) / cold_time * 100):.1f}%)")
        except Exception as e:
            print(f"   Error during warm check: {e}")
            warm_time = cold_time

        self.warmed_up = True
        self.last_call_time = time.time()

        return {
            "cold_start_ms": cold_time,
            "warm_start_ms": warm_time,
            "improvement_ms": cold_time - warm_time,
            "improvement_pct": ((cold_time - warm_time) / cold_time * 100) if cold_time > 0 else 0,
        }

    def keepalive_ping(self) -> None:
        """Send a minimal keep-alive request to keep model loaded."""
        try:
            chat(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 1, "temperature": 0},
            )
            self.last_call_time = time.time()
        except Exception:
            pass  # Silently ignore keep-alive failures

    def start_keepalive_thread(self) -> None:
        """Start background thread to send periodic keep-alive requests."""
        if self._keepalive_thread is not None:
            return  # Already running

        def keepalive_loop():
            print(f"🔄 Keep-alive thread started (interval: {self.keep_alive_interval}s)")
            while not self._stop_keepalive:
                time.sleep(self.keep_alive_interval)
                if not self._stop_keepalive:
                    idle_time = time.time() - self.last_call_time
                    if idle_time >= self.keep_alive_interval:
                        print(f"   Keep-alive ping (idle: {idle_time:.0f}s)")
                        self.keepalive_ping()

        self._stop_keepalive = False
        self._keepalive_thread = threading.Thread(target=keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive_thread(self) -> None:
        """Stop the background keep-alive thread."""
        if self._keepalive_thread is not None:
            self._stop_keepalive = True
            self._keepalive_thread.join(timeout=2)
            self._keepalive_thread = None
            print("🛑 Keep-alive thread stopped")

    def record_call(self) -> None:
        """Record that a call was made (resets idle timer)."""
        self.last_call_time = time.time()


def test_warmup_impact():
    """Test the impact of warm-up on latency."""
    print("="*60)
    print("LLM WARM-UP IMPACT TEST")
    print("="*60)

    warmer = LLMWarmer("gpt-oss:20b")

    # Test 1: Cold start vs warm start
    print("\n1. Testing cold vs warm start...")
    metrics = warmer.warmup()

    # Test 2: Latency after idle period
    print("\n2. Testing latency after idle period...")
    print("   Waiting 5 seconds...")
    time.sleep(5)

    idle_start = time.perf_counter()
    chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "test"}],
        options={"num_predict": 1},
    )
    idle_time = (time.perf_counter() - idle_start) * 1000
    print(f"   After 5s idle: {idle_time:.0f}ms")

    # Test 3: Immediate follow-up
    print("\n3. Testing immediate follow-up call...")
    quick_start = time.perf_counter()
    chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "test"}],
        options={"num_predict": 1},
    )
    quick_time = (time.perf_counter() - quick_start) * 1000
    print(f"   Immediate: {quick_time:.0f}ms")
    print(f"   Difference: {idle_time - quick_time:.0f}ms")

    # Test 4: Keep-alive effectiveness
    print("\n4. Testing keep-alive effectiveness...")
    warmer.start_keepalive_thread()

    print("   Waiting 35 seconds (should trigger keep-alive)...")
    time.sleep(35)

    after_keepalive = time.perf_counter()
    chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "test"}],
        options={"num_predict": 1},
    )
    keepalive_time = (time.perf_counter() - after_keepalive) * 1000
    print(f"   After keep-alive: {keepalive_time:.0f}ms")

    warmer.stop_keepalive_thread()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Cold start:        {metrics['cold_start_ms']:.0f}ms")
    print(f"Warm start:        {metrics['warm_start_ms']:.0f}ms")
    print(f"After 5s idle:     {idle_time:.0f}ms")
    print(f"Immediate:         {quick_time:.0f}ms")
    print(f"After keep-alive:  {keepalive_time:.0f}ms")
    print(f"\nWarm-up savings:   {metrics['improvement_ms']:.0f}ms ({metrics['improvement_pct']:.1f}%)")

    return metrics


if __name__ == "__main__":
    test_warmup_impact()
