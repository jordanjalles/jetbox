"""Simple asynchronous event bus implementation.

This module provides an :class:`EventBus` class that allows components to
publish events and subscribe to them.  Subscriptions can specify a filter
function that receives the event payload and returns ``True`` if the
handler should be invoked.

The bus is fully asynchronous: publishing an event will schedule all
matching handlers to run concurrently using :func:`asyncio.create_task`.
Handlers may be either normal callables or ``async`` functions.

Example usage::

    import asyncio
    from event_bus import EventBus

    bus = EventBus()

    async def on_user_created(user):
        print("User created:", user)

    bus.subscribe("user.created", on_user_created)

    async def main():
        await bus.publish("user.created", {"id": 1, "name": "Alice"})

    asyncio.run(main())

"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

# Type alias for event handlers.  They can be sync or async callables.
EventHandler = Callable[[Any], Awaitable[None] | None]

# Type alias for filter functions.
EventFilter = Callable[[Any], bool]


class Subscription:
    """Represents a single subscription.

    Attributes
    ----------
    handler:
        The callable to invoke when an event matches.
    filter:
        Optional filter function that receives the event payload and
        returns ``True`` if the handler should be called.
    """

    def __init__(self, handler: EventHandler, filter: Optional[EventFilter] = None):
        self.handler = handler
        self.filter = filter

    def matches(self, payload: Any) -> bool:
        if self.filter is None:
            return True
        try:
            return bool(self.filter(payload))
        except Exception as exc:  # pragma: no cover - defensive
            # If filter raises, we treat it as non-match.
            return False


class EventBus:
    """Asynchronous publish/subscribe event bus.

    The bus keeps a mapping from event type (string) to a list of
    :class:`Subscription` objects.  Publishing an event will schedule all
    matching handlers to run concurrently.
    """

    def __init__(self) -> None:
        # Mapping: event_type -> list of subscriptions
        self._subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        filter: Optional[EventFilter] = None,
    ) -> None:
        """Register a handler for *event_type*.

        Parameters
        ----------
        event_type:
            The name of the event to subscribe to.
        handler:
            Callable that receives the event payload.  It may be a normal
            function or an ``async`` function.
        filter:
            Optional callable that receives the payload and returns
            ``True`` if the handler should be invoked.
        """
        sub = Subscription(handler, filter)
        # Use lock to protect concurrent modifications.
        asyncio.get_event_loop().create_task(self._add_subscription(event_type, sub))

    async def _add_subscription(self, event_type: str, sub: Subscription) -> None:
        async with self._lock:
            self._subscribers[event_type].append(sub)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler.

        If the handler is not found, the call is ignored.
        """
        asyncio.get_event_loop().create_task(self._remove_subscription(event_type, handler))

    async def _remove_subscription(self, event_type: str, handler: EventHandler) -> None:
        async with self._lock:
            subs = self._subscribers.get(event_type, [])
            self._subscribers[event_type] = [s for s in subs if s.handler is not handler]

    async def publish(self, event_type: str, payload: Any) -> None:
        """Publish an event.

        All matching handlers are scheduled to run concurrently.  The
        method returns immediately after scheduling.
        """
        async with self._lock:
            subs = list(self._subscribers.get(event_type, []))
        for sub in subs:
            if sub.matches(payload):
                # Schedule handler; capture exceptions to avoid task cancellation.
                asyncio.create_task(self._run_handler(sub.handler, payload))

    async def _run_handler(self, handler: EventHandler, payload: Any) -> None:
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(payload)
            else:
                # Run sync handler in default executor to avoid blocking event loop.
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, handler, payload)
        except Exception as exc:  # pragma: no cover - log for debugging
            # In production, replace with proper logging.
            print(f"Event handler raised exception: {exc}")

    async def clear(self) -> None:
        """Remove all subscriptions.  Useful for tests."""
        async with self._lock:
            self._subscribers.clear()

# Simple helper to create a global bus instance if desired.
_global_bus: Optional[EventBus] = None


def get_global_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus

# End of event_bus.py
