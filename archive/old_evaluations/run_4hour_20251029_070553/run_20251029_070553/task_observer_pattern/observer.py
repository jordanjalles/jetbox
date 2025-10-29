"""Implementation of a simple observer pattern.

This module defines:

* :class:`Subject` – maintains a list of observers and notifies them of events.
* :class:`Observer` – base class for observers that can subscribe to a subject.
* :class:`Event` – simple event container used to pass data to observers.

The implementation is intentionally lightweight and uses only the Python standard
library.  It is suitable for educational purposes and can be extended for more
complex use‑cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Any


class Observer(Protocol):
    """Protocol for observers.

    An observer must implement a :meth:`notify` method that receives an
    :class:`Event` instance.
    """

    def notify(self, event: "Event") -> None: ...


@dataclass
class Event:
    """Simple event container.

    Attributes
    ----------
    name:
        Name of the event.
    payload:
        Arbitrary data associated with the event.
    """

    name: str
    payload: Any = None


class Subject:
    """Subject that manages observers and dispatches events.

    Observers can subscribe to specific event names.  When an event is
    emitted via :meth:`notify`, all observers registered for that event are
    called.
    """

    def __init__(self) -> None:
        # Mapping from event name to list of observers
        self._observers: Dict[str, List[Observer]] = {}

    def subscribe(self, event_name: str, observer: Observer) -> None:
        """Register *observer* to receive notifications for *event_name*.

        Parameters
        ----------
        event_name:
            The name of the event to subscribe to.
        observer:
            The observer instance.
        """
        self._observers.setdefault(event_name, []).append(observer)

    def unsubscribe(self, event_name: str, observer: Observer) -> None:
        """Remove *observer* from the list for *event_name*.

        If the observer is not registered, the call is ignored.
        """
        observers = self._observers.get(event_name)
        if observers and observer in observers:
            observers.remove(observer)
            if not observers:
                # Clean up empty lists to keep the dict tidy
                del self._observers[event_name]

    def notify(self, event: Event) -> None:
        """Notify all observers subscribed to *event.name*.

        The event is passed unchanged to each observer's :meth:`notify`
        method.
        """
        for observer in list(self._observers.get(event.name, [])):
            observer.notify(event)

    def clear(self) -> None:
        """Remove all observers from all events."""
        self._observers.clear()


# Example usage (uncomment to test manually)
# if __name__ == "__main__":
#     class PrintObserver:
#         def notify(self, event: Event) -> None:
#             print(f"Received {event.name} with payload: {event.payload}")
#
#     subject = Subject()
#     observer = PrintObserver()
#     subject.subscribe("data", observer)
#     subject.notify(Event(name="data", payload={"key": "value"}))
#     subject.unsubscribe("data", observer)
#     subject.notify(Event(name="data", payload={"key": "value"}))

"""
END OF FILE
"""
