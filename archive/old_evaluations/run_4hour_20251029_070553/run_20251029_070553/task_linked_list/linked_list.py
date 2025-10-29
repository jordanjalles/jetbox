"""Linked list implementation.

This module provides a simple singly linked list with basic operations:
- append(value): add a new node with the given value to the end of the list.
- remove(value): remove the first node that contains the given value.
- contains(value): return True if a node with the given value exists.
- to_list(): return a Python list of all node values in order.

The implementation uses a private Node class.
"""

from __future__ import annotations

class _Node:
    """Internal node class for LinkedList.

    Attributes
    ----------
    value : Any
        The value stored in the node.
    next : _Node | None
        Reference to the next node in the list.
    """

    __slots__ = ("value", "next")

    def __init__(self, value: object, next: _Node | None = None):
        self.value = value
        self.next = next

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"_Node({self.value!r})"


class LinkedList:
    """A simple singly linked list.

    The list keeps a reference to the head node and the tail node for
    efficient appends. It also tracks the number of elements.
    """

    def __init__(self) -> None:
        self._head: _Node | None = None
        self._tail: _Node | None = None
        self._size: int = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    def append(self, value: object) -> None:
        """Append a new node with *value* to the end of the list.

        Parameters
        ----------
        value : object
            The value to store.
        """
        new_node = _Node(value)
        if self._head is None:
            # Empty list
            self._head = self._tail = new_node
        else:
            assert self._tail is not None  # for type checkers
            self._tail.next = new_node
            self._tail = new_node
        self._size += 1

    def remove(self, value: object) -> bool:
        """Remove the first node containing *value*.

        Returns
        -------
        bool
            ``True`` if a node was removed, ``False`` otherwise.
        """
        prev: _Node | None = None
        current = self._head
        while current is not None:
            if current.value == value:
                if prev is None:
                    # Removing head
                    self._head = current.next
                    if self._head is None:
                        # List became empty
                        self._tail = None
                else:
                    prev.next = current.next
                    if current.next is None:
                        # Removed tail
                        self._tail = prev
                self._size -= 1
                return True
            prev = current
            current = current.next
        return False

    def contains(self, value: object) -> bool:
        """Return ``True`` if *value* is present in the list."""
        current = self._head
        while current is not None:
            if current.value == value:
                return True
            current = current.next
        return False

    def to_list(self) -> list[object]:
        """Return a Python list of all values in the linked list."""
        result: list[object] = []
        current = self._head
        while current is not None:
            result.append(current.value)
            current = current.next
        return result

    def __iter__(self):  # pragma: no cover - trivial
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"LinkedList({self.to_list()!r})"
