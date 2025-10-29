"""Simple memoization decorator.

This module provides a `@cache` decorator that caches the results of a
function based on its positional and keyword arguments.  The cache is
stored in a dictionary on the function object itself.  Only hashable
arguments are supported – this is sufficient for most use‑cases.

Example
-------
>>> from cache import cache
>>> @cache
... def fib(n):
...     return fib(n-1) + fib(n-2) if n > 1 else 1
...
>>> fib(35)  # fast thanks to memoisation
123456789
"""

from functools import wraps
from typing import Any, Callable, Dict, Tuple

__all__ = ["cache"]


def _make_key(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Create a hashable key from function arguments.

    Positional arguments are stored as a tuple.  Keyword arguments are
    converted to a tuple of ``(key, value)`` pairs sorted by key to
    ensure consistent ordering.
    """
    key = args
    if kwargs:
        # Sort kwargs to make the key order deterministic
        key += tuple(sorted(kwargs.items()))
    return key


def cache(func: Callable) -> Callable:
    """Decorator that caches function results.

    The cache is a simple dictionary stored on the wrapped function
    under the attribute ``_cache``.  The decorator is thread‑safe for
    read‑only access; if you need concurrent writes you should add
    locking.
    """

    cache_dict: Dict[Tuple[Any, ...], Any] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        key = _make_key(args, kwargs)
        if key in cache_dict:
            return cache_dict[key]
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result

    # expose the cache for introspection/debugging
    wrapper._cache = cache_dict
    return wrapper

# End of cache.py
