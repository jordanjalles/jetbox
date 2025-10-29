"""Utility decorators for caching function results.

This module provides a simple @memoize decorator that caches the results of
function calls based on their arguments. It also exposes a helper function
`clear_cache` that can be used to clear the cache for a specific function.

Example usage:

>>> from decorators import memoize, clear_cache
>>> @memoize
... def fib(n):
...     if n < 2:
...         return n
...     return fib(n-1) + fib(n-2)

>>> fib(10)
55
>>> clear_cache(fib)
>>> fib(10)  # recomputed
55
"""

from functools import wraps
from typing import Any, Callable, Dict, Tuple

# Global registry of caches for functions
_caches: Dict[Callable, Dict[Tuple[Any, ...], Any]] = {}


def _make_key(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Create a hashable key from function arguments.

    Positional arguments are used as-is. Keyword arguments are sorted by key
    and converted to a tuple of (key, value) pairs.
    """
    key = args
    if kwargs:
        # Sort kwargs to ensure consistent ordering
        key += tuple(sorted(kwargs.items()))
    return key


def memoize(func: Callable) -> Callable:
    """Decorator that caches function results.

    The cache is stored in a dictionary keyed by the function's arguments.
    The decorated function receives an additional attribute ``cache`` which
    holds the cache dictionary, and a method ``cache_clear`` to clear it.
    """
    cache: Dict[Tuple[Any, ...], Any] = {}
    _caches[func] = cache

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = _make_key(args, kwargs)
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    # expose cache and clear method
    wrapper.cache = cache
    def cache_clear():
        cache.clear()
    wrapper.cache_clear = cache_clear

    return wrapper


def clear_cache(func: Callable) -> None:
    """Clear the cache for a memoized function.

    Raises ``KeyError`` if the function has not been memoized.
    """
    if func not in _caches:
        raise KeyError("Function not memoized")
    _caches[func].clear()

# End of decorators.py
