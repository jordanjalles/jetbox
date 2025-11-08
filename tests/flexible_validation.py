"""
Flexible validation helper for evaluations.

Tests functionality regardless of file structure.
"""
import sys
import glob
from pathlib import Path


def validate_todo_app(workspace: Path) -> tuple[bool, str]:
    """Validate todo app has required functionality."""
    try:
        # Find any Python file
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        # Add workspace to path
        sys.path.insert(0, str(workspace))

        # Try to find TodoManager in any file
        import importlib.util
        todo_manager = None
        for py_file in py_files:
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'TodoManager'):
                        todo_manager = module.TodoManager
                        break
            except Exception as e:
                continue

        if not todo_manager:
            return False, f"TodoManager class not found in {[f.name for f in py_files]}"

        # Test functionality
        tm = todo_manager()
        tm.add_todo('Task 1', 'work')
        todos = tm.get_todos() if hasattr(tm, 'get_todos') else tm.todos

        if len(todos) != 1:
            return False, f"Expected 1 todo, got {len(todos)}"

        return True, "TodoManager works correctly"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        # Clean up sys.path
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_blog_system(workspace: Path) -> tuple[bool, str]:
    """Validate blog system has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find BlogManager
        blog_manager = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'BlogManager'):
                    blog_manager = module.BlogManager
                    break
            except Exception:
                continue

        if not blog_manager:
            return False, f"BlogManager not found"

        # Test functionality
        bm = blog_manager()
        bm.create_post('Title', 'Content')
        posts = bm.get_posts() if hasattr(bm, 'get_posts') else bm.posts

        if len(posts) != 1:
            return False, f"Expected 1 post, got {len(posts)}"

        return True, "BlogManager works correctly"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_inventory_system(workspace: Path) -> tuple[bool, str]:
    """Validate inventory system has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find Inventory
        inventory = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'Inventory'):
                    inventory = module.Inventory
                    break
            except Exception:
                continue

        if not inventory:
            return False, f"Inventory class not found"

        # Test functionality
        inv = inventory()
        inv.add_product('Widget', 10, 2.5)
        products = inv.get_products() if hasattr(inv, 'get_products') else inv.products

        if len(products) != 1:
            return False, f"Expected 1 product, got {len(products)}"

        return True, "Inventory works correctly"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_url_shortener(workspace: Path) -> tuple[bool, str]:
    """Validate URL shortener has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find URLShortener
        url_shortener = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'URLShortener'):
                    url_shortener = module.URLShortener
                    break
            except Exception:
                continue

        if not url_shortener:
            return False, f"URLShortener class not found"

        # Test functionality
        us = url_shortener()
        short = us.shorten('https://example.com')

        if not short or len(short) >= 20:
            return False, f"Short URL not generated correctly: {short}"

        return True, "URLShortener works correctly"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_email_validator(workspace: Path) -> tuple[bool, str]:
    """Validate email validator has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find EmailValidator
        email_validator = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'EmailValidator'):
                    email_validator = module.EmailValidator
                    break
            except Exception:
                continue

        if not email_validator:
            return False, f"EmailValidator class not found"

        # Test functionality
        ev = email_validator()
        result = ev.validate('test@example.com')

        if not result:
            return False, f"Email validation failed for valid email"

        return True, "EmailValidator works correctly"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


# ============================================================================
# L6 Validators
# ============================================================================

def validate_observer_pattern(workspace: Path) -> tuple[bool, str]:
    """Validate observer pattern has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find Subject and Observer
        subject_cls = None
        observer_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'Subject'):
                    subject_cls = module.Subject
                if hasattr(module, 'Observer'):
                    observer_cls = module.Observer
                if subject_cls and observer_cls:
                    break
            except Exception:
                continue

        if not subject_cls:
            return False, "Subject class not found"
        if not observer_cls:
            return False, "Observer class not found"

        # Test basic functionality
        subject = subject_cls()
        if not (hasattr(subject, 'attach') or hasattr(subject, 'subscribe')):
            return False, "Subject missing attach/subscribe method"
        if not (hasattr(subject, 'notify') or hasattr(subject, 'notify_all')):
            return False, "Subject missing notify method"

        return True, "Observer pattern classes found and validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_factory_pattern(workspace: Path) -> tuple[bool, str]:
    """Validate factory pattern has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find Factory
        factory_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'Factory'):
                    factory_cls = module.Factory
                    break
            except Exception:
                continue

        if not factory_cls:
            return False, "Factory class not found"

        # Test basic functionality
        factory = factory_cls()
        if not (hasattr(factory, 'create') or hasattr(factory, 'create_product')):
            return False, "Factory missing create method"

        return True, "Factory pattern validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_dependency_injection(workspace: Path) -> tuple[bool, str]:
    """Validate DI container has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find Container
        container_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'Container') or hasattr(module, 'DIContainer'):
                    container_cls = getattr(module, 'Container', None) or module.DIContainer
                    break
            except Exception:
                continue

        if not container_cls:
            return False, "Container class not found"

        # Test basic functionality
        container = container_cls()
        if not hasattr(container, 'register'):
            return False, "Container missing register method"
        if not hasattr(container, 'resolve'):
            return False, "Container missing resolve method"

        return True, "DI container validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_plugin_system(workspace: Path) -> tuple[bool, str]:
    """Validate plugin system has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find PluginManager
        plugin_manager_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'PluginManager'):
                    plugin_manager_cls = module.PluginManager
                    break
            except Exception:
                continue

        if not plugin_manager_cls:
            return False, "PluginManager class not found"

        # Test basic functionality
        pm = plugin_manager_cls()
        if not (hasattr(pm, 'load') or hasattr(pm, 'load_plugin')):
            return False, "PluginManager missing load method"
        if not (hasattr(pm, 'register') or hasattr(pm, 'register_plugin')):
            return False, "PluginManager missing register method"

        return True, "Plugin system validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_event_bus(workspace: Path) -> tuple[bool, str]:
    """Validate event bus has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find EventBus
        event_bus_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'EventBus'):
                    event_bus_cls = module.EventBus
                    break
            except Exception:
                continue

        if not event_bus_cls:
            return False, "EventBus class not found"

        # Test basic functionality
        eb = event_bus_cls()
        if not (hasattr(eb, 'publish') or hasattr(eb, 'emit')):
            return False, "EventBus missing publish method"
        if not (hasattr(eb, 'subscribe') or hasattr(eb, 'on')):
            return False, "EventBus missing subscribe method"

        return True, "Event bus validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


# ============================================================================
# L7 Validators
# ============================================================================

def validate_rate_limiter(workspace: Path) -> tuple[bool, str]:
    """Validate rate limiter has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find RateLimiter
        rate_limiter_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'RateLimiter'):
                    rate_limiter_cls = module.RateLimiter
                    break
            except Exception:
                continue

        if not rate_limiter_cls:
            return False, "RateLimiter class not found"

        # Test basic functionality
        rl = rate_limiter_cls()
        if not (hasattr(rl, 'allow') or hasattr(rl, 'is_allowed') or hasattr(rl, 'check')):
            return False, "RateLimiter missing allow/check method"

        return True, "Rate limiter validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_connection_pool(workspace: Path) -> tuple[bool, str]:
    """Validate connection pool has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find ConnectionPool or Pool
        pool_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'ConnectionPool'):
                    pool_cls = module.ConnectionPool
                    break
                elif hasattr(module, 'Pool'):
                    pool_cls = module.Pool
                    break
            except Exception:
                continue

        if not pool_cls:
            return False, "ConnectionPool/Pool class not found"

        # Test basic functionality
        if not (hasattr(pool_cls, 'acquire') or hasattr(pool_cls, 'get')):
            return False, "Pool missing acquire/get method"
        if not (hasattr(pool_cls, 'release') or hasattr(pool_cls, 'put')):
            return False, "Pool missing release/put method"

        return True, "Connection pool validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_circuit_breaker(workspace: Path) -> tuple[bool, str]:
    """Validate circuit breaker has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find CircuitBreaker
        circuit_breaker_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'CircuitBreaker'):
                    circuit_breaker_cls = module.CircuitBreaker
                    break
            except Exception:
                continue

        if not circuit_breaker_cls:
            return False, "CircuitBreaker class not found"

        # Test basic functionality
        cb = circuit_breaker_cls()
        if not (hasattr(cb, 'call') or hasattr(cb, 'execute')):
            return False, "CircuitBreaker missing call/execute method"

        return True, "Circuit breaker validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


def validate_distributed_cache(workspace: Path) -> tuple[bool, str]:
    """Validate distributed cache has required functionality."""
    try:
        py_files = list(workspace.glob("*.py"))
        if not py_files:
            return False, "No Python files found"

        sys.path.insert(0, str(workspace))

        # Find Cache or DistributedCache
        cache_cls = None
        for py_file in py_files:
            module_name = py_file.stem
            try:
                module = __import__(module_name)
                if hasattr(module, 'DistributedCache'):
                    cache_cls = module.DistributedCache
                    break
                elif hasattr(module, 'Cache'):
                    cache_cls = module.Cache
                    break
            except Exception:
                continue

        if not cache_cls:
            return False, "Cache class not found"

        # Test basic functionality
        cache = cache_cls()
        if not (hasattr(cache, 'get') or hasattr(cache, 'read')):
            return False, "Cache missing get method"
        if not (hasattr(cache, 'set') or hasattr(cache, 'write')):
            return False, "Cache missing set method"

        return True, "Distributed cache validated"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))


# ============================================================================
# Validator Registry
# ============================================================================

VALIDATORS = {
    # L5 tasks
    'todo_app': validate_todo_app,
    'blog_system': validate_blog_system,
    'inventory_system': validate_inventory_system,
    'url_shortener': validate_url_shortener,
    'email_validator_service': validate_email_validator,
    # L6 tasks
    'observer_pattern': validate_observer_pattern,
    'factory_pattern': validate_factory_pattern,
    'dependency_injection': validate_dependency_injection,
    'plugin_system': validate_plugin_system,
    'event_bus': validate_event_bus,
    # L7 tasks
    'rate_limiter': validate_rate_limiter,
    'connection_pool': validate_connection_pool,
    'circuit_breaker': validate_circuit_breaker,
    'distributed_cache': validate_distributed_cache,
}


def validate_task(task_name: str, workspace: Path) -> tuple[bool, str]:
    """
    Run flexible validation for a task.

    Returns:
        (success: bool, message: str)
    """
    validator = VALIDATORS.get(task_name)
    if not validator:
        return False, f"No flexible validator for {task_name}"

    return validator(workspace)
