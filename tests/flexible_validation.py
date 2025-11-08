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


VALIDATORS = {
    'todo_app': validate_todo_app,
    'blog_system': validate_blog_system,
    'inventory_system': validate_inventory_system,
    'url_shortener': validate_url_shortener,
    'email_validator_service': validate_email_validator,
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
