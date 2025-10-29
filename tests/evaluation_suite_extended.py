"""
Extended Evaluation Suite: 7 Difficulty Levels with Multiple Tasks per Level

Designed to run for ~4 hours with comprehensive testing.

Levels:
- L1: Basic (single file, simple functions) - 6 tasks
- L2: Intermediate (classes, multiple files) - 6 tasks
- L3: Advanced (algorithms, data processing) - 6 tasks
- L4: Complex (APIs, async, testing) - 6 tasks
- L5: Integration (multi-component systems) - 5 tasks
- L6: Architecture (design patterns, refactoring) - 5 tasks
- L7: Expert (optimization, edge cases, production-ready) - 4 tasks

Total: 38 tasks × 5 runs = 190 evaluation runs
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.evaluation_suite import TaskDefinition


def get_extended_tasks() -> list[TaskDefinition]:
    """Define all 38 evaluation tasks across 7 difficulty levels."""
    return [
        # ===== LEVEL 1: BASIC (6 tasks, 20 rounds each) =====
        TaskDefinition(
            level=1,
            name="simple_function",
            description="Single file with one simple function",
            goal="Create a Python file called greet.py with a function greet(name) that returns 'Hello, {name}!'",
            expected_files=["greet.py"],
            validation_commands=[
                ["python", "-c", "from greet import greet; assert greet('World') == 'Hello, World!'"]
            ],
            timeout_rounds=20,
            tags=["single-file", "basic"]
        ),
        TaskDefinition(
            level=1,
            name="simple_math",
            description="Basic arithmetic operations",
            goal="Create math_ops.py with functions add(a,b), subtract(a,b), multiply(a,b), divide(a,b)",
            expected_files=["math_ops.py"],
            validation_commands=[
                ["python", "-c", "from math_ops import add, subtract, multiply, divide; assert add(2,3)==5; assert subtract(5,2)==3; assert multiply(4,5)==20; assert divide(10,2)==5"]
            ],
            timeout_rounds=20,
            tags=["single-file", "arithmetic"]
        ),
        TaskDefinition(
            level=1,
            name="list_operations",
            description="List manipulation functions",
            goal="Create list_utils.py with functions: get_first(lst), get_last(lst), reverse_list(lst)",
            expected_files=["list_utils.py"],
            validation_commands=[
                ["python", "-c", "from list_utils import get_first, get_last, reverse_list; assert get_first([1,2,3])==1; assert get_last([1,2,3])==3; assert reverse_list([1,2,3])==[3,2,1]"]
            ],
            timeout_rounds=20,
            tags=["single-file", "lists"]
        ),
        TaskDefinition(
            level=1,
            name="string_operations",
            description="String manipulation functions",
            goal="Create string_utils.py with: uppercase(s), lowercase(s), reverse_string(s), count_vowels(s)",
            expected_files=["string_utils.py"],
            validation_commands=[
                ["python", "-c", "from string_utils import uppercase, lowercase, reverse_string, count_vowels; assert uppercase('hello')=='HELLO'; assert lowercase('WORLD')=='world'; assert reverse_string('abc')=='cba'; assert count_vowels('hello')==2"]
            ],
            timeout_rounds=20,
            tags=["single-file", "strings"]
        ),
        TaskDefinition(
            level=1,
            name="number_checks",
            description="Number validation functions",
            goal="Create number_checks.py with: is_even(n), is_odd(n), is_positive(n), is_negative(n)",
            expected_files=["number_checks.py"],
            validation_commands=[
                ["python", "-c", "from number_checks import is_even, is_odd, is_positive, is_negative; assert is_even(4); assert is_odd(5); assert is_positive(10); assert is_negative(-5)"]
            ],
            timeout_rounds=20,
            tags=["single-file", "validation"]
        ),
        TaskDefinition(
            level=1,
            name="temperature_converter",
            description="Temperature conversion functions",
            goal="Create temp_converter.py with: celsius_to_fahrenheit(c), fahrenheit_to_celsius(f)",
            expected_files=["temp_converter.py"],
            validation_commands=[
                ["python", "-c", "from temp_converter import celsius_to_fahrenheit, fahrenheit_to_celsius; assert celsius_to_fahrenheit(0)==32; assert abs(fahrenheit_to_celsius(32)-0)<0.1"]
            ],
            timeout_rounds=20,
            tags=["single-file", "conversion"]
        ),

        # ===== LEVEL 2: INTERMEDIATE (6 tasks, 30 rounds each) =====
        TaskDefinition(
            level=2,
            name="person_class",
            description="Simple class with properties",
            goal="Create person.py with a Person class having name, age properties and a greet() method",
            expected_files=["person.py"],
            validation_commands=[
                ["python", "-c", "from person import Person; p=Person('Alice',30); assert p.name=='Alice'; assert p.age==30; assert 'Alice' in p.greet()"]
            ],
            timeout_rounds=30,
            tags=["oop", "class"]
        ),
        TaskDefinition(
            level=2,
            name="calculator_class",
            description="Calculator class with methods",
            goal="Create calculator.py with Calculator class having methods: add, subtract, multiply, divide, and history tracking",
            expected_files=["calculator.py"],
            validation_commands=[
                ["python", "-c", "from calculator import Calculator; c=Calculator(); assert c.add(2,3)==5; assert c.subtract(5,2)==3; assert len(c.history)>=2"]
            ],
            timeout_rounds=30,
            tags=["oop", "state"]
        ),
        TaskDefinition(
            level=2,
            name="multi_file_package",
            description="Package with multiple modules",
            goal="Create package 'shapes' with circle.py (area, circumference) and square.py (area, perimeter)",
            expected_files=["shapes/__init__.py", "shapes/circle.py", "shapes/square.py"],
            validation_commands=[
                ["python", "-c", "from shapes.circle import area as circle_area; from shapes.square import area as square_area; assert circle_area(1) > 3; assert square_area(2) == 4"]
            ],
            timeout_rounds=30,
            tags=["multi-file", "package"]
        ),
        TaskDefinition(
            level=2,
            name="file_reader_writer",
            description="File I/O operations",
            goal="Create file_ops.py with write_file(path, content) and read_file(path) functions",
            expected_files=["file_ops.py"],
            validation_commands=[
                ["python", "-c", "from file_ops import write_file, read_file; write_file('test.txt', 'hello'); assert read_file('test.txt')=='hello'"]
            ],
            timeout_rounds=30,
            tags=["file-io"]
        ),
        TaskDefinition(
            level=2,
            name="data_validator",
            description="Input validation functions",
            goal="Create validator.py with: validate_email(email), validate_phone(phone), validate_age(age)",
            expected_files=["validator.py"],
            validation_commands=[
                ["python", "-c", "from validator import validate_email, validate_phone, validate_age; assert validate_email('test@example.com'); assert not validate_email('invalid'); assert validate_age(25); assert not validate_age(-5)"]
            ],
            timeout_rounds=30,
            tags=["validation", "regex"]
        ),
        TaskDefinition(
            level=2,
            name="counter_class",
            description="Stateful counter class",
            goal="Create counter.py with Counter class: increment(), decrement(), reset(), get_value()",
            expected_files=["counter.py"],
            validation_commands=[
                ["python", "-c", "from counter import Counter; c=Counter(); c.increment(); c.increment(); assert c.get_value()==2; c.decrement(); assert c.get_value()==1; c.reset(); assert c.get_value()==0"]
            ],
            timeout_rounds=30,
            tags=["oop", "state"]
        ),

        # ===== LEVEL 3: ADVANCED (6 tasks, 40 rounds each) =====
        TaskDefinition(
            level=3,
            name="bubble_sort",
            description="Implement bubble sort",
            goal="Create sorting.py with bubble_sort(lst) function that sorts a list in ascending order",
            expected_files=["sorting.py"],
            validation_commands=[
                ["python", "-c", "from sorting import bubble_sort; assert bubble_sort([3,1,4,1,5,9,2,6])==[1,1,2,3,4,5,6,9]"]
            ],
            timeout_rounds=40,
            tags=["algorithm", "sorting"]
        ),
        TaskDefinition(
            level=3,
            name="binary_search",
            description="Implement binary search",
            goal="Create search.py with binary_search(lst, target) that returns index of target or -1",
            expected_files=["search.py"],
            validation_commands=[
                ["python", "-c", "from search import binary_search; assert binary_search([1,2,3,4,5], 3)==2; assert binary_search([1,2,3,4,5], 6)==-1"]
            ],
            timeout_rounds=40,
            tags=["algorithm", "search"]
        ),
        TaskDefinition(
            level=3,
            name="json_parser",
            description="JSON file reader/writer",
            goal="Create json_utils.py with: load_json(path), save_json(path, data), get_value(data, key)",
            expected_files=["json_utils.py"],
            validation_commands=[
                ["python", "-c", "from json_utils import save_json, load_json; save_json('test.json', {'name':'Alice'}); data=load_json('test.json'); assert data['name']=='Alice'"]
            ],
            timeout_rounds=40,
            tags=["json", "file-io"]
        ),
        TaskDefinition(
            level=3,
            name="csv_processor",
            description="CSV data processing",
            goal="Create csv_utils.py with: read_csv(path), write_csv(path, rows), filter_rows(rows, condition)",
            expected_files=["csv_utils.py"],
            validation_commands=[
                ["python", "-c", "from csv_utils import write_csv, read_csv; write_csv('test.csv', [['name','age'],['Alice','30']]); rows=read_csv('test.csv'); assert len(rows)==2"]
            ],
            timeout_rounds=40,
            tags=["csv", "data"]
        ),
        TaskDefinition(
            level=3,
            name="cache_decorator",
            description="Memoization decorator",
            goal="Create cache.py with @cache decorator that memoizes function results",
            expected_files=["cache.py"],
            validation_commands=[
                ["python", "-c", "from cache import cache; @cache\ndef fib(n): return n if n<2 else fib(n-1)+fib(n-2)\nassert fib(10)==55"]
            ],
            timeout_rounds=40,
            tags=["decorator", "optimization"]
        ),
        TaskDefinition(
            level=3,
            name="linked_list",
            description="Linked list data structure",
            goal="Create linked_list.py with LinkedList class: append(val), remove(val), contains(val), to_list()",
            expected_files=["linked_list.py"],
            validation_commands=[
                ["python", "-c", "from linked_list import LinkedList; ll=LinkedList(); ll.append(1); ll.append(2); ll.append(3); assert ll.to_list()==[1,2,3]; assert ll.contains(2)"]
            ],
            timeout_rounds=40,
            tags=["data-structure"]
        ),

        # ===== LEVEL 4: COMPLEX (6 tasks, 50 rounds each) =====
        TaskDefinition(
            level=4,
            name="rest_api_mock",
            description="Mock REST API with Flask",
            goal="Create api.py with Flask app having GET /users and POST /users endpoints with in-memory storage",
            expected_files=["api.py"],
            validation_commands=[
                ["python", "-c", "from api import app; assert app is not None"]
            ],
            timeout_rounds=50,
            tags=["flask", "api"]
        ),
        TaskDefinition(
            level=4,
            name="sqlite_manager",
            description="SQLite database wrapper",
            goal="Create db.py with Database class: create_table, insert, query, update, delete",
            expected_files=["db.py"],
            validation_commands=[
                ["python", "-c", "from db import Database; db=Database(':memory:'); db.create_table('users', 'name TEXT, age INTEGER'); db.insert('users', {'name':'Alice','age':30}); rows=db.query('users'); assert len(rows)==1"]
            ],
            timeout_rounds=50,
            tags=["database", "sqlite"]
        ),
        TaskDefinition(
            level=4,
            name="async_downloader",
            description="Async file downloader",
            goal="Create downloader.py with async download_file(url, path) and download_multiple(urls)",
            expected_files=["downloader.py"],
            validation_commands=[
                ["python", "-c", "from downloader import download_multiple; import asyncio; assert download_multiple is not None"]
            ],
            timeout_rounds=50,
            tags=["async", "network"]
        ),
        TaskDefinition(
            level=4,
            name="test_framework_basic",
            description="Mini testing framework",
            goal="Create test_framework.py with TestRunner class that can run test functions and report results",
            expected_files=["test_framework.py"],
            validation_commands=[
                ["python", "-c", "from test_framework import TestRunner; runner=TestRunner(); assert hasattr(runner, 'run')"]
            ],
            timeout_rounds=50,
            tags=["testing", "framework"]
        ),
        TaskDefinition(
            level=4,
            name="command_parser",
            description="CLI argument parser",
            goal="Create cli_parser.py with Parser class that parses command line arguments with flags and options",
            expected_files=["cli_parser.py"],
            validation_commands=[
                ["python", "-c", "from cli_parser import Parser; p=Parser(); p.add_argument('--name'); args=p.parse(['--name','Alice']); assert args.name=='Alice'"]
            ],
            timeout_rounds=50,
            tags=["cli", "parsing"]
        ),
        TaskDefinition(
            level=4,
            name="config_loader",
            description="Configuration file loader",
            goal="Create config.py with Config class that loads YAML/JSON config files with environment variable interpolation",
            expected_files=["config.py"],
            validation_commands=[
                ["python", "-c", "from config import Config; c=Config(); assert hasattr(c, 'load')"]
            ],
            timeout_rounds=50,
            tags=["config", "yaml"]
        ),

        # ===== LEVEL 5: INTEGRATION (5 tasks, 60 rounds each) =====
        TaskDefinition(
            level=5,
            name="blog_system",
            description="Simple blog with posts and comments",
            goal="Create blog system: Post model, Comment model, BlogManager with CRUD operations, persistence to JSON",
            expected_files=["blog.py", "models.py", "storage.py"],
            validation_commands=[
                ["python", "-c", "from blog import BlogManager; bm=BlogManager(); bm.create_post('Title', 'Content'); assert len(bm.get_posts())==1"]
            ],
            timeout_rounds=60,
            tags=["integration", "crud"]
        ),
        TaskDefinition(
            level=5,
            name="todo_app",
            description="Todo application with categories",
            goal="Create todo app: Todo model, Category model, TodoManager with filtering, sorting, and JSON persistence",
            expected_files=["todo.py", "models.py", "manager.py"],
            validation_commands=[
                ["python", "-c", "from todo import TodoManager; tm=TodoManager(); tm.add_todo('Task 1', 'work'); assert len(tm.get_todos())==1"]
            ],
            timeout_rounds=60,
            tags=["integration", "app"]
        ),
        TaskDefinition(
            level=5,
            name="inventory_system",
            description="Product inventory manager",
            goal="Create inventory system: Product model, Inventory class with add/remove/search, low-stock alerts, CSV export",
            expected_files=["inventory.py", "product.py", "alerts.py"],
            validation_commands=[
                ["python", "-c", "from inventory import Inventory; inv=Inventory(); inv.add_product('Widget', 10, 2.5); assert len(inv.get_products())==1"]
            ],
            timeout_rounds=60,
            tags=["integration", "business"]
        ),
        TaskDefinition(
            level=5,
            name="url_shortener",
            description="URL shortener service",
            goal="Create URL shortener: generate short codes, store mappings, redirect lookup, statistics tracking",
            expected_files=["shortener.py", "storage.py", "stats.py"],
            validation_commands=[
                ["python", "-c", "from shortener import URLShortener; us=URLShortener(); short=us.shorten('https://example.com'); assert len(short)<20"]
            ],
            timeout_rounds=60,
            tags=["integration", "service"]
        ),
        TaskDefinition(
            level=5,
            name="email_validator_service",
            description="Email validation and verification",
            goal="Create email service: syntax validation, domain verification, disposable email detection, bulk validation",
            expected_files=["email_service.py", "validators.py", "blacklist.py"],
            validation_commands=[
                ["python", "-c", "from email_service import EmailValidator; ev=EmailValidator(); assert ev.validate('test@example.com')"]
            ],
            timeout_rounds=60,
            tags=["integration", "validation"]
        ),

        # ===== LEVEL 6: ARCHITECTURE (5 tasks, 70 rounds each) =====
        TaskDefinition(
            level=6,
            name="observer_pattern",
            description="Implement observer design pattern",
            goal="Create observer pattern: Subject, Observer classes, event system with subscribe/unsubscribe/notify",
            expected_files=["observer.py", "subject.py", "events.py"],
            validation_commands=[
                ["python", "-c", "from observer import Subject, Observer; s=Subject(); o=Observer(); s.attach(o); s.notify(); assert True"]
            ],
            timeout_rounds=70,
            tags=["design-pattern", "architecture"]
        ),
        TaskDefinition(
            level=6,
            name="factory_pattern",
            description="Implement factory design pattern",
            goal="Create factory pattern: Product interface, ConcreteProducts, Factory class with create_product method",
            expected_files=["factory.py", "products.py"],
            validation_commands=[
                ["python", "-c", "from factory import Factory; f=Factory(); p=f.create_product('type_a'); assert p is not None"]
            ],
            timeout_rounds=70,
            tags=["design-pattern", "architecture"]
        ),
        TaskDefinition(
            level=6,
            name="dependency_injection",
            description="Dependency injection container",
            goal="Create DI container: register services, resolve dependencies, singleton/transient lifetimes",
            expected_files=["container.py", "services.py"],
            validation_commands=[
                ["python", "-c", "from container import Container; c=Container(); c.register('service', lambda: 'test'); assert c.resolve('service')=='test'"]
            ],
            timeout_rounds=70,
            tags=["design-pattern", "di"]
        ),
        TaskDefinition(
            level=6,
            name="plugin_system",
            description="Plugin architecture",
            goal="Create plugin system: Plugin base class, PluginManager for loading/registering, plugin discovery",
            expected_files=["plugin_manager.py", "plugin.py", "loader.py"],
            validation_commands=[
                ["python", "-c", "from plugin_manager import PluginManager; pm=PluginManager(); assert hasattr(pm, 'load_plugin')"]
            ],
            timeout_rounds=70,
            tags=["architecture", "extensibility"]
        ),
        TaskDefinition(
            level=6,
            name="event_bus",
            description="Event-driven architecture",
            goal="Create event bus: publish/subscribe system, event filtering, async event handling",
            expected_files=["event_bus.py", "handlers.py", "events.py"],
            validation_commands=[
                ["python", "-c", "from event_bus import EventBus; eb=EventBus(); eb.subscribe('test', lambda e: None); eb.publish('test', {}); assert True"]
            ],
            timeout_rounds=70,
            tags=["architecture", "events"]
        ),

        # ===== LEVEL 7: EXPERT (4 tasks, 80 rounds each) =====
        TaskDefinition(
            level=7,
            name="rate_limiter",
            description="Production-grade rate limiter",
            goal="Create rate limiter: token bucket algorithm, sliding window, distributed support, Redis backend",
            expected_files=["rate_limiter.py", "algorithms.py", "backends.py"],
            validation_commands=[
                ["python", "-c", "from rate_limiter import RateLimiter; rl=RateLimiter(10, 60); assert rl.allow('user1')"]
            ],
            timeout_rounds=80,
            tags=["expert", "optimization", "production"]
        ),
        TaskDefinition(
            level=7,
            name="connection_pool",
            description="Database connection pooling",
            goal="Create connection pool: acquire/release connections, max pool size, timeout handling, health checks",
            expected_files=["pool.py", "connection.py", "health.py"],
            validation_commands=[
                ["python", "-c", "from pool import ConnectionPool; pool=ConnectionPool(5); conn=pool.acquire(); assert conn is not None; pool.release(conn)"]
            ],
            timeout_rounds=80,
            tags=["expert", "database", "production"]
        ),
        TaskDefinition(
            level=7,
            name="circuit_breaker",
            description="Circuit breaker pattern",
            goal="Create circuit breaker: failure detection, half-open state, automatic recovery, metrics tracking",
            expected_files=["circuit_breaker.py", "states.py", "metrics.py"],
            validation_commands=[
                ["python", "-c", "from circuit_breaker import CircuitBreaker; cb=CircuitBreaker(); assert hasattr(cb, 'call')"]
            ],
            timeout_rounds=80,
            tags=["expert", "resilience", "production"]
        ),
        TaskDefinition(
            level=7,
            name="distributed_cache",
            description="Distributed caching system",
            goal="Create distributed cache: consistent hashing, replication, cache invalidation, TTL support",
            expected_files=["cache.py", "hash_ring.py", "replication.py"],
            validation_commands=[
                ["python", "-c", "from cache import DistributedCache; dc=DistributedCache(); dc.set('key', 'value'); assert dc.get('key')=='value'"]
            ],
            timeout_rounds=80,
            tags=["expert", "distributed", "production"]
        ),
    ]
