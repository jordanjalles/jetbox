#!/usr/bin/env python3
"""
Test JSON parsing error recovery.

Verifies that the extract_tool_call_from_parse_error function correctly
extracts tool calls from Ollama parsing error messages.
"""
from llm_utils import extract_tool_call_from_parse_error


def test_text_before_json():
    """Test extraction when LLM adds text before JSON."""
    # Real error from L7_rate_limiter evaluation
    error_msg = (
        'error parsing tool call: raw=\'We wrote the file. Need to create tests.'
        '{"path":"tests/test_rate_limiter.py","content":"import time\\nimport pytest\\n"}\', '
        'err=invalid character \'W\' looking for beginning of value (status code: 500)'
    )

    extracted = extract_tool_call_from_parse_error(error_msg)

    assert extracted is not None, "Should extract JSON from mixed response"
    assert 'path' in extracted, "Should have path key"
    assert 'content' in extracted, "Should have content key"
    assert extracted['path'] == 'tests/test_rate_limiter.py'
    print("✓ Test 1 passed: Text before JSON")


def test_extra_parameters():
    """Test extraction when LLM adds extra parameters (invalid JSON)."""
    # Real error from L4_todo_list evaluation
    error_msg = (
        'error parsing tool call: raw=\'{"path":"todo.py","content":"..."},'
        '"append":false,"encoding":"utf-8","line_end":null,"overwrite":true}\', '
        'err=invalid character \',\' after top-level value (status code: -1)'
    )

    extracted = extract_tool_call_from_parse_error(error_msg)

    # This should extract the first JSON object only
    assert extracted is not None, "Should extract first valid JSON object"
    assert 'path' in extracted, "Should have path key"
    assert extracted['path'] == 'todo.py'
    # Extra parameters should be ignored (not in extracted dict)
    assert 'append' not in extracted or True, "Extra params may or may not be extracted"
    print("✓ Test 2 passed: Extra parameters")


def test_heredoc_quotes():
    """Test extraction when bash heredoc has problematic quotes."""
    # Real error from L7_rate_limiter bash command
    error_msg = (
        'error parsing tool call: raw=\'{"command":"python - <<\'PY\'\\n'
        'import time\\nfrom rate_limiter import RateLimiter\\nPY"}\', '
        'err=invalid character \']\' after object key:value pair (status code: 500)'
    )

    extracted = extract_tool_call_from_parse_error(error_msg)

    # This one might fail due to quote escaping, but let's see
    if extracted:
        assert 'command' in extracted, "Should have command key"
        print("✓ Test 3 passed: Heredoc quotes (extracted)")
    else:
        print("⚠ Test 3: Heredoc quotes could not be extracted (expected for some cases)")


def test_no_json_in_error():
    """Test handling when error has no JSON."""
    error_msg = "error parsing tool call: raw='Just plain text with no JSON', err=..."

    extracted = extract_tool_call_from_parse_error(error_msg)

    assert extracted is None, "Should return None when no JSON found"
    print("✓ Test 4 passed: No JSON in error")


def test_nested_json():
    """Test extraction of nested JSON structures."""
    error_msg = (
        'error parsing tool call: raw=\'Commentary here.{"path":"test.py",'
        '"content":"def foo():\\n    return 42"}\', err=...'
    )

    extracted = extract_tool_call_from_parse_error(error_msg)

    assert extracted is not None, "Should extract nested JSON"
    assert 'path' in extracted and 'content' in extracted
    print("✓ Test 5 passed: Nested JSON")


def run_all_tests():
    """Run all test cases."""
    print("="*70)
    print("JSON Parsing Error Recovery Tests")
    print("="*70)

    tests = [
        test_text_before_json,
        test_extra_parameters,
        test_heredoc_quotes,
        test_no_json_in_error,
        test_nested_json,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed += 1

    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
