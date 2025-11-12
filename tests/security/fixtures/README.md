# Test Fixtures for Security Tests

This directory contains test data for security feature testing.

## Contents (to be added in later phases)

### Phase 4A: Input Validation
- `injection_corpus.json` - Prompt injection test samples (malicious + benign)
- `malicious_files/` - Files containing various injection patterns
- `benign_files/` - Clean files for false positive testing

### Phase 4B: Access Auditing
- `credential_files/` - Mock credential files for testing access patterns
- `normal_workflow_files/` - Files for legitimate access pattern testing

### Phase 4C: Network Audit
- Mock git repositories for testing staging area analysis
- Sample commands for network operation detection

## Usage

Tests will load fixtures from this directory during test execution.
See individual test files for fixture usage examples.
