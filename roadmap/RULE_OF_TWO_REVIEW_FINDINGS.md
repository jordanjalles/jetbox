# Rule of Two System - Deep Review Findings

**Date**: 2025-11-12
**Status**: Issues found - 1 critical, 2 judgment calls

---

## ✅ FIXED: Critical Security Gap

### Issue: CODE_EXECUTION_COMMANDS Missing

**Problem**: CommandTools only checked FILE_READING_COMMANDS (`cat`, `grep`) for [A], missing CODE_EXECUTION_COMMANDS (`python`, `node`, `ruby`).

**Attack scenario**:
1. Workspace has `malicious.py` with prompt injection
2. Whitelist has `python` but not `cat`
3. CommandTools incorrectly gets `[]` instead of `[A]`
4. Agent runs `python malicious.py` → prompt injection succeeds

**Fix applied**:
```python
CODE_EXECUTION_COMMANDS = {
    'python', 'python3', 'node', 'ruby', 'perl', 'php',
    'bash', 'sh', 'zsh', 'java', 'gcc', 'cargo', 'go', ...
}

# Check if whitelist can access untrusted files (read OR execute)
has_file_access = bool(self.whitelist & (
    self.FILE_READING_COMMANDS | self.CODE_EXECUTION_COMMANDS
))
```

**Impact**: Now `python` + untrusted workspace correctly triggers [A].

---

## ⚠️ JUDGMENT CALL: ServerTools and Docker Always [B]?

### Current State

**ServerToolsBehavior**:
```python
# Always [B], regardless of workspace
props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)
if security_context.workspace_has_network_access:
    props.add(RuleOfTwoProperty.EXTERNAL_ACTION)
```

**DockerBehavior**:
```python
# Static [B]
rule_of_two_properties = {RuleOfTwoProperty.SENSITIVE_ACCESS}
```

### The Question

Should ServerTools and Docker **always** have [B], or should they be workspace-dependent like other behaviors?

**Arguments for ALWAYS [B]** (current):
1. **Opaque processes**: We don't control what Docker containers or servers do internally
2. **Unrestricted access**: They can access ANY file in workspace, not limited by whitelist
3. **Conservative**: Better safe than sorry for high-risk behaviors
4. **User expectation**: Users expect Docker/servers to be "dangerous"

**Arguments for WORKSPACE-DEPENDENT [B]**:
1. **Consistency**: Every other behavior checks workspace state
2. **Accuracy**: Can't leak .env if .env doesn't exist
3. **Workspace-centric model**: User declares workspace characteristics, behaviors adapt
4. **Better risk assessment**: `[]` is more accurate than `[B]` when no sensitive files

### Proposed Change (if workspace-centric)

**ServerToolsBehavior**:
```python
def get_rule_of_two_properties(self, agent, security_context):
    props = set()
    # [B] ONLY if workspace has sensitive files
    if security_context and security_context.workspace_has_sensitive_files:
        props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)
    if security_context and security_context.workspace_has_network_access:
        props.add(RuleOfTwoProperty.EXTERNAL_ACTION)
    return props
```

**DockerBehavior**:
```python
def get_rule_of_two_properties(self, agent, security_context):
    props = set()
    # [B] ONLY if workspace has sensitive files
    if security_context and security_context.workspace_has_sensitive_files:
        props.add(RuleOfTwoProperty.SENSITIVE_ACCESS)
    return props
```

### Impact

**If kept always [B]**:
- Jetbox development (no sensitive files): ServerTools is `[B]` or `[BC]` (more restrictive than needed)
- Safety project (with .env): ServerTools is `[B]` or `[BC]` (correct)

**If made workspace-dependent**:
- Jetbox development: ServerTools is `[]` or `[C]` (more permissive, more accurate)
- Safety project: ServerTools is `[B]` or `[BC]` (same as before)

**Recommendation**: Lean toward **workspace-dependent** for consistency with the workspace-centric model, but mark ServerTools/Docker as "high-risk opaque behaviors" in docs.

---

## 📝 DOCUMENTATION FIX NEEDED

### Issue: WriteFile Mismatch

**Documentation says**: WriteFileBehavior is static `[B]`
**Code says**: `rule_of_two_properties = set()` (i.e., `[]`)

**Correct answer**: Code is right, WriteFile is `[]` because:
- Writing files is NOT accessing sensitive data (reading is)
- Writing agent-generated content is NOT processing untrusted input
- Writing locally is NOT external communication

**Fix needed**: Update classification docs to say WriteFile is `[]`.

---

## 🎯 Additional Findings

### 1. HttpRequestBehavior Special Case

HttpRequestBehavior is the **only** behavior where [A] is intrinsic (not workspace-dependent):
- HTTP responses are ALWAYS untrusted external data
- [A] is inherent to HTTP fetching, not from workspace files

All other [A] and [B] properties are workspace-dependent.

**Documentation note needed**: Call out HttpRequest as the exception to workspace-centric model.

### 2. Future: External Secrets

For future SecretsManagerBehavior (fetches from 1Password/Vault), need to extend workspace config:

```yaml
workspace:
  has_sensitive_files: false           # No .env in workspace
  accesses_external_secrets: true      # Fetches from external vault
```

Then SecretsManager would check both flags for [B].

**Action**: Note for future extension, not urgent now.

---

## Summary of Actions

### Completed ✅
1. ✅ Added CODE_EXECUTION_COMMANDS to CommandTools
2. ✅ Fixed [A] detection to include code execution

### Needs Decision 🤔
1. **Should ServerTools and Docker be workspace-dependent for [B]?**
   - Current: Always [B]
   - Proposal: [B] only if workspace_has_sensitive_files
   - Recommendation: Make workspace-dependent for consistency

### Needs Update 📝
2. **Fix documentation mismatch**:
   - Update classification docs: WriteFile is `[]`, not `[B]`
   - Call out HttpRequest as exception to workspace-centric model
   - Update matrix if ServerTools/Docker made dynamic

### Future 🔮
3. **Consider** extending SecurityContext for external secrets access

---

## Test Coverage Gaps

Should add tests for:
1. ✅ CommandTools with `python` + untrusted workspace → [A]
2. ServerTools with sensitive=true → [B] (or [] if made dynamic)
3. Docker with sensitive=true → [B] (or [] if made dynamic)
4. HttpRequest always [A] when network enabled (special case)

---

## Conclusion

The workspace-centric model is **fundamentally sound**. Main findings:
- 1 critical gap fixed (CODE_EXECUTION_COMMANDS)
- 2 judgment calls (ServerTools/Docker dynamic vs static)
- 1 documentation fix (WriteFile is `[]`)
- Overall system is consistent and well-designed

**Recommendation**: Make ServerTools/Docker workspace-dependent to fully commit to workspace-centric model.
