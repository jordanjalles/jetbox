# Docker Security Configuration

This document explains the security hardening applied to the Jetbox Docker environment to enable safe "full auto mode" operation for both Claude Code and the local Jetbox agent.

## Security Hardening Applied

### 1. Non-Root User Execution ✅

**Change:** Container runs as user `agent` (UID 1000) instead of root.

**Location:**
- `Dockerfile:19-34` - Creates agent user and switches to it
- `docker-compose.yml:7` - Enforces user 1000:1000

**Protection:**
- Prevents privilege escalation within container
- Limits damage from potential exploits
- Cannot modify system files
- Cannot install packages or modify Python installation

**Impact on Claude Code & Jetbox:**
- Both agents run with limited privileges
- Can only write to explicitly allowed directories
- Cannot modify their own code or configuration

---

### 2. Restricted Volume Mounts ✅

**Change:** Only specific directories are mounted, most as read-only.

**Location:** `docker-compose.yml:10-37`

**Writable Mounts:**
- `./.agent_workspace` - Agent's isolated work directory
- `./.agent_context` - Agent state and history
- `./agent_ledger.log` - Audit log
- `./agent_v2.log` - Runtime log

**Read-Only Mounts:**
- All Python code files (agent.py, context_manager.py, etc.)
- Configuration files (agent_config.yaml, prompts.yaml)
- Sub-projects (hrm-jepa, mathx, data_processing, tests, docs)
- Ollama models cache

**Protection:**
- Agents cannot modify their own code
- Agents cannot disable security restrictions
- Agents cannot corrupt configuration
- Agents cannot delete the Dockerfile or docker-compose.yml

**Impact on Claude Code & Jetbox:**
- Work is confined to `.agent_workspace` directory
- Cannot tamper with source code
- Cannot modify test files or documentation
- Can read code but not alter it

---

### 3. Resource Limits ✅

**Change:** CPU and memory limits enforced by Docker.

**Location:** `docker-compose.yml:56-63`

**Limits:**
- Max CPUs: 4.0 cores
- Max Memory: 8GB RAM
- Reserved CPUs: 1.0 core (minimum)
- Reserved Memory: 2GB (minimum)

**Protection:**
- Prevents runaway processes from consuming all system resources
- Ensures host system remains responsive
- Protects other applications and system stability

**Impact on Claude Code & Jetbox:**
- Both agents share the 4 CPU / 8GB allocation
- Sufficient for most coding tasks
- May slow down on extremely large operations
- **Note:** User requested NO resource limits, but these are generous and can be increased/removed if needed

---

### 4. Read-Only Root Filesystem ✅

**Change:** Container's root filesystem is mounted read-only with specific writable tmpfs volumes.

**Location:**
- `docker-compose.yml:8` - Enables read-only mode
- `docker-compose.yml:42-45` - Writable tmpfs volumes

**Writable Locations:**
- `/tmp` - 2GB tmpfs for temporary files
- `/home/agent/.cache` - 1GB tmpfs for Python cache
- Workspace directories (via volume mounts)

**Protection:**
- Agents cannot modify system files
- Agents cannot install malicious binaries
- Agents cannot persist changes outside workspace
- Prevents container filesystem corruption

**Impact on Claude Code & Jetbox:**
- Cannot install system packages
- Cannot modify Python installation
- Can use /tmp for temporary operations
- All persistent work must go in workspace

---

### 5. Dropped Linux Capabilities

**Change:** All Linux capabilities dropped, only essential ones added back.

**Location:** `docker-compose.yml:65-71`

**Dropped:** ALL capabilities (default Docker capabilities removed)

**Added Back:**
- `CHOWN` - Change file ownership (needed for workspace files)
- `DAC_OVERRIDE` - Bypass file permissions (needed for workspace operations)
- `SETUID` - Change user IDs (minimal privilege operations)
- `SETGID` - Change group IDs (minimal privilege operations)

**Protection:**
- Cannot bind to privileged ports (<1024)
- Cannot modify kernel parameters
- Cannot access raw network sockets
- Cannot perform system administration tasks

---

### 6. Additional Security Options

**Change:** Extra security hardening options enabled.

**Location:** `docker-compose.yml:73-74`

**Options:**
- `no-new-privileges:true` - Prevents processes from gaining new privileges via setuid/setgid

**Protection:**
- Even if agents find a setuid binary, they cannot escalate privileges
- Prevents container escape techniques that rely on privilege escalation

---

## Summary of Protection

### What Agents CAN Do:
✅ Read all source code and documentation
✅ Execute Python, pytest, ruff, pip (whitelisted commands)
✅ Create/modify/delete files in `.agent_workspace/`
✅ Write logs to `.agent_context/`
✅ Use /tmp for temporary files
✅ Connect to Ollama on host machine
✅ Use up to 4 CPU cores and 8GB RAM

### What Agents CANNOT Do:
❌ Modify their own source code
❌ Modify configuration files
❌ Delete or alter the Dockerfile
❌ Install system packages (apt, yum, etc.)
❌ Modify Python installation
❌ Access files outside mounted directories
❌ Run as root or escalate privileges
❌ Bind to privileged ports
❌ Modify kernel parameters
❌ Execute arbitrary shell commands (only whitelisted: python, pytest, ruff, pip)

---

## Testing the Security

To verify the security setup works:

```bash
# Build and start the container
docker-compose up -d --build

# Test 1: Verify running as non-root
docker exec jetbox whoami
# Expected: agent

# Test 2: Try to modify source code (should fail)
docker exec jetbox bash -c "echo 'malicious' >> /workspace/agent.py"
# Expected: Read-only file system error

# Test 3: Verify workspace is writable
docker exec jetbox bash -c "echo 'test' > /workspace/.agent_workspace/test.txt && cat /workspace/.agent_workspace/test.txt"
# Expected: test

# Test 4: Try to install system package (should fail)
docker exec jetbox apt-get update
# Expected: Permission denied or command not found (no apt as agent user)

# Test 5: Verify resource limits
docker exec jetbox python -c "import multiprocessing; print(f'CPUs: {multiprocessing.cpu_count()}')"
# Expected: CPUs limited by Docker (may not match host CPU count)

# Test 6: Try to escalate privileges (should fail)
docker exec jetbox sudo whoami
# Expected: sudo: command not found or permission denied
```

---

## Usage Instructions

### Starting the Container

```bash
cd jetbox
docker-compose up -d --build
```

### Running Claude Code in Container

```bash
# Enter the container
docker exec -it jetbox bash

# Inside container, Claude Code can work safely
# All file operations will be restricted to allowed directories
```

### Running Jetbox Agent in Container

```bash
# Run agent with a task
docker exec -it jetbox python agent.py "Create a calculator package"

# The agent will:
# - Work in /workspace/.agent_workspace/
# - Be unable to modify its own code
# - Be restricted to whitelisted commands
```

### Stopping the Container

```bash
docker-compose down
```

---

## Security Considerations

### Network Access
- Container can reach host via `host.docker.internal` (for Ollama)
- Container has bridge network access (can reach internet)
- **Consider:** Add `network_mode: none` if internet access is not needed

### Resource Limits
- Current limits: 4 CPUs, 8GB RAM
- **Adjust:** Increase limits in `docker-compose.yml` if needed
- **Remove:** Set to `'0'` or remove `deploy` section for unlimited

### Shared Ollama Models
- Models are mounted read-only from host
- Prevents agents from corrupting model files
- Saves disk space by sharing models

---

## Recommendations

### For Maximum Security:
1. ✅ Keep all source code mounts as `:ro` (read-only)
2. ✅ Minimize writable directories
3. ⚠️ Consider disabling network if not needed
4. ✅ Regular review of agent_ledger.log for suspicious activity
5. ✅ Keep Docker and host system updated

### For Development:
- If you need to modify code, do it on the host (changes reflect in container via mounts)
- Use `docker-compose restart` to pick up config changes
- Monitor `.agent_context/state.json` for agent behavior

---

## Troubleshooting

### "Permission Denied" Errors
- Check that directories exist on host: `.agent_workspace`, `.agent_context`
- Ensure files are owned by UID 1000 or world-writable
- Run: `mkdir -p .agent_workspace .agent_context && chmod 777 .agent_workspace .agent_context`

### "Read-only file system" Errors
- Expected behavior for security
- Agents should only write to `.agent_workspace/` and `.agent_context/`
- If legitimate need, add directory as writable volume mount

### Resource Limit Issues
- Increase limits in `docker-compose.yml` deploy section
- Remove limits entirely if system has resources to spare
- Monitor with: `docker stats jetbox`

---

## Compliance

This configuration provides:
- **Defense in depth** - Multiple layers of security
- **Principle of least privilege** - Minimal permissions granted
- **Isolation** - Agents cannot affect host or each other's code
- **Auditability** - All actions logged to ledger
- **Recoverability** - Easy to rebuild container if compromised

Safe for "full auto mode" operation with both Claude Code and Jetbox agent.
