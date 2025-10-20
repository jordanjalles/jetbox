# Docker Setup for Jetbox Agent

Run the coding agent in an isolated Docker container for safe YOLO mode execution.

## Why Docker?

- **Isolation**: Agent runs in its own environment, can't affect your host system
- **Safety**: Perfect for unrestricted/YOLO mode - let the agent do anything within the container
- **Reproducibility**: Consistent environment across machines
- **Easy cleanup**: Just delete the container if something goes wrong

## Prerequisites

1. **Docker Desktop** installed and running
2. **Ollama** running on your host machine (so models are shared)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# Run the agent with a task
docker-compose exec jetbox python agent.py "Create mathx package with add and multiply functions"

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 2: Using Docker Commands

```bash
# Build the image
docker build -t jetbox .

# Run the container
docker run -d \
  --name jetbox \
  -v "$(pwd):/workspace" \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=gpt-oss:20b \
  --add-host host.docker.internal:host-gateway \
  jetbox

# Execute agent tasks
docker exec -it jetbox python agent.py "your task here"

# Stop and remove
docker stop jetbox && docker rm jetbox
```

## Usage Patterns

### Interactive Shell

```bash
# Enter the container
docker-compose exec jetbox bash

# Now you're inside - run commands directly
python agent.py "Create a new package"
pytest
ruff check .
```

### One-off Tasks

```bash
# Run a single task and view output
docker-compose exec jetbox python agent.py "Add multiply function to mathx"
```

### Monitor Agent Activity

```bash
# Watch the agent log in real-time
docker-compose exec jetbox tail -f agent.log

# View ledger
docker-compose exec jetbox cat agent_ledger.log
```

### Using Different Models

```bash
# Set model via environment variable
docker-compose exec -e OLLAMA_MODEL=qwen2.5-coder:7b jetbox python agent.py "your task"

# Or modify docker-compose.yml to change default
```

## File Persistence

All files created by the agent are saved to your host machine because of the volume mount:
- `./workspace` in container → `./` on host
- Changes persist even after container restarts

## YOLO Mode Safety

When running Claude Code in YOLO mode to interact with this agent:

1. **Container isolation** prevents system-level damage
2. **Volume mounts** limit file access to project directory only
3. **Network isolation** prevents unintended external connections
4. **Easy reset**: `docker-compose down -v && docker-compose up -d`

## Troubleshooting

### Can't connect to Ollama

**Problem**: Agent can't reach Ollama on host
**Solution**:
```bash
# Verify Ollama is running on host
ollama list

# Check if host.docker.internal works
docker-compose exec jetbox ping -c 1 host.docker.internal

# Try alternative: use host IP instead
docker-compose exec -e OLLAMA_HOST=http://192.168.1.X:11434 jetbox python agent.py "test"
```

### Models not found

**Problem**: Docker can't access Ollama models
**Solution**: Use host Ollama (default setup) rather than installing Ollama in container

### Permission errors on Windows

**Problem**: File permission issues with volume mounts
**Solution**:
```bash
# Ensure Docker Desktop has access to your drive
# Settings → Resources → File Sharing → Add your project directory
```

### Container won't start

```bash
# View logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Development Workflow

### 1. Development Cycle

```bash
# Start container
docker-compose up -d

# Make code changes on host (your IDE/editor)
# Changes are instantly reflected in container via volume mount

# Test in container
docker-compose exec jetbox pytest

# Run agent
docker-compose exec jetbox python agent.py "implement feature X"
```

### 2. Multiple Tasks

```bash
# Run multiple terminals
# Terminal 1: Run agent
docker-compose exec jetbox python agent.py "task 1"

# Terminal 2: Monitor logs
docker-compose exec jetbox tail -f agent.log

# Terminal 3: Check files
docker-compose exec jetbox ls -la mathx/
```

### 3. Clean Slate

```bash
# Remove all containers, volumes, and start fresh
docker-compose down -v
docker-compose up -d
```

## Advanced Configuration

### Custom Python Packages

Edit `Dockerfile` and add to the `RUN pip install` line:
```dockerfile
RUN pip install --no-cache-dir ollama pytest ruff numpy pandas
```

Then rebuild:
```bash
docker-compose build
```

### Resource Limits

Edit `docker-compose.yml` to add resource constraints:
```yaml
services:
  jetbox:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

### Different Base Image

For smaller size, use alpine:
```dockerfile
FROM python:3.11-alpine
```

For GPU support (if needed later):
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
```

## Tips

1. **Keep container running**: The default CMD keeps it alive for quick exec commands
2. **Share models**: Use host Ollama to avoid duplicating 13GB+ models
3. **Watch resources**: Use `docker stats jetbox` to monitor usage
4. **Logs are king**: Always check `docker-compose logs` if something fails
5. **Clean regularly**: `docker system prune` to free up space

## Example: Full YOLO Session

```bash
# Setup
docker-compose up -d

# Run Claude Code in YOLO mode, point it at Docker
# Claude can now run unrestricted commands via:
docker-compose exec jetbox python agent.py "complex task"
docker-compose exec jetbox bash -c "cd /workspace && <any command>"

# When done, review changes
git status
git diff

# Clean up if needed
docker-compose down
```

---

**Status**: Ready to use
**Tested on**: Windows with Docker Desktop
**Model**: Works with any Ollama model on host
