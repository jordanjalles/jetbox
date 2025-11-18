# Dockerfile for jetbox coding agent
# Provides isolated environment for safe YOLO mode execution

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python packages (before creating user)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ollama pytest ruff pyyaml && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create non-root user for agent execution
RUN useradd -m -u 1000 -s /bin/bash agent && \
    mkdir -p /workspace/.agent_workspace /workspace/.agent_context && \
    mkdir -p /home/agent/.jetbox/logs && \
    chown -R agent:agent /workspace /home/agent/.jetbox

# Set working directory
WORKDIR /workspace

# Copy project files
COPY --chown=agent:agent . .

# Install the project in editable mode if pyproject.toml exists
RUN if [ -f "pyproject.toml" ]; then pip install -e .; fi

# Make jetbox.py executable
RUN chmod +x jetbox.py 2>/dev/null || true

# Switch to non-root user
USER agent

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Create volume mount point for persistent data
VOLUME ["/workspace"]

# Default command: start Ollama service and wait
CMD ["bash", "-c", "echo 'Container ready. Use: docker exec -it jetbox python agent.py \"your task\"' && tail -f /dev/null"]
