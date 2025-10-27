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
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ollama pytest ruff pyyaml requests

# Create non-root user for agent execution
RUN useradd -m -u 1000 -s /bin/bash agent && \
    mkdir -p /workspace/.agent_workspace /workspace/.agent_context && \
    chown -R agent:agent /workspace

# Set working directory
WORKDIR /workspace

# Copy project files
COPY --chown=agent:agent . .

# Install the project in editable mode if pyproject.toml exists
RUN if [ -f "pyproject.toml" ]; then pip install -e .; fi

# Switch to non-root user
USER agent

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Create volume mount point for persistent data
VOLUME ["/workspace"]

# Default command: start Ollama service and wait
CMD ["bash", "-c", "echo 'Container ready. Use: docker exec -it jetbox python agent.py \"your task\"' && tail -f /dev/null"]
