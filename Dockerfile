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

# Set working directory
WORKDIR /workspace

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ollama pytest ruff

# Copy project files
COPY . .

# Install the project in editable mode if pyproject.toml exists
RUN if [ -f "pyproject.toml" ]; then pip install -e .; fi

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Create volume mount point for persistent data
VOLUME ["/workspace"]

# Default command: start Ollama service and wait
CMD ["bash", "-c", "echo 'Container ready. Use: docker exec -it jetbox python agent.py \"your task\"' && tail -f /dev/null"]
