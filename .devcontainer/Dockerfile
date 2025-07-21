# Use Ubuntu as base image for better compatibility with Ollama
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    zsh \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create directory for Ollama data
RUN mkdir -p /root/.ollama && chown -R $USERNAME:$USERNAME /root/.ollama

# Set working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements*.txt ./

# Install Python dependencies
RUN pip3 install -r requirements-minimal.txt

# Try to install full requirements, but don't fail if some packages are not available
RUN pip3 install -r requirements.txt || echo "Some optional packages failed to install - continuing with minimal setup"

# Install additional development tools
RUN pip3 install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy

# Switch to non-root user
USER $USERNAME

# Set up zsh for the user
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Create Ollama data directory for the user
RUN mkdir -p /home/$USERNAME/.ollama

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_PORT=11434

# Expose Ollama port
EXPOSE 11434

# Default command
CMD ["/bin/bash"]
