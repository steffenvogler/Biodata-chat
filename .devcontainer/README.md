# BioData Chat Development Container

This directory contains the development container configuration for BioData Chat, providing a consistent development environment with all necessary dependencies pre-installed.

## Features

- **Ubuntu 22.04** base image for better Ollama compatibility
- **Python 3.11** with all project dependencies
- **Ollama** pre-installed and configured
- **VS Code extensions** for Python development
- **Development tools** (pytest, black, flake8, etc.)
- **Separate Ollama service** for better resource management

## Quick Start

### Option 1: VS Code with Dev Containers Extension

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
2. Open the project folder in VS Code
3. Click "Reopen in Container" when prompted, or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
4. Wait for the container to build and initialize

### Option 2: Docker Compose (Manual)

1. Build and start the services:
   ```bash
   cd .devcontainer
   docker-compose up -d
   ```

2. Connect to the development container:
   ```bash
   docker-compose exec biodata-chat /bin/zsh
   ```

## Using Ollama in the Container

The devcontainer includes both a local Ollama installation and a separate Ollama service:

### Local Ollama (in development container)
```bash
# Start Ollama service
ollama serve &

# Pull a model
ollama pull tinyllama

# Test the model
ollama run tinyllama "Hello, world!"
```

### Separate Ollama Service
The Docker Compose setup includes a dedicated Ollama service that's automatically started:

```bash
# Pull models using the service
curl http://ollama:11434/api/pull -d '{"name":"tinyllama"}'

# Or configure your application to use: http://ollama:11434
```

## Development Workflow

1. **Start the devcontainer** using one of the methods above
2. **Install dependencies** (done automatically via postCreateCommand):
   ```bash
   pip install -r requirements.txt
   chmod +x biodata_chat.py manage_servers.sh
   ```
3. **Pull an Ollama model**:
   ```bash
   ollama pull tinyllama
   ```
4. **Run the application**:
   ```bash
   ./biodata_chat.py
   ```

## Ports

The following ports are forwarded from the container:

- **8000-8002**: MCP server ports
- **11434**: Ollama API (development container)
- **11435**: Ollama service (separate container)

## Persistent Data

- **Ollama models** are stored in a Docker volume (`ollama-data`) and persist between container rebuilds
- **Source code** is mounted from the host, so changes are immediately reflected

## Troubleshooting

### Ollama Connection Issues
If you can't connect to Ollama:

1. Check if the service is running:
   ```bash
   docker-compose ps
   ```

2. Check Ollama logs:
   ```bash
   docker-compose logs ollama
   ```

3. Test connection:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Python Dependencies
If some packages fail to install, the container will continue with minimal dependencies. You can manually install missing packages:

```bash
pip install package-name
```

### Container Rebuild
If you need to rebuild the container completely:

```bash
# VS Code: Ctrl+Shift+P → "Dev Containers: Rebuild Container"

# Docker Compose:
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## GPU Support

To enable GPU support for Ollama (NVIDIA only):

1. Uncomment the GPU configuration in `docker-compose.yml`
2. Ensure you have the NVIDIA Container Toolkit installed on the host
3. Rebuild the containers

## Environment Variables

- `OLLAMA_HOST=0.0.0.0` - Makes Ollama accessible from other containers
- `OLLAMA_PORT=11434` - Default Ollama port

## Extensions and Settings

The devcontainer automatically installs useful VS Code extensions:

- Python support and formatting
- Linting and type checking
- GitHub Copilot (if available)
- JSON and YAML support

All Python files are automatically formatted with Black on save, and imports are organized with isort.
