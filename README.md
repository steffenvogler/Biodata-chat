# 🧬 BioData Chat

**Intelligent Terminal Interface for Scientific Database Interaction**

BioData Chat is a sophisticated terminal application that enables natural language interaction with multiple scientific databases through MCP (Model Context Protocol) servers, powered by local LLM models for privacy and efficiency.

## 🌟 Overview

This application provides seamless access to three major scientific databases:

- **🔬 Bionomia Database** - Species attribution and collector information
- **🌍 Encyclopedia of Life (EOL)** - Comprehensive species data, traits, and ecological interactions  
- **🏛️ CKAN Natural History Museum** - Museum datasets and collections

## ✨ Features

### 🎨 User Experience
- **Rich Terminal Interface** - Beautiful, colorful CLI using the `rich` library
- **Interactive Chat Mode** - Natural language conversations with scientific data
- **Status Monitoring** - Real-time server status and health checks
- **Progress Indicators** - Visual feedback for long-running operations

### 🔧 Technical Capabilities
- **MCP Server Integration** - Orchestrates multiple database servers automatically
- **Local LLM Processing** - Uses Ollama for privacy-focused AI responses
- **Resource Efficiency** - Optimized for local execution with lightweight models
- **Verbose Logging** - Detailed debugging information when needed
- **Error Handling** - Graceful error recovery and user-friendly messages

### 🎯 Smart Features
- **Conversation History** - Maintains context across chat sessions
- **Model Switching** - Change LLM models on-the-fly
- **Dependency Checking** - Automatic validation of required components
- **Server Management** - Integrated start/stop/restart functionality

## 🚀 Quick Start

### Prerequisites

1. **Ollama** - Install from [ollama.ai](https://ollama.ai)
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **FastMCP** - Should be available in your environment
3. **Python 3.8+** with pip

### Installation

#### Option 1: Development Container (Recommended)

The easiest way to get started is using the pre-configured development container:

1. **Prerequisites:**
   - [Docker](https://www.docker.com/get-started) installed
   - [VS Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in container:**
   ```bash
   git clone https://github.com/steffenvogler/Biodata-chat.git
   cd Biodata-chat
   code .
   ```
   Then click "Reopen in Container" when prompted.

3. **Everything is pre-configured:**
   - ✅ Python 3.11 with all dependencies
   - ✅ Ollama pre-installed and configured
   - ✅ Development tools ready
   - ✅ VS Code extensions loaded

#### Option 2: Local Installation

1. **Clone and navigate to the project directory:**
   ```bash
   cd /path/to/Biodata-chat
   ```

2. **Install initial minimal dependencies:**
   ```bash
   pip install -r requirements-minimal.txt
   ```

3. **Install full dependencies if needed:**
   ```bash
   pip install -r requirements.txt
   ```
   For machines with dependency issues, try only minimal requirements first.

4. **Make scripts executable:**
   ```bash
   chmod +x biodata_chat.py
   chmod +x manage_servers.sh
   ```

### Running the Application

1. **Basic usage:**
   ```bash
   ./biodata_chat.py
   ```

2. **With verbose logging:**
   ```bash
   ./biodata_chat.py --verbose
   ```

3. **Using a specific model:**
   ```bash
   ./biodata_chat.py --model llama3.2:3b
   ```

## 📖 Usage Guide

### Starting a Chat Session

When you run the application, it will:

1. 🔍 Check all dependencies
2. 📥 Download required LLM model (if needed)
3. 🚀 Start MCP servers automatically
4. 🔗 Initialize database connections
5. 💬 Launch interactive chat mode

### Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands and usage tips |
| `/status` | Display current MCP server status |
| `/history` | Show recent conversation history |
| `/clear` | Clear conversation history |
| `/model <name>` | Switch to a different Ollama model |
| `/verbose` | Toggle verbose logging on/off |
| `/quit` | Exit the application safely |

### Example Queries

Try asking questions like:

- 🐻 *"Find information about polar bears"*
- 🌸 *"Show me data about plant collectors"*
- 🦋 *"What ecological interactions involve butterflies?"*
- 📊 *"Search for datasets about marine biodiversity"*
- 👨‍🔬 *"Who collected specimens in Madagascar?"*

## 🏗️ Architecture

### MCP Servers

The application manages three specialized MCP servers:

1. **Bionomia Server** (`bananompy_server.py`)
   - Collector search and attribution
   - Species occurrence data

2. **EOL Server** (`eol_server.py`)
   - Species information and media
   - Taxonomic hierarchies
   - Ecological traits and interactions

3. **CKAN Server** (`ckan_server.py`)
   - Dataset discovery
   - Collection browsing
   - Resource metadata

### Server Management

The `manage_servers.sh` script provides:
- **Start:** Launch all MCP servers
- **Stop:** Gracefully shut down servers  
- **Restart:** Stop and restart all servers
- **Pause:** Suspend server processes
- **Continue:** Resume suspended servers

## 🔧 Configuration

### Model Selection

The application defaults to `llama3.2:1b` for resource efficiency. You can specify other models:

```bash
# Larger, more capable model
./biodata_chat.py --model llama3.2:3b

# Specialized models
./biodata_chat.py --model codellama:7b
```

### Verbose Mode

Enable detailed logging for troubleshooting:

```bash
./biodata_chat.py --verbose
```

Or toggle during runtime with `/verbose` command.

## 🛠️ Development

### File Structure

```
fastmcp/
├── biodata_chat.py              # Main application
├── manage_servers.sh            # Server orchestration script
├── requirements.txt             # Python dependencies
├── src/fastmcp/
│   ├── bananompy_server.py     # Bionomia MCP server
│   ├── eol_server.py           # EOL MCP server
│   └── ckan_server.py          # CKAN MCP server
└── BIODATA_CHAT_README.md      # This documentation
```

### Adding New Databases

To integrate additional scientific databases:

1. Create a new MCP server script in `src/fastmcp/`
2. Add server management to `manage_servers.sh`
3. Update the client initialization in `biodata_chat.py`
4. Extend the system prompt with database context

## 🐛 Troubleshooting

### Common Issues

**Ollama not responding:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

**MCP servers won't start:**
```bash
# Check script permissions
chmod +x manage_servers.sh

# Test manually
./manage_servers.sh start
```

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Error Messages

The application provides detailed error messages with suggested solutions. Use `--verbose` flag for additional diagnostic information.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 🔌 Additional database integrations
- 🎨 Enhanced UI components
- 🧠 Better query understanding
- 📊 Data visualization features
- 🔍 Advanced search capabilities

## 📄 License

This project extends FastMCP and follows its licensing terms. Please refer to the FastMCP license for details.

## 🙏 Acknowledgments

- **FastMCP** - Provides the MCP server framework
- **Ollama** - Local LLM execution platform
- **Rich** - Beautiful terminal interfaces
- **Scientific Databases** - Bionomia, EOL, and CKAN for their open APIs

---

*Built with ❤️ for the scientific research community*
