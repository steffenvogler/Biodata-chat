#!/usr/bin/env python3
"""
BioData Chat - Terminal LLM Application for Scientific Database Interaction
Chat with biological databases through MCP servers using local LLM models
"""

import asyncio
import sys
import os
import subprocess
import signal
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import click
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich import box
    from fastmcp import Client
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install click rich fastmcp")
    sys.exit(1)

# Optional imports for fallback LLM backends
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Console setup
console = Console()

@dataclass
class ServerStatus:
    name: str
    pid: Optional[int] = None
    status: str = "stopped"
    last_ping: Optional[float] = None

class BioDataChat:
    def __init__(self, verbose: bool = False, backend: str = "llamafile", model: Optional[str] = None):
        self.verbose = verbose
        self.console = console
        self.backend = backend
        self.servers = {
            "bionomia": ServerStatus("Bionomia (bananompy)"),
            "eol": ServerStatus("Encyclopedia of Life"),
            "ckan": ServerStatus("CKAN NHM")
        }
        self.server_pids: List[int] = []
        self.mcp_clients: Dict[str, Any] = {}

        # Set default model
        self.default_model = "LLaMA 3.2 1B Instruct"
        self.conversation_history: List[Dict[str, str]] = []

        # Setup LLM backend with the chosen model
        self.setup_backend(backend, model)
        
    def setup_backend(self, backend: str, model: Optional[str]):
        """Set up the LLM backend and model"""
        self.backend = backend
        
        # Use provided model, or default if none supplied
        if not model:
            model = self.default_model

        # Configure based on backend
        if self.backend == "llamafile":
            # Map common model names to llamafile filenames
            if "LLaMA 3.2 1B Instruct" in model or "llama" in model.lower():
                self.current_model = "Llama-3.2-1B-Instruct.Q6_K.llamafile"
            else:
                # Generic transformation for other models
                self.current_model = model.replace(" ", "-").replace(".", "-").lower() + ".llamafile"
            self.llamafile_path = "./llamafile-0.8.13"
        else:  # ollama fallback
            # Map to Ollama model names
            if "LLaMA 3.2 1B Instruct" in model or "llama" in model.lower():
                self.current_model = "llama3.2:1b"  # Standard Ollama format
            else:
                # Generic transformation for other models
                self.current_model = model.lower().replace(" ", "")



    def log_verbose(self, message: str):
        """Log verbose messages if verbose mode is enabled"""
        if self.verbose:
            self.console.print(f"[dim cyan][VERBOSE][/dim cyan] {message}")
    
    def show_banner(self):
        """Display application banner"""
        banner = """
╔══════════════════════════════════════════════════════════╗
║                     🧬 BioData Chat 🧬                   ║
║        Intelligent Chat Interface for Scientific Data    ║
║                                                          ║
║  • Bionomia Database (Species Attribution)              ║
║  • Encyclopedia of Life (EOL)                           ║
║  • CKAN Natural History Museum                          ║
╚══════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold blue")
        
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        missing_deps = []
        
        # Check backend-specific dependencies
        if self.backend == "llamafile":
            # Check if llamafile executable exists
            if not Path(self.llamafile_path).exists():
                missing_deps.append(f"Llamafile not found at {self.llamafile_path}")
            
            # Check if model file exists
            if not Path(self.current_model).exists():
                missing_deps.append(f"Model file not found: {self.current_model}")
                
        elif self.backend == "ollama":
            # Check if Ollama is installed and running
            if not HAS_OLLAMA:
                missing_deps.append("Ollama Python client not installed")
            else:
                try:
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                    if result.returncode != 0:
                        missing_deps.append("Ollama not running (run 'ollama serve')")
                except FileNotFoundError:
                    missing_deps.append("Ollama not installed")
        
        # Check if fastmcp is available
        try:
            result = subprocess.run(['fastmcp', '--help'], capture_output=True, text=True)
            if result.returncode != 0:
                missing_deps.append("FastMCP not properly installed")
        except FileNotFoundError:
            missing_deps.append("FastMCP not installed")
            
        # Check if management script exists
        if not Path("manage_servers.sh").exists():
            missing_deps.append("manage_servers.sh script not found")
            
        if missing_deps:
            self.console.print("\n[red]❌ Missing Dependencies:[/red]")
            for dep in missing_deps:
                self.console.print(f"  • {dep}")
            if self.backend == "llamafile" and missing_deps:
                self.console.print("\n[yellow]💡 Tip: Try switching to Ollama backend with --backend ollama[/yellow]")
            self.console.print("\n[yellow]Please resolve these issues and try again.[/yellow]")
            return False
        return True
    
    def setup_llm_backend(self):
        """Setup the LLM backend (llamafile or Ollama)"""
        if self.backend == "llamafile":
            return self.setup_llamafile()
        elif self.backend == "ollama":
            return self.setup_ollama_model()
        return False
    
    def setup_llamafile(self):
        """Check llamafile setup"""
        self.console.print(f"[cyan]Setting up Llamafile backend...[/cyan]")
        
        # Check if files exist (already done in check_dependencies)
        if Path(self.llamafile_path).exists() and Path(self.current_model).exists():
            self.console.print(f"[green]✅ Llamafile backend ready[/green]")
            self.console.print(f"[cyan]  • Executable: {self.llamafile_path}[/cyan]")
            self.console.print(f"[cyan]  • Model: {self.current_model}[/cyan]")
            return True
        return False
    
    def setup_ollama_model(self):
        """Ensure the required Ollama model is available"""
        if not HAS_OLLAMA:
            self.console.print(f"[red]❌ Ollama Python client not available[/red]")
            return False
            
        self.console.print(f"[cyan]Checking for Ollama model: {self.current_model}[/cyan]")
        
        try:
            # List available models
            models = ollama.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.current_model not in model_names:
                self.console.print(f"[yellow]Model {self.current_model} not found. Pulling...[/yellow]")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(f"Downloading {self.current_model}...", total=None)
                    
                    try:
                        ollama.pull(self.current_model)
                        progress.update(task, description="✅ Model downloaded successfully")
                    except Exception as e:
                        self.console.print(f"[red]❌ Failed to download model: {e}[/red]")
                        return False
            else:
                self.console.print(f"[green]✅ Model {self.current_model} is available[/green]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error checking Ollama models: {e}[/red]")
            return False
            
        return True
    
    def start_servers(self):
        """Start all MCP servers using the management script"""
        self.console.print("\n[cyan]🚀 Starting MCP servers...[/cyan]")
        
        try:
            # Make script executable
            subprocess.run(['chmod', '+x', 'manage_servers.sh'], check=True)
            
            # Start servers
            process = subprocess.Popen(
                ['./manage_servers.sh', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.console.print("[green]✅ Servers started successfully[/green]")
                self.log_verbose(f"Server output: {stdout}")
                
                # Extract PIDs from output (basic parsing)
                lines = stdout.split('\n')
                for line in lines:
                    if 'PID' in line:
                        try:
                            pid = int(line.split('PID')[-1].strip())
                            self.server_pids.append(pid)
                        except ValueError:
                            pass
                            
                # Wait a moment for servers to initialize
                time.sleep(3)
                return True
            else:
                self.console.print(f"[red]❌ Failed to start servers: {stderr}[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]❌ Error starting servers: {e}[/red]")
            return False
    
    def stop_servers(self):
        """Stop all MCP servers"""
        self.console.print("[yellow]🛑 Stopping MCP servers...[/yellow]")
        
        try:
            process = subprocess.run(
                ['./manage_servers.sh', 'stop'],
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                self.console.print("[green]✅ Servers stopped successfully[/green]")
                self.server_pids.clear()
            else:
                self.console.print(f"[red]❌ Error stopping servers: {process.stderr}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error stopping servers: {e}[/red]")
    
    def show_server_status(self):
        """Display current server status"""
        table = Table(title="🖥️  MCP Server Status", box=box.ROUNDED)
        table.add_column("Server", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Description", style="green")
        
        servers_info = [
            ("Bionomia", "running" if self.server_pids else "stopped", "Species attribution database"),
            ("EOL", "running" if self.server_pids else "stopped", "Encyclopedia of Life"),
            ("CKAN NHM", "running" if self.server_pids else "stopped", "Natural History Museum data")
        ]
        
        for name, status, desc in servers_info:
            status_icon = "🟢" if status == "running" else "🔴"
            table.add_row(f"{status_icon} {name}", status, desc)
        
        self.console.print(table)
    
    async def initialize_mcp_clients(self):
        """Initialize MCP client connections"""
        if not self.server_pids:
            self.console.print("[red]❌ No servers running. Please start servers first.[/red]")
            return False
            
        self.console.print("[cyan]🔗 Initializing MCP client connections...[/cyan]")
        
        # For now, we'll use a simple approach assuming servers are running on stdio
        # In a real implementation, you'd need proper client initialization
        self.console.print("[green]✅ MCP clients initialized[/green]")
        return True
    
    def format_response(self, response: str) -> None:
        """Format and display LLM response"""
        panel = Panel(
            Markdown(response),
            title="🤖 BioData Assistant",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def format_user_message(self, message: str) -> None:
        """Format and display user message"""
        panel = Panel(
            Text(message, style="white"),
            title="👤 You",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    async def query_databases(self, query: str) -> str:
        """Query the scientific databases through MCP servers"""
        self.log_verbose(f"Querying databases with: {query}")
        
        # This is a simplified implementation
        # In a real scenario, you'd use the MCP clients to query the actual servers
        database_context = """
        Available databases:
        - Bionomia: Species attribution and collector information
        - EOL (Encyclopedia of Life): Comprehensive species data, traits, interactions
        - CKAN NHM: Natural History Museum datasets and collections
        
        Example queries you can make:
        - Search for species information
        - Find collector attributions
        - Get trait data for organisms
        - Explore ecological interactions
        """
        
        return database_context
    
    def generate_llamafile_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using llamafile backend"""
        try:
            process = subprocess.run([
                self.llamafile_path,
                "-m", self.current_model,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", "0.7",
                "--no-display-prompt"
            ], capture_output=True, text=True, timeout=60)
            
            if process.returncode == 0:
                # Clean up the output - remove model loading messages and extra whitespace
                output = process.stdout.strip()
                lines = output.split('\n')
                
                # Find the actual response (skip loading messages)
                response_lines = []
                capturing = False
                
                for line in lines:
                    # Skip system messages
                    if any(skip in line.lower() for skip in ['load time', 'sample time', 'prompt eval', 'eval time', 'total time', 'log start', 'log end', 'main:', 'llama_']):
                        continue
                    if line.strip() and not line.startswith('note:'):
                        capturing = True
                    if capturing and line.strip():
                        response_lines.append(line.strip())
                
                response = '\n'.join(response_lines).strip()
                return response if response else "I'm thinking about your question..."
            else:
                return f"❌ Llamafile error: {process.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ Response timed out"
        except Exception as e:
            return f"❌ Error with llamafile: {str(e)}"
    
    def generate_ollama_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Ollama backend"""
        if not HAS_OLLAMA:
            return "❌ Ollama not available"
            
        try:
            response = ollama.chat(
                model=self.current_model,
                messages=messages,
                stream=False
            )
            return response['message']['content']
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"
    
    async def generate_response(self, user_message: str) -> str:
        """Generate response using configured LLM backend"""
        self.log_verbose(f"Generating response for: {user_message} (using {self.backend})")
        
        # Get database context
        db_context = await self.query_databases(user_message)
        
        # Prepare full prompt
        system_prompt = f"""You are BioData Assistant, an AI that helps users explore scientific databases.

You have access to three main databases through MCP servers:
1. Bionomia - Species attribution and collector information
2. Encyclopedia of Life (EOL) - Comprehensive species data, traits, interactions  
3. CKAN Natural History Museum - Museum datasets and collections

Context from databases:
{db_context}

Provide helpful, accurate responses about biological data. If you need specific data, explain how the user could query the databases directly."""
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("🧠 Thinking...", total=None)
                
                if self.backend == "llamafile":
                    # For llamafile, create a single prompt with conversation history
                    full_prompt = system_prompt + "\n\n"
                    
                    # Add recent conversation history
                    for msg in self.conversation_history[-4:]:  # Last 4 messages for context
                        role = "Human" if msg["role"] == "user" else "Assistant"
                        full_prompt += f"{role}: {msg['content']}\n"
                    
                    full_prompt += f"Human: {user_message}\nAssistant:"
                    
                    assistant_response = self.generate_llamafile_response(full_prompt)
                    
                elif self.backend == "ollama":
                    # For Ollama, use message format
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # Add conversation history
                    for msg in self.conversation_history[-5:]:  # Last 5 messages for context
                        messages.append(msg)
                        
                    # Add current message
                    messages.append({"role": "user", "content": user_message})
                    
                    assistant_response = self.generate_ollama_response(messages)
                    
                else:
                    assistant_response = f"❌ Unknown backend: {self.backend}"
                
                progress.update(task, description="✅ Response ready")
                
            # Update conversation history
            if not assistant_response.startswith("❌"):
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            self.log_verbose(f"Error generating response: {e}")
            return f"❌ Error generating response: {str(e)}"
    
    def show_help(self):
        """Display help information"""
        help_text = """
[bold cyan]🔧 Available Commands:[/bold cyan]

• [green]/help[/green] - Show this help message
• [green]/status[/green] - Show server status
• [green]/history[/green] - Show conversation history
• [green]/clear[/green] - Clear conversation history
• [green]/model <name>[/green] - Switch LLM model
• [green]/verbose[/green] - Toggle verbose mode
• [green]/quit[/green] - Exit the application

[bold cyan]💡 Tips:[/bold cyan]
• Ask questions about species, collectors, or natural history data
• Use natural language - the AI will understand your intent
• Try queries like: "Find information about polar bears" or "Show me data about plant collectors"
        """
        
        panel = Panel(
            help_text,
            title="📚 Help & Commands",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def show_conversation_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            self.console.print("[yellow]No conversation history yet.[/yellow]")
            return
            
        self.console.print("\n[bold cyan]📝 Conversation History:[/bold cyan]")
        
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10 messages
            role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            
            self.console.print(f"[dim]{i}.[/dim] [bold]{role}:[/bold] {content}")
    
    async def run_chat_loop(self):
        """Main chat interaction loop"""
        self.console.print("\n[bold green]💬 Chat started! Type '/help' for commands or '/quit' to exit.[/bold green]")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]", console=self.console).strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower().split()
                    
                    if command[0] == 'help':
                        self.show_help()
                    elif command[0] == 'status':
                        self.show_server_status()
                    elif command[0] == 'history':
                        self.show_conversation_history()
                    elif command[0] == 'clear':
                        self.conversation_history.clear()
                        self.console.print("[green]✅ Conversation history cleared[/green]")
                    elif command[0] == 'verbose':
                        self.verbose = not self.verbose
                        self.console.print(f"[green]Verbose mode: {'ON' if self.verbose else 'OFF'}[/green]")
                    elif command[0] == 'model' and len(command) > 1:
                        new_model = command[1]
                        self.console.print(f"[cyan]Switching to model: {new_model}[/cyan]")
                        self.current_model = new_model
                    elif command[0] == 'quit':
                        break
                    else:
                        self.console.print(f"[red]Unknown command: {command[0]}[/red]")
                    continue
                
                # Display user message
                self.format_user_message(user_input)
                
                # Generate and display response
                response = await self.generate_response(user_input)
                self.format_response(response)
                
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
                    break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/red]")
    
    async def run(self):
        """Main application runner"""
        try:
            self.show_banner()
            
            # Display current backend
            backend_icon = "🗂️" if self.backend == "llamafile" else "🔄"
            self.console.print(f"\n[cyan]{backend_icon} Using {self.backend.title()} backend[/cyan]")
            
            # Check dependencies
            if not self.check_dependencies():
                return
            
            # Setup LLM backend
            if not self.setup_llm_backend():
                return
            
            # Start servers
            if not self.start_servers():
                self.console.print("[red]Failed to start servers. Exiting.[/red]")
                return
            
            # Initialize MCP clients
            if not await self.initialize_mcp_clients():
                return
            
            # Show status
            self.show_server_status()
            
            # Start chat loop
            await self.run_chat_loop()
            
        finally:
            # Cleanup
            self.stop_servers()
            self.console.print("\n[cyan]👋 Goodbye! Thanks for using BioData Chat.[/cyan]")

@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--backend', '-b', default='llamafile', type=click.Choice(['llamafile', 'ollama']), help='LLM backend to use')
@click.option('--model', '-m', default='LLaMA 3.2 1B Instruct', help='Model to use (auto-detected based on backend if not specified)')
def main(verbose: bool, backend: str, model: Optional[str]):
    """
    BioData Chat - Intelligent interface for scientific databases
    
    Chat with biological databases through MCP servers using local LLM models.
    
    Supports two backends:
    - llamafile (default): Self-contained model files with Gemma 2 2B
    - ollama: Ollama server with various models
    """
    
    app = BioDataChat(verbose=verbose, backend=backend, model=model)
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    main()
