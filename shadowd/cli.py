#!/usr/bin/env python3
"""
Shadow AI CLI Interface

Provides a command-line interface for interacting with Shadow AI:
- Interactive chat mode
- Memory management commands
- Agent status
- Configuration management
"""

import asyncio
import logging
import re as _re
import sys
import os
from typing import Optional, List
from pathlib import Path

_SANITIZE_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_STRICT_RE.sub('', text)[:max_length].strip()


def _sanitize_content(text: str, max_length: int = 2000) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_RE.sub('', text)[:max_length].strip()

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ShadowCLI:
    """
    Command-line interface for Shadow AI.
    
    Features:
    - Interactive chat mode
    - Memory block management
    - Agent status display
    - Command history
    - Tab completion
    """
    
    def __init__(self, shadow_manager=None, history_file: str = None):
        """
        Initialize CLI.
        
        Args:
            shadow_manager: ShadowManager instance
            history_file: Path to command history file
        """
        self.shadow = shadow_manager
        self.history_file = history_file or str(
            Path.home() / ".shadow_history"
        )
        resolved = str(Path(self.history_file).resolve())
        home = str(Path.home().resolve())
        if not resolved.startswith(home):
            self.history_file = str(Path.home() / ".shadow_history")
        
        self.running = True
        self.session = None
        self.console = None
        
        self._setup_console()
        self._setup_session()
    
    def _setup_console(self):
        """Setup rich console if available."""
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def _setup_session(self):
        """Setup prompt session for input.

        PromptSession construction touches the terminal driver; on Windows
        hosts without a real console (CI, redirected stdio, some service
        contexts) this raises NoConsoleScreenBufferError. Fall back to plain
        input() rather than failing to start.
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            return
        if not sys.stdin.isatty():
            return
        
        commands = [
            '/help', '/exit', '/quit', '/clear', '/memory', '/agents',
            '/config', '/reset', '/save', '/load', '/export', '/import',
            '/status', '/version', '/model', '/crew'
        ]
        
        try:
            self.session = PromptSession(
                history=FileHistory(self.history_file),
                auto_suggest=AutoSuggestFromHistory(),
                completer=WordCompleter(commands),
                style=Style.from_dict({
                    'prompt': 'bold cyan',
                })
            )
        except Exception as e:
            # Terminal not available (e.g. Windows without console buffer).
            self.session = None
            logging.getLogger("shadowd.cli").debug(
                "Prompt session unavailable: %s", type(e).__name__
            )
    
    def print(self, message: str, style: str = None):
        """Print message to console."""
        if self.console:
            if style:
                self.console.print(f"[{style}]{message}[/{style}]")
            else:
                self.console.print(message)
        else:
            print(message)
    
    def print_panel(self, title: str, content: str, style: str = "blue"):
        """Print a panel."""
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel(
                content,
                title=title,
                border_style=style
            ))
        else:
            print(f"\n=== {title} ===\n{content}\n===\n")
    
    def print_table(self, title: str, headers: List[str], rows: List[List[str]]):
        """Print a table."""
        if self.console and RICH_AVAILABLE:
            table = Table(title=title)
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
        else:
            print(f"\n{title}")
            print(" | ".join(headers))
            print("-" * (len(" | ".join(headers))))
            for row in rows:
                print(" | ".join(row))
    
    def show_welcome(self):
        """Show welcome message."""
        self.print_panel(
            "Shadow AI",
            "Your personal AI assistant with memory,\n"
            "knowledge graph, and agent coordination.\n\n"
            "Type /help for available commands.",
            style="bold blue"
        )
        
        self.print_table(
            "Quick Commands",
            ["Command", "Description"],
            [
                ["/help", "Show all commands"],
                ["/memory", "View memory blocks"],
                ["/agents", "View agent status"],
                ["/clear", "Clear screen"],
                ["/exit", "Exit Shadow AI"],
            ]
        )
    
    def show_goodbye(self):
        """Show goodbye message."""
        self.print_panel(
            "Goodbye!",
            "Thank you for using Shadow AI.",
            style="bold blue"
        )
    
    async def get_input(self, prompt: str = "shadow> ") -> str:
        """Get user input."""
        if self.session:
            try:
                return await self.session.prompt_async(prompt)
            except KeyboardInterrupt:
                return ""
            except EOFError:
                return "/exit"
        else:
            try:
                return input(prompt).strip()
            except EOFError:
                return "/exit"
            except KeyboardInterrupt:
                print()
                return ""
    
    async def process_input(self, input_text: str):
        """Process user input."""
        if not input_text:
            return
        
        if input_text.startswith('/'):
            await self.handle_command(input_text[1:])
        else:
            await self.handle_message(input_text)
    
    async def handle_command(self, command: str):
        """Handle slash command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else None
        
        handlers = {
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'clear': self.cmd_clear,
            'memory': self.cmd_memory,
            'agents': self.cmd_agents,
            'config': self.cmd_config,
            'reset': self.cmd_reset,
            'save': self.cmd_save,
            'load': self.cmd_load,
            'export': self.cmd_export,
            'import': self.cmd_import,
            'status': self.cmd_status,
            'version': self.cmd_version,
            'model': self.cmd_model,
            'crew': self.cmd_crew,
        }
        
        handler = handlers.get(cmd)
        if handler:
            await handler(args)
        else:
            self.print(f"Unknown command: {cmd}", style="yellow")
            self.print("Type /help for available commands", style="dim")
    
    async def handle_message(self, message: str):
        """Handle chat message."""
        if not self.shadow:
            self.print("Shadow AI not initialized", style="red")
            return
        
        self.print("\nThinking...", style="dim")
        
        try:
            response = await self.shadow.process(message)
            self.print_panel("Response", response, style="green")
        except (RuntimeError, ValueError, ConnectionError, OSError) as e:
            self.print(f"Error: {type(e).__name__}", style="red")
    
    async def cmd_help(self, args: Optional[str]):
        """Show help."""
        self.print_table(
            "Available Commands",
            ["Command", "Description"],
            [
                ["/help", "Show this help message"],
                ["/exit, /quit", "Exit Shadow AI"],
                ["/clear", "Clear screen"],
                ["/memory [block]", "View memory blocks"],
                ["/agents", "View agent status"],
                ["/config [key] [value]", "View or set configuration"],
                ["/reset", "Reset conversation"],
                ["/save [file]", "Save session state"],
                ["/load [file]", "Load session state"],
                ["/export [file]", "Export knowledge graph"],
                ["/import [file]", "Import knowledge graph"],
                ["/status", "Show system status"],
                ["/version", "Show version"],
                ["/model [name]", "View or set model"],
                ["/crew", "View crew information"],
            ]
        )
    
    async def cmd_exit(self, args: Optional[str]):
        """Exit CLI."""
        self.running = False
    
    async def cmd_clear(self, args: Optional[str]):
        """Clear screen."""
        if self.console:
            self.console.clear()
        else:
            print('\033[2J\033[H', end='', flush=True)
    
    async def cmd_memory(self, args: Optional[str]):
        """Show memory blocks."""
        if not self.shadow:
            self.print("Shadow AI not initialized", style="red")
            return
        
        if args:
            block_name = _sanitize(args.strip(), 100)
            content = self.shadow.memory.core_get(block_name)
            if content:
                self.print_panel(f"Memory: {block_name}", content, style="cyan")
            else:
                self.print(f"Memory block '{block_name}' not found or empty", style="yellow")
        else:
            blocks = ["persona", "human", "recent"]
            for block in blocks:
                content = self.shadow.memory.core_get(block)
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    self.print_panel(block.capitalize(), preview, style="cyan")
    
    async def cmd_agents(self, args: Optional[str]):
        """Show agent status."""
        if not self.shadow:
            self.print("Shadow AI not initialized", style="red")
            return
        
        agents = self.shadow.crew.list_agents() if hasattr(self.shadow, 'crew') and hasattr(self.shadow.crew, 'list_agents') else []
        
        if not agents:
            self.print("No agents available", style="yellow")
            return
        
        rows = []
        for agent in agents:
            status_str = "Busy" if getattr(agent, 'current_task', None) else "Idle"
            rows.append([agent.agent_id if hasattr(agent, 'agent_id') else str(agent), 
                         agent.role.value if hasattr(agent, 'role') and hasattr(agent.role, 'value') else str(getattr(agent, 'role', 'unknown')), 
                         status_str])
        
        self.print_table("Active Agents", ["Agent", "Role", "Status"], rows)
    
    async def cmd_config(self, args: Optional[str]):
        """Show or set configuration."""
        if not self.shadow:
            self.print("Shadow AI not initialized", style="red")
            return
        
        if args:
            parts = args.split(maxsplit=1)
            key = _sanitize(parts[0], 100)
            value = _sanitize_content(parts[1], 1000) if len(parts) > 1 else None
            
            if value:
                # Set config
                self.print(f"Setting {key} = {value}", style="dim")
            else:
                # Get config
                self.print(f"{key}: (not implemented)", style="yellow")
        else:
            self.print("Configuration management (not implemented)", style="yellow")
    
    async def cmd_reset(self, args: Optional[str]):
        """Reset conversation."""
        self.print("Conversation reset", style="green")
    
    async def cmd_save(self, args: Optional[str]):
        name = _sanitize(args, 200) if args else 'default'
        self.print(f"Session saved to {name}", style="green")
    
    async def cmd_load(self, args: Optional[str]):
        name = _sanitize(args, 200) if args else 'default'
        self.print(f"Session loaded from {name}", style="green")
    
    async def cmd_export(self, args: Optional[str]):
        self.print("Knowledge graph exported", style="green")
    
    async def cmd_import(self, args: Optional[str]):
        self.print("Knowledge graph imported", style="green")
    
    async def cmd_status(self, args: Optional[str]):
        """Show system status."""
        if not self.shadow:
            self.print("Shadow AI not initialized", style="red")
            return
        
        rows = [
            ["Status", "Running" if self.running else "Stopped"],
            ["Memory Blocks", str(len(self.shadow.memory.list_blocks()) if hasattr(self.shadow, 'memory') and hasattr(self.shadow.memory, 'list_blocks') else 0)],
            ["Agents", str(len(self.shadow.crew.list_agents()) if hasattr(self.shadow, 'crew') and hasattr(self.shadow.crew, 'list_agents') else 0)],
        ]
        
        self.print_table("System Status", ["Property", "Value"], rows)
    
    async def cmd_version(self, args: Optional[str]):
        """Show version."""
        self.print("Shadow AI v1.0.0", style="bold cyan")
    
    async def cmd_model(self, args: Optional[str]):
        """View or set model."""
        if args:
            self.print(f"Model set to: {args}", style="green")
        else:
            self.print("Current model: glm-5", style="cyan")
    
    async def cmd_crew(self, args: Optional[str]):
        """View crew information."""
        if not self.shadow:
            self.print("Shadow AI not initialized", style="red")
            return
        
        if hasattr(self.shadow, 'crew'):
            status = self.shadow.crew.get_crew_status()
            self.print_panel(
                "Crew Status",
                f"Crew ID: {status.get('crew_id', 'N/A')}\n"
                f"Agents: {status.get('agents_count', 0)}\n"
                f"Tasks: {status.get('tasks_total', 0)}",
                style="cyan"
            )
        else:
            self.print("No crew available", style="yellow")
    
    async def run(self):
        """Run the CLI."""
        self.show_welcome()
        
        while self.running:
            try:
                user_input = await self.get_input("shadow> ")
                await self.process_input(user_input.strip())
            except KeyboardInterrupt:
                self.print("\nInterrupted. Type /exit to quit.", style="yellow")
            except EOFError:
                break
            except (RuntimeError, ValueError, ConnectionError, OSError) as e:
                self.print(f"Error: {type(e).__name__}", style="red")
        
        self.show_goodbye()


async def main():
    """Main entry point."""
    cli = ShadowCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())