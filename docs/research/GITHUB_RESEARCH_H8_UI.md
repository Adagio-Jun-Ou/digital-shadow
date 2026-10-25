# Hour 8 Research: UI/Interaction Patterns

## Overview
Research on user interface and interaction patterns for Shadow AI, based on GitHub projects.

---

## Textual (35.2k stars)
**GitHub**: https://github.com/Textualize/textual

### Key Concepts
- Python TUI (Terminal User Interface) framework
- Runs in both terminal and web browser
- Async support built-in
- Widget library (buttons, inputs, trees, tables)
- CSS-based styling
- Command palette with fuzzy search

### Pattern
```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, Button

class ShadowApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    .input {
        margin: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Shadow AI", id="title")
        yield Input(placeholder="Enter query...", id="query")
        yield Button("Submit", id="submit")
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            query = self.query_one(Input).value
            self.run_worker(self.process_query(query))
    
    async def process_query(self, query: str):
        # Async processing
        result = await self.shadow.process(query)
        self.query_one(Static).update(result)

if __name__ == "__main__":
    app = ShadowApp()
    app.run()
```

---

## Bubble Tea (41.2k stars)
**GitHub**: https://github.com/charmbracelet/bubbletea

### Key Concepts
- Go TUI framework based on Elm Architecture
- Model-View-Update (MVU) pattern
- State management
- Command pattern for async I/O
- Composable components

### Pattern (Go)
```go
type model struct {
    choices  []string
    cursor   int
    selected map[int]struct{}
}

func (m model) Init() tea.Cmd {
    return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.KeyMsg:
        switch msg.String() {
        case "ctrl+c", "q":
            return m, tea.Quit
        case "up", "k":
            if m.cursor > 0 {
                m.cursor--
            }
        case "down", "j":
            if m.cursor < len(m.choices)-1 {
                m.cursor++
            }
        case "enter", " ":
            _, ok := m.selected[m.cursor]
            if ok {
                delete(m.selected, m.cursor)
            } else {
                m.selected[m.cursor] = struct{}{}
            }
        }
    }
    return m, nil
}

func (m model) View() string {
    s := "What should we do?\n\n"
    for i, choice := range m.choices {
        cursor := " "
        if m.cursor == i {
            cursor = ">"
        }
        checked := " "
        if _, ok := m.selected[i]; ok {
            checked = "x"
        }
        s += fmt.Sprintf("%s [%s] %s\n", cursor, checked, choice)
    }
    s += "\nPress q to quit.\n"
    return s
}
```

---

## Streamlit (44.1k stars)
**GitHub**: https://github.com/streamlit/streamlit

### Key Concepts
- Rapid prototyping for data apps
- Live editing with instant updates
- Simple Python API
- Built-in widgets (sliders, inputs, charts)
- Automatic state management
- Community Cloud deployment

### Pattern
```python
import streamlit as st

# Sidebar
st.sidebar.title("Shadow AI Settings")
model = st.sidebar.selectbox("Model", ["glm-5", "glm-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Main content
st.title("Shadow AI Chat")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask Shadow AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt, model, temperature))
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Data display
st.dataframe(data)
st.line_chart(data)
```

---

## Key Patterns for Shadow AI

### 1. Terminal Interface (Textual-based)
```python
from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Input, Button,
    DataTable, Tree, TabbedContent, TabPane
)
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive

class ShadowTerminal(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }
    .sidebar {
        dock: left;
        width: 30;
    }
    .chat {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+n", "new_chat", "New Chat"),
        ("ctrl+s", "settings", "Settings"),
    ]
    
    messages: reactive[list] = reactive([])
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(classes="sidebar"):
            yield Static("Shadow AI", id="title")
            yield Button("New Chat", id="new_chat")
            yield Tree("Conversations", id="conversations")
        with Container(classes="main"):
            with TabbedContent():
                with TabPane("Chat"):
                    yield DataTable(id="chat_history")
                    yield Input(placeholder="Type message...", id="input")
                with TabPane("Memory"):
                    yield DataTable(id="memory_blocks")
                with TabPane("Agents"):
                    yield Tree("Agents", id="agents")
        yield Footer()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value:
            await self.process_message(event.value)
            event.input.value = ""
    
    async def process_message(self, message: str):
        self.messages.append({"role": "user", "content": message})
        self.update_chat_display()
        
        # Process with Shadow AI
        response = await self.shadow.process(message)
        
        self.messages.append({"role": "assistant", "content": response})
        self.update_chat_display()
    
    def update_chat_display(self):
        table = self.query_one(DataTable)
        table.clear()
        for msg in self.messages:
            table.add_row(msg["role"], msg["content"])
```

### 2. Web Interface (Streamlit-based)
```python
import streamlit as st
from shadowd import ShadowManager

# Configuration
st.set_page_config(
    page_title="Shadow AI",
    page_icon="🎭",
    layout="wide"
)

# Initialize
if "shadow" not in st.session_state:
    st.session_state.shadow = ShadowManager()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Shadow AI")
    
    model = st.selectbox("Model", ["glm-5", "glm-4"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.divider()
    
    # Memory blocks
    st.subheader("Memory Blocks")
    for block in ["persona", "human", "recent"]:
        with st.expander(block.capitalize()):
            content = st.text_area(
                f"{block}_content",
                value=st.session_state.shadow.memory.core_get(block) or "",
                label_visibility="collapsed"
            )
            if st.button(f"Update {block}"):
                st.session_state.shadow.memory.core_set(block, content)
    
    # Agents
    st.subheader("Agents")
    agents = st.session_state.shadow.crew.agents
    for agent_id in agents.keys():
        st.checkbox(agent_id, value=True)

# Main content
st.title("Chat with Shadow AI")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask Shadow AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(
            st.session_state.shadow.process_stream(prompt, model, temperature)
        )
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Knowledge graph view
with st.expander("Knowledge Graph"):
    graph_data = st.session_state.shadow.knowledge_graph.export_visualization()
    st.graphviz_chart(graph_data)
```

### 3. CLI Interface
```python
from typing import Optional
import sys
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

class ShadowCLI:
    def __init__(self, shadow_manager):
        self.shadow = shadow_manager
        self.console = Console()
        self.session = PromptSession(
            history=FileHistory('.shadow_history'),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter([
                'exit', 'quit', 'clear', 'memory', 'agents',
                'reset', 'save', 'load', 'help'
            ])
        )
        self.running = True
    
    async def run(self):
        self.show_welcome()
        
        while self.running:
            try:
                user_input = await self.session.prompt_async(
                    'shadow> ',
                    multiline=False
                )
                
                await self.process_input(user_input.strip())
                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        self.show_goodbye()
    
    def show_welcome(self):
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold blue]Shadow AI[/bold blue]\n"
            "Your personal AI assistant",
            border_style="blue"
        ))
        
        table = Table(show_header=False)
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        table.add_row("/help", "Show available commands")
        table.add_row("/memory", "View memory blocks")
        table.add_row("/agents", "View active agents")
        table.add_row("/clear", "Clear screen")
        table.add_row("/exit", "Exit Shadow AI")
        self.console.print(table)
    
    async def process_input(self, input_text: str):
        if not input_text:
            return
        
        if input_text.startswith('/'):
            await self.handle_command(input_text[1:])
        else:
            await self.handle_message(input_text)
    
    async def handle_command(self, command: str):
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else None
        
        match cmd:
            case 'exit' | 'quit':
                self.running = False
            case 'clear':
                self.console.clear()
            case 'help':
                self.show_welcome()
            case 'memory':
                await self.show_memory(args)
            case 'agents':
                await self.show_agents()
            case 'reset':
                await self.reset_conversation()
            case _:
                self.console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
    
    async def handle_message(self, message: str):
        with self.console.status("[bold blue]Thinking...[/bold blue]"):
            response = await self.shadow.process(message)
        
        md = Markdown(response)
        self.console.print(Panel(md, border_style="green"))
    
    async def show_memory(self, block_name: Optional[str] = None):
        if block_name:
            content = self.shadow.memory.core_get(block_name)
            self.console.print(Panel(
                content or "[dim]Empty[/dim]",
                title=f"Memory: {block_name}",
                border_style="cyan"
            ))
        else:
            for block in ["persona", "human", "recent"]:
                content = self.shadow.memory.core_get(block)
                if content:
                    self.console.print(Panel(
                        content[:200] + "..." if len(content) > 200 else content,
                        title=block.capitalize(),
                        border_style="cyan"
                    ))
    
    async def show_agents(self):
        table = Table(title="Active Agents")
        table.add_column("Agent", style="cyan")
        table.add_column("Role", style="white")
        table.add_column("Status", style="green")
        
        for agent_id, agent in self.shadow.crew.agents.items():
            table.add_row(
                agent_id,
                agent.role.value,
                "Active" if not agent.current_task else "Busy"
            )
        
        self.console.print(table)
    
    def show_goodbye(self):
        self.console.print(Panel.fit(
            "[bold blue]Goodbye![/bold blue]\n"
            "Thank you for using Shadow AI",
            border_style="blue"
        ))


async def main():
    from shadowd import ShadowManager
    shadow = ShadowManager()
    cli = ShadowCLI(shadow)
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

### Textual (Python TUI)
- Full-featured terminal UI framework
- CSS-based styling
- Widget library with complex controls
- Runs in both terminal and browser
- Good for: Complex desktop TUI applications

### Bubble Tea (Go TUI)
- Elm Architecture (MVU pattern)
- Composable components
- Command-based async
- Good for: Go-based terminal applications

### Streamlit (Python Web)
- Rapid prototyping
- Auto-state management
- Built-in widgets
- Easy deployment
- Good for: Data apps, ML demos, quick prototypes

### Shadow AI UI Strategy
- **CLI**: Simple command-line interface for basic interaction
- **TUI**: Full terminal UI with Textual for advanced features
- **Web**: Streamlit-based web interface for remote access

**Implementation Priority**:
1. CLI for basic usage and scripting
2. TUI for interactive terminal sessions
3. Web interface for non-terminal users