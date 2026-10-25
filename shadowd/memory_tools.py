#!/usr/bin/env python3
"""
Letta-style Memory Tools for Shadow AI.

Implements the v2 memory editing protocol from Letta/MemGPT:
- memory_replace: Find and replace exact strings in memory blocks
- memory_insert: Insert content at a specific position
- memory_rethink: Full rewrite of a memory block
- memory_finish_edits: Signal completion of editing session

Also implements XML rendering for core memory blocks with metadata,
following the Letta pattern of showing char counts and limits.
"""

import json
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("shadowd.memory_tools")

_XML_TAG_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()


def _sanitize_strict(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()


def _sanitize_label(label: str) -> str:
    """Sanitize a block label for use as an XML tag name.
    Strip any characters that aren't alphanumeric, underscore, or hyphen.
    Replace invalid chars with underscore."""
    if not label:
        return "block"
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', label)
    if not sanitized or sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized


class MemoryEditResult(BaseModel):
    success: bool
    block_label: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    error: Optional[str] = None
    chars_used: int = 0
    chars_limit: int = 0


def render_memory_xml(blocks, include_metadata: bool = True) -> str:
    """
    Render memory blocks as structured XML following the Letta pattern.
    
    Shows each block with its label, description, character budget,
    and current content. This gives the LLM full visibility into
    what it can edit and how much room it has.
    """
    parts = ["<memory_blocks>"]
    parts.append("The following memory blocks are currently engaged in your core memory unit:\n")
    
    for label, block in blocks.items():
        safe_label = _sanitize_label(label)
        parts.append(f"<{safe_label}>")
        if include_metadata and hasattr(block, 'description') and block.description:
            parts.append(f"<description>\n{block.description}\n</description>")
        if include_metadata:
            chars = len(block.value) if hasattr(block, 'value') and block.value else 0
            limit = block.limit if hasattr(block, 'limit') else 2000
            read_only = block.protected if hasattr(block, 'protected') else False
            parts.append("<metadata>")
            parts.append(f"  read_only={'true' if read_only else 'false'}")
            parts.append(f"  chars_current={chars}")
            parts.append(f"  chars_limit={limit}")
            parts.append("</metadata>")
        if hasattr(block, 'value') and block.value:
            parts.append(f"<value>\n{block.value}\n</value>")
        parts.append(f"</{safe_label}>\n")
    
    parts.append("</memory_blocks>")
    return "\n".join(parts)


class MemoryTools:
    """
    Letta-style memory editing tools for Shadow AI.
    
    Provides granular editing primitives for memory blocks:
    - replace: Precise find-and-replace with uniqueness validation
    - insert: Insert content at a specific position
    - rethink: Full rewrite of a block for reorganization
    - finish: Signal that editing session is complete
    """
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self._lock = threading.Lock()
    
    def memory_replace(self, label: str, old_string: str, new_string: str) -> MemoryEditResult:
        """
        Replace an exact string in a memory block.
        
        Validates that old_string appears exactly once to prevent
        accidental multi-match replacements.
        """
        label = _sanitize_strict(label)
        with self._lock:
            block = self.memory.blocks.get(label)
            if block is None:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error="Block not found"
                )
            
            if block.protected:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error="Block is read-only and cannot be modified."
                )
            
            count = block.value.count(old_string)
            if count == 0:
                nearby = self._find_similar_strings(old_string, block.value)
                suggestion = f" Did you mean one of: {nearby}?" if nearby else ""
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error=f"String not found in block.{suggestion}"
                )
            if count > 1:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    old_value=old_string,
                    error=f"String appears {count} times in block. Provide more context to make it unique."
                )
            
            old_value = block.value
            new_value = old_value.replace(old_string, new_string, 1)
            
            if len(new_value) > block.limit:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    chars_used=len(new_value),
                    chars_limit=block.limit,
                    error=f"Replacement would exceed block limit ({len(new_value)}/{block.limit} chars). "
                          f"Condense the content or use memory_rethink to reorganize."
                )
            
            block.value = new_value
            self.memory.save()
            
            return MemoryEditResult(
                success=True,
                block_label=label,
                old_value=old_string,
                new_value=new_string,
                chars_used=len(new_value),
                chars_limit=block.limit
            )
    
    def memory_insert(self, label: str, new_string: str, insert_line: int = -1) -> MemoryEditResult:
        """Insert content into a memory block at a specific line position."""
        label = _sanitize_strict(label)
        new_string = _sanitize_content(new_string)
        with self._lock:
            block = self.memory.blocks.get(label)
            if block is None:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error="Block not found"
                )
            
            if block.protected:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error="Block is read-only and cannot be modified."
                )
            
            lines = block.value.split('\n') if block.value else []
            
            if insert_line == -1 or insert_line >= len(lines):
                new_value = block.value + ('\n' if block.value else '') + new_string
            elif insert_line < 0:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error=f"Invalid insert_line: {insert_line}. Must be >= 0 or -1 for append."
                )
            else:
                lines.insert(insert_line, new_string)
                new_value = '\n'.join(lines)
            
            if len(new_value) > block.limit:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    chars_used=len(new_value),
                    chars_limit=block.limit,
                    error=f"Insertion would exceed block limit ({len(new_value)}/{block.limit} chars)."
                )
            
            old_value = block.value
            block.value = new_value
            self.memory.save()
            
            return MemoryEditResult(
                success=True,
                block_label=label,
                old_value=old_value,
                new_value=new_string,
                chars_used=len(new_value),
                chars_limit=block.limit
            )
    
    def memory_rethink(self, label: str, new_memory: str) -> MemoryEditResult:
        """Completely rewrite a memory block."""
        label = _sanitize_strict(label)
        new_memory = _sanitize_content(new_memory)
        with self._lock:
            block = self.memory.blocks.get(label)
            if block is None:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error="Block not found"
                )
            
            if block.protected:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    error="Block is read-only and cannot be modified."
                )
            
            if len(new_memory) > block.limit:
                return MemoryEditResult(
                    success=False,
                    block_label=label,
                    chars_used=len(new_memory),
                    chars_limit=block.limit,
                    error=f"New content exceeds block limit ({len(new_memory)}/{block.limit} chars)."
                )
            
            old_value = block.value
            block.value = new_memory
            self.memory.save()
            
            return MemoryEditResult(
                success=True,
                block_label=label,
                old_value=old_value,
                new_value=new_memory,
                chars_used=len(new_memory),
                chars_limit=block.limit
            )
    
    def memory_finish_edits(self) -> Dict:
        """Signal that the editing session is complete. Returns block summary."""
        with self._lock:
            summary = {}
            for label, block in self.memory.blocks.items():
                summary[label] = {
                    "chars": len(block.value) if block.value else 0,
                    "limit": block.limit,
                    "protected": block.protected if hasattr(block, 'protected') else False,
                }
            return {"status": "editing_complete", "blocks": summary}
    
    def _find_similar_strings(self, target: str, content: str, max_suggestions: int = 3) -> List[str]:
        """Find similar strings in content for helpful error messages."""
        target_words = set(target.lower().split())
        candidates = []
        for line in content.split('\n'):
            line_words = set(line.lower().split())
            overlap = len(target_words & line_words)
            if overlap > 0 and overlap >= len(target_words) * 0.5:
                candidates.append(line.strip()[:80])
        return candidates[:max_suggestions]


class ContextWindowBudget:
    """
    Track and manage the LLM context window budget.
    
    Sections:
    - system_prompt: Base instructions, personality
    - core_memory: Memory blocks (rendered as XML)
    - tools: Tool definitions and schemas
    - messages: Conversation history
    - observations: Current environmental context
    """
    
    def __init__(self, max_context_tokens: int = 8192):
        self.max_context_tokens = max(1, max_context_tokens)
        self.summarization_threshold = 0.9
        self.section_budgets = {
            "system_prompt": int(max_context_tokens * 0.10),
            "core_memory": int(max_context_tokens * 0.20),
            "tools": int(max_context_tokens * 0.10),
            "messages": int(max_context_tokens * 0.45),
            "observations": int(max_context_tokens * 0.15),
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)
    
    def get_budget_overview(self, system_prompt: str = "", memory_xml: str = "",
                           tool_definitions: str = "", messages: list = None,
                           observations: list = None) -> Dict:
        """Get a budget overview showing token usage per section."""
        messages = messages or []
        observations = observations or []
        
        sections = {
            "system_prompt": self.estimate_tokens(system_prompt),
            "core_memory": self.estimate_tokens(memory_xml),
            "tools": self.estimate_tokens(tool_definitions),
            "messages": sum(self.estimate_tokens(m.get("content", "")) for m in messages),
            "observations": sum(self.estimate_tokens(json.dumps(o)) for o in observations),
        }
        
        total = sum(sections.values())
        budget = self.section_budgets
        
        return {
            "total_tokens_used": total,
            "max_context_tokens": self.max_context_tokens,
            "utilization": total / self.max_context_tokens,
            "should_summarize": total > self.max_context_tokens * self.summarization_threshold,
            "sections": {
                name: {
                    "tokens_used": sections[name],
                    "tokens_budget": budget[name],
                    "utilization": sections[name] / budget[name] if budget[name] > 0 and sections[name] > 0 else 0,
                }
                for name in sections
            }
        }
    
    def trim_messages(self, messages: list, target_tokens: int) -> list:
        """
        Trim conversation messages to fit within budget.
        
        Keeps the first message (system) and last N messages,
        replacing the middle with a summary placeholder.
        """
        if not messages:
            return messages
        
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        conversation = messages[1:] if system else messages
        
        total = sum(self.estimate_tokens(m.get("content", "")) for m in conversation)
        if total <= target_tokens:
            return messages
        
        trimmed = []
        remaining_budget = target_tokens
        
        # Keep messages from the end (most recent context)
        for msg in reversed(conversation):
            msg_tokens = self.estimate_tokens(msg.get("content", ""))
            if remaining_budget >= msg_tokens:
                trimmed.insert(0, msg)
                remaining_budget -= msg_tokens
            else:
                break
        
        if system and trimmed:
            trimmed.insert(0, {
                "role": "system",
                "content": system.get("content", "") + 
                    f"\n\n[Note: {len(conversation) - len(trimmed)} earlier messages were condensed to fit context window.]"
            })
        
        return trimmed