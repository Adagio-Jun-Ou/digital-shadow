#!/usr/bin/env python3
"""
Unified Shadow AI v2 - Integration of all research-based implementations
"""

import asyncio
import copy
import re as _re
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.state_graph import StateGraph
from shadowd.block_memory import BlockBasedMemory

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text)


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text)


@dataclass
class ShadowContext:
    query: str
    domain: Optional[str] = None
    user_id: Optional[str] = None
    memory_context: str = ""
    abilities: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.query = _sanitize_content(self.query)
        if self.user_id:
            self.user_id = _sanitize(self.user_id)
        if self.domain:
            self.domain = _sanitize(self.domain)


class UnifiedShadowV2:
    _MAX_KNOWLEDGE_PER_TYPE = 500

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "v2")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.block_memory = BlockBasedMemory(self.data_dir / "blocks")
        self.knowledge: Dict[str, List[Dict]] = {}
        self.workflow = self._create_workflow()
        self._lock = threading.Lock()
        
        logger.info("UnifiedShadowV2 initialized")
    
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(name="shadow_v2")
        
        async def understand(state: Dict) -> Dict:
            query = state.get("query", "")
            domain = self._classify_domain(query)
            return {"domain": domain}
        
        async def recall(state: Dict) -> Dict:
            query = state.get("query", "")
            memory_context = self.block_memory.recall(query)
            self.block_memory.working_add("system", f"Context for: {query}")
            return {"memory_context": memory_context}
        
        async def analyze(state: Dict) -> Dict:
            query = state.get("query", "")
            domain = state.get("domain", "")
            abilities = self._select_abilities(query, domain)
            return {"abilities": abilities}
        
        async def synthesize(state: Dict) -> Dict:
            full_context = self.block_memory.get_full_context()
            return {"synthesis": "Response synthesized"}
        
        workflow.add_node("understand", understand)
        workflow.add_node("recall", recall)
        workflow.add_node("analyze", analyze)
        workflow.add_node("synthesize", synthesize)
        
        workflow.add_edge("understand", "recall")
        workflow.add_edge("recall", "analyze")
        workflow.add_edge("analyze", "synthesize")
        
        workflow.set_entry_point("understand")
        workflow.set_finish_point("synthesize")
        
        return workflow
    
    def _classify_domain(self, query: str) -> str:
        query_lower = query.lower()
        domains = {
            "health": ["health", "doctor", "symptom"],
            "finance": ["money", "budget", "invest"],
            "career": ["work", "job", "meeting"],
        }
        for domain, keywords in domains.items():
            if any(_re.search(r'\b' + _re.escape(kw) + r'\b', query_lower) for kw in keywords):
                return domain
        return "general"
    
    def _select_abilities(self, query: str, domain: str) -> List[str]:
        query_lower = query.lower()
        abilities = []
        if _re.search(r'\b(will|predict)\b', query_lower):
            abilities.append("oracle")
        if _re.search(r'\b(remember|when)\b', query_lower):
            abilities.append("perfect_memory")
        if not abilities:
            abilities.append("omni_sense")
        return abilities
    
    async def ask(self, query: str, user_id: str = None) -> Dict:
        context = ShadowContext(query=query, user_id=user_id)
        result = await self.workflow.execute({
            "query": query,
            "user_id": user_id,
            "memory_context": "",
            "abilities": [],
            "domain": ""
        })
        self.block_memory.working_add("user", query)
        return {
            "query": query,
            "domain": result.get("domain"),
            "abilities": result.get("abilities"),
            "memory_context": result.get("memory_context"),
            "result": result.get("synthesis")
        }
    
    def observe(self, observation: Dict):
        content = _sanitize_content(observation.get("content", str(observation)))
        raw_metadata = observation.get("metadata", {})
        if isinstance(raw_metadata, dict):
            metadata = {str(k)[:100]: _sanitize_content(str(v))[:500] for k, v in list(raw_metadata.items())[:50]}
        else:
            metadata = {}
        memory_id = self.block_memory.archival_insert(content, metadata)
        event_type = _sanitize(observation.get("type", "unknown"))
        with self._lock:
            if event_type not in self.knowledge:
                self.knowledge[event_type] = []
            self.knowledge[event_type].append({
                "id": memory_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": content
            })
            if len(self.knowledge[event_type]) > self._MAX_KNOWLEDGE_PER_TYPE:
                self.knowledge[event_type] = self.knowledge[event_type][-self._MAX_KNOWLEDGE_PER_TYPE:]
        logger.info("Observed: %s", event_type)
        return memory_id
    
    def recall(self, query: str) -> str:
        return self.block_memory.recall(query)
    
    def get_memory_stats(self) -> Dict:
        with self._lock:
            stats = copy.deepcopy(self.block_memory.get_stats())
            stats["knowledge_events"] = sum(len(v) for v in self.knowledge.values())
            return stats
    
    def get_workflow_diagram(self) -> str:
        return self.workflow.visualize()


if __name__ == "__main__":
    import tempfile
    async def test():
        shadow = UnifiedShadowV2(Path(tempfile.mkdtemp()))
        shadow.observe({"type": "commitment", "content": "Call John tomorrow"})
        shadow.observe({"type": "observation", "content": "Had coffee"})
        
        recall = shadow.recall("John")
        print(f"Recall:\n{recall}\n")
        
        result = await shadow.ask("What did I commit to?")
        print(f"Result: {result}\n")
        
        stats = shadow.get_memory_stats()
        print(f"Stats: {stats}\n")
        
        print("Workflow:")
        print(shadow.get_workflow_diagram())
    
    asyncio.run(test())
