#!/usr/bin/env python3
"""
Shadow AI Enhanced - Life Optimization with RAG Memory
Integrates insights from:
- LaminaAI: RAG, summarization, semantic embeddings
- Digital Twin Template: 7 life domains organization
- LifeBuddy: Ambient intelligence, context awareness
"""

import asyncio
import copy
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging

from .memory_rag import RAGMemorySystem, RAGContext
from .atomic_io import fsync_dir
from shikigami import ShikigamiRegistry, ShikigamiType

logger = logging.getLogger("shadowd.shadow_enhanced")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_CONTENT_LENGTH = 64000
_MAX_RECORD_CONTENT = 500
_MAX_QUESTION_LENGTH = 10000
_MAX_PEOPLE = 500
_MAX_DECISIONS = 1000
_MAX_PATTERNS = 500
_MAX_PEOPLE_DEVICES = 50


def _sanitize_content(text: str, max_length: int = _MAX_CONTENT_LENGTH) -> str:
    if not isinstance(text, str):
        return ""
    text = _SANITIZE_CONTENT_RE.sub('', text).strip()
    return text[:max_length]


def _sanitize(text: str, max_length: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    text = _SANITIZE_STRICT_RE.sub('', text).strip()
    return text[:max_length]


@dataclass
class LifeDomain:
    """One of 7 life domains for organized memory."""
    name: str
    description: str
    records: List[Dict] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    goals: List[Dict] = field(default_factory=list)


class EnhancedShadowAI:
    """
    Enhanced Shadow AI with:
    - RAG Memory (LaminaAI style)
    - 7 Life Domains (Digital Twin style)
    - Ambient Intelligence (LifeBuddy style)
    - 10 Unified Abilities (Original Shadow)
    """
    
    # 7 Life Domains (from Digital Twin Template)
    DOMAIN_HEALTH = "health"
    DOMAIN_FINANCE = "finance"
    DOMAIN_LEGAL = "legal"
    DOMAIN_GROWTH = "growth"
    DOMAIN_CAREER = "career"
    DOMAIN_HOME = "home"
    DOMAIN_ADMIN = "admin"
    
    _MAX_RECORDS_PER_DOMAIN = 1000
    _MAX_GOALS_PER_DOMAIN = 100
    _MAX_INSIGHTS_QUEUE = 500

    def __init__(self, shikigami_registry: ShikigamiRegistry, mind_engine, data_dir: Path = None):
        self.registry = shikigami_registry
        self.mind = mind_engine
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data")
        
        # RAG Memory System
        self.memory = RAGMemorySystem(self.data_dir / "memory")
        self.rag_context = RAGContext(self.memory)
        
        # 7 Life Domains
        self.domains = {
            self.DOMAIN_HEALTH: LifeDomain(
                name="Health",
                description="Medical records, fitness, wellness, symptoms"
            ),
            self.DOMAIN_FINANCE: LifeDomain(
                name="Finance",
                description="Budgets, investments, expenses, tax"
            ),
            self.DOMAIN_LEGAL: LifeDomain(
                name="Legal",
                description="Contracts, agreements, legal matters"
            ),
            self.DOMAIN_GROWTH: LifeDomain(
                name="Personal Growth",
                description="Goals, learning, skills, habits"
            ),
            self.DOMAIN_CAREER: LifeDomain(
                name="Career & Education",
                description="Work, projects, education, career goals"
            ),
            self.DOMAIN_HOME: LifeDomain(
                name="Home & Living",
                description="Property, maintenance, household"
            ),
            self.DOMAIN_ADMIN: LifeDomain(
                name="Admin & Planning",
                description="Schedules, admin tasks, planning"
            )
        }
        
        # Track people, decisions, patterns (with lock for thread safety)
        self._lock = threading.Lock()
        self.people = {}
        self.decisions = []
        self.patterns = {}
        
        # Current context (ambient)
        self.current_context = {
            "location": None,
            "people": [],
            "activity": None,
            "domain": None
        }
        
        # Proactive insights queue
        self.insights_queue = []
    
    def _cap_insights_queue(self):
        if len(self.insights_queue) > self._MAX_INSIGHTS_QUEUE:
            self.insights_queue = self.insights_queue[-self._MAX_INSIGHTS_QUEUE:]

    def _cap_people(self):
        if len(self.people) > _MAX_PEOPLE:
            oldest = list(self.people.keys())[:len(self.people) - _MAX_PEOPLE]
            for k in oldest:
                del self.people[k]
                logger.warning("People evicted: %s", k[:8])

    def _cap_decisions(self):
        if len(self.decisions) > _MAX_DECISIONS:
            self.decisions = self.decisions[-_MAX_DECISIONS:]

    def _cap_patterns(self):
        if len(self.patterns) > _MAX_PATTERNS:
            oldest = list(self.patterns.keys())[:len(self.patterns) - _MAX_PATTERNS]
            for k in oldest:
                del self.patterns[k]
                logger.warning("Pattern evicted: %s", k[:8])

    def _validate_domain_key(self, domain_key: str) -> str:
        sanitized = _sanitize(domain_key, max_length=50)
        valid_domains = {self.DOMAIN_HEALTH, self.DOMAIN_FINANCE, self.DOMAIN_LEGAL,
                        self.DOMAIN_GROWTH, self.DOMAIN_CAREER, self.DOMAIN_HOME, self.DOMAIN_ADMIN}
        return sanitized if sanitized in valid_domains else ""

    def _load_domains(self):
        """Load domain data from disk."""
        for domain_key in self.domains:
            safe_key = self._validate_domain_key(domain_key)
            if not safe_key:
                continue
            domain_file = self.data_dir / "memory" / f"domain_{safe_key}.json"
            if domain_file.exists():
                try:
                    data = json.loads(domain_file.read_text())
                    self.domains[domain_key].records = data.get("records", [])
                    self.domains[domain_key].insights = data.get("insights", [])
                    self.domains[domain_key].goals = data.get("goals", [])
                except (json.JSONDecodeError, OSError, ValueError):
                    pass
    
    async def observe(self, event: Dict):
        """
        Observe an event and route to appropriate systems.
        
        Routes to:
        - Memory (RAG)
        - Life domains
        - Proactive analysis
        - Shikigami (abilities)
        """
        content_str = json.dumps(event).lower() if isinstance(event, dict) else _sanitize_content(str(event))
        domain = self._classify_domain(event, content_str=content_str)

        event_content = _sanitize_content(content_str)[:_MAX_CONTENT_LENGTH]

        try:
            await self.memory.add_observation(
                content=event_content,
                metadata={"type": _sanitize(str(event.get("type", "unknown"))) if isinstance(event, dict) else "unknown"},
                domain=domain
            )
        except (ValueError, TypeError, OSError, RuntimeError) as e:
            logger.error("Memory add_observation failed: %s", type(e).__name__)

        # Add to domain (thread-safe)
        if domain:
            with self._lock:
                self.domains[domain].records.append({
                    "content": _sanitize_content(event_content, max_length=_MAX_RECORD_CONTENT),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                if len(self.domains[domain].records) > self._MAX_RECORDS_PER_DOMAIN:
                    self.domains[domain].records = self.domains[domain].records[-self._MAX_RECORDS_PER_DOMAIN:]

        # Update current context (thread-safe)
        self._update_context(event, content_str=content_str)

        # Check for proactive insights
        try:
            await self._check_insights(event, content_str=content_str)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Insights check failed: %s", type(e).__name__)
    
    def _classify_domain(self, event: Dict, content_str: str = None) -> Optional[str]:
        """Classify event into life domain (multi-label)."""
        content_lower = content_str or (json.dumps(event).lower() if isinstance(event, dict) else _sanitize_content(str(event)).lower())
        
        domain_keywords = {
            self.DOMAIN_HEALTH: ["health", "doctor", "symptom", "fitness", "workout", "exercise", "medical", "wellness"],
            self.DOMAIN_FINANCE: ["money", "budget", "expense", "invest", "tax", "salary", "financial", "payment"],
            self.DOMAIN_LEGAL: ["contract", "legal", "law", "agreement", "sign", "court", "attorney"],
            self.DOMAIN_GROWTH: ["learn", "goal", "skill", "habit", "book", "course", "meditation", "growth"],
            self.DOMAIN_CAREER: ["work", "project", "meeting", "boss", "deadline", "career", "job", "education"],
            self.DOMAIN_HOME: ["home", "house", "apartment", "repair", "clean", "maintenance", "pet", "family"],
            self.DOMAIN_ADMIN: ["schedule", "appointment", "task", "todo", "plan", "calendar", "admin"],
        }
        
        scored = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', content_lower))
            if score > 0:
                scored[domain] = score
        
        if not scored:
            return None
        
        return max(scored, key=scored.get)
    
    def _classify_domains_multi(self, event: Dict) -> List[str]:
        """Classify event into multiple life domains (multi-label)."""
        content_str = json.dumps(event).lower() if isinstance(event, dict) else _sanitize_content(str(event)).lower()
        
        domain_keywords = {
            self.DOMAIN_HEALTH: ["health", "doctor", "symptom", "fitness", "workout", "exercise", "medical", "wellness"],
            self.DOMAIN_FINANCE: ["money", "budget", "expense", "invest", "tax", "salary", "financial", "payment"],
            self.DOMAIN_LEGAL: ["contract", "legal", "law", "agreement", "sign", "court", "attorney"],
            self.DOMAIN_GROWTH: ["learn", "goal", "skill", "habit", "book", "course", "meditation", "growth"],
            self.DOMAIN_CAREER: ["work", "project", "meeting", "boss", "deadline", "career", "job", "education"],
            self.DOMAIN_HOME: ["home", "house", "apartment", "repair", "clean", "maintenance", "pet", "family"],
            self.DOMAIN_ADMIN: ["schedule", "appointment", "task", "todo", "plan", "calendar", "admin"],
        }
        
        results = []
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', content_str))
            if score > 0:
                results.append((domain, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, score in results[:3]]
    
    def _update_context(self, event: Dict, content_str: str = None):
        """Update ambient context."""
        event_type = _sanitize(str(event.get("type", "unknown"))) if isinstance(event, dict) else "unknown"
        
        with self._lock:
            # Update activity
            if event_type in ["file_write", "terminal", "git"]:
                self.current_context["activity"] = "coding"
            elif event_type in ["window", "clipboard"]:
                content = event.get("content", "")
                if isinstance(content, dict):
                    self.current_context["activity"] = _sanitize_content(str(content.get("title", "unknown")), max_length=200)
                else:
                    self.current_context["activity"] = "unknown"
            
            # Update domain
            domain = self._classify_domain(event, content_str=content_str)
            if domain:
                self.current_context["domain"] = domain
    
    async def _check_insights(self, event: Dict, content_str: str = None):
        """Check for proactive insights (LifeBuddy style)."""
        content_lower = content_str or (json.dumps(event).lower() if isinstance(event, dict) else _sanitize_content(str(event)).lower())
        content_type = _sanitize(str(event.get("type", "unknown"))) if isinstance(event, dict) else "event"
        
        commitment_words = ["call", "promise", "tomorrow", "later", "will", "meeting", "deadline"]
        if any(re.search(r'\b' + re.escape(w) + r'\b', content_lower) for w in commitment_words):
            with self._lock:
                self.insights_queue.append({
                    "priority": "high",
                    "type": "commitment",
                    "message": _sanitize_content(f"Commitment detected in {content_type}"),
                    "timestamp": datetime.now(timezone.utc)
                })
                self._cap_insights_queue()
        
        # Pattern detection
        domain = self._classify_domain(event, content_str=content_str)
        if domain and len(self.domains[domain].records) > 3:
            similar = [r for r in self.domains[domain].records[-10:] 
                       if isinstance(r.get("content", ""), str) and content_type in r.get("content", "")]
            if len(similar) > 3:
                with self._lock:
                    self.insights_queue.append({
                        "priority": "medium",
                        "type": "pattern",
                        "message": _sanitize_content(f"Pattern in {domain}: {event.get('type', 'unknown')} happens frequently"),
                        "timestamp": datetime.now(timezone.utc)
                    })
                    self._cap_insights_queue()
    
    async def ask(self, question: str, domain: str = None) -> Dict:
        """
        Ask Shadow AI with RAG-enhanced context.
        
        Combines:
        - RAG memory retrieval
        - Life domain context
        - 10 abilities synthesis
        """
        question = _sanitize_content(question, max_length=_MAX_QUESTION_LENGTH)
        if domain:
            domain = _sanitize(domain, max_length=50)
        
        # Prepare RAG context
        context_data = await self.rag_context.prepare_context(question, domain)
        
        # Determine current domain
        if not domain:
            domain = self._classify_domain({"content": question})
        
        # Get domain knowledge
        with self._lock:
            domain_context = copy.deepcopy(self.domains[domain].records[-5:]) if domain and domain in self.domains else []
            current_ctx = dict(self.current_context) if isinstance(self.current_context, dict) else self.current_context
        
        # Build unified context
        unified_context = {
            "observations": self.registry._load_json(Path("data/memory/long_term_memory.json")),
            "domain": domain,
            "domain_records": domain_context,
            "recent": context_data.get("recent", []),
            "relevant_memories": context_data.get("relevant", []),
            "current_context": current_ctx
        }
        
        # Auto-select relevant abilities
        abilities_to_use = self._select_abilities_for_query(question, domain)
        
        # Execute abilities
        responses = {}
        for ability in abilities_to_use:
            try:
                result = await self.registry.execute(ability, question, unified_context)
                responses[ability.value] = {
                    "output": result.output,
                    "data": result.data
                }
            except (ValueError, RuntimeError, AttributeError, TypeError, OSError) as e:
                logger.error("Ability %s failed: %s", ability.value, type(e).__name__)
        
        # Synthesize response with domain expertise
        if domain:
            domain_insight = await self._get_domain_insight(question, domain)
            responses[f"{domain}_insight"] = {"output": domain_insight}
        
        return {
            "question": question,
            "domain": domain,
            "abilities_used": [a.value for a in abilities_to_use],
            "responses": responses,
            "context": {
                "recent_count": len(context_data.get("recent", [])),
                "memories_found": len(context_data.get("relevant", [])),
                "domain": domain
            }
        }
    
    def _select_abilities_for_query(self, question: str, domain: str = None) -> List:
        """Select most relevant abilities for query."""
        
        abilities = []
        q = question.lower()
        
        # Domain-specific ability selection
        if domain == self.DOMAIN_HEALTH:
            if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["will", "predict", "happen"]):
                abilities.append(ShikigamiType.ORACLE)
            abilities.append(ShikigamiType.PERFECT_MEMORY)
        
        elif domain == self.DOMAIN_FINANCE:
            if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["should", "decide"]):
                abilities.append(ShikigamiType.SHADOW_COUNCIL)
            abilities.append(ShikigamiType.PATTERN_SIGHT)
        
        elif domain == self.DOMAIN_GROWTH:
            abilities.append(ShikigamiType.SHADOW_COUNCIL)
            abilities.append(ShikigamiType.PERFECT_MEMORY)
        
        elif domain == self.DOMAIN_CAREER:
            if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["will", "future", "predict"]):
                abilities.append(ShikigamiType.ORACLE)
            abilities.append(ShikigamiType.INTENT_SEER)
        
        # General abilities
        if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["is anyone", "nearby", "present", "where"]):
            abilities.append(ShikigamiType.X_RAY)
            abilities.append(ShikigamiType.LIFE_PULSE)
        
        if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["remember", "when did", "what did"]):
            abilities.append(ShikigamiType.PERFECT_MEMORY)
        
        if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["pattern", "trend", "why"]):
            abilities.append(ShikigamiType.PATTERN_SIGHT)
        
        if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ["should", "decide", "choice"]):
            abilities.append(ShikigamiType.SHADOW_COUNCIL)
        
        # Default to Omni Sense
        if not abilities:
            abilities.append(ShikigamiType.OMNI_SENSE)
        
        return list(set(abilities))
    
    async def _get_domain_insight(self, question: str, domain: str) -> str:
        """Get domain-specific insight."""
        if domain not in self.domains:
            return ""
        
        domain_info = self.domains[domain]
        with self._lock:
            records = copy.deepcopy(domain_info.records[-5:])
        
        if not records:
            return f"No specific {domain_info.name} records found."
        
        # Simple insight extraction
        recent_summaries = [_sanitize_content(str(r.get("content", r)), max_length=50) for r in records]
        return f"Recent {domain_info.name.lower()} activity: " + "; ".join(recent_summaries)
    
    def get_proactive_insights(self) -> List[Dict]:
        """Get queued proactive insights."""
        with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=1)
            filtered = []
            for i in self.insights_queue:
                ts = i["timestamp"]
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        continue
                if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    filtered.append(i)
            self.insights_queue = filtered
            result = [
                {
                    "priority": i["priority"],
                    "type": i["type"],
                    "message": i["message"]
                }
                for i in self.insights_queue
                if i["priority"] in ["high", "medium"]
            ]
        return copy.deepcopy(result)
    
    def get_domain_summary(self, domain: str) -> Dict:
        """Get summary of a life domain."""
        with self._lock:
            if domain not in self.domains:
                return {"error": f"Unknown domain: {domain}"}
            domain_info = self.domains[domain]
            return {
                "domain": domain,
                "name": domain_info.name,
                "description": domain_info.description,
                "records_count": len(domain_info.records),
                "recent_records": copy.deepcopy(domain_info.records[-5:]),
                "insights": copy.deepcopy(domain_info.insights),
                "goals": copy.deepcopy(domain_info.goals),
            }
    
    def add_domain_goal(self, domain: str, goal: Dict):
        """Add a goal to a domain."""
        domain = self._validate_domain_key(domain)
        if not domain:
            return
        with self._lock:
            if domain not in self.domains:
                return
            sanitized_goal = {}
            for k, v in goal.items():
                safe_k = _sanitize(str(k), max_length=100)
                if isinstance(v, str):
                    sanitized_goal[safe_k] = _sanitize_content(v, max_length=5000)
                elif isinstance(v, (int, float)):
                    sanitized_goal[safe_k] = round(float(v), 4)
                elif isinstance(v, bool):
                    sanitized_goal[safe_k] = v
                else:
                    sanitized_goal[safe_k] = _sanitize_content(str(v), max_length=5000)
            self.domains[domain].goals.append({
                **sanitized_goal,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if len(self.domains[domain].goals) > self._MAX_GOALS_PER_DOMAIN:
                self.domains[domain].goals = self.domains[domain].goals[-self._MAX_GOALS_PER_DOMAIN:]
            domain_file = self.data_dir / "memory" / f"domain_{domain}.json"
            domain_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = {
                    "records": self.domains[domain].records,
                    "insights": self.domains[domain].insights,
                    "goals": self.domains[domain].goals
                }
                tmp = domain_file.with_suffix('.json.tmp')
                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                try:
                    with os.fdopen(fd, 'w') as f:
                        f.write(json.dumps(data, indent=2, default=str))
                        f.flush()
                        os.fsync(f.fileno())
                    fsync_dir(domain_file.parent)
                    os.replace(str(tmp), str(domain_file))
                except (OSError, ValueError, TypeError):
                    try:
                        os.unlink(str(tmp))
                    except OSError:
                        pass
                    raise
            except (OSError, ValueError, TypeError) as e:
                logger.error("Failed to save domain %s: %s", domain, type(e).__name__)
    
    def get_status(self) -> Dict:
        """Get comprehensive status."""
        memory_stats = self.memory.get_stats()
        with self._lock:
            context_copy = copy.deepcopy(self.current_context) if isinstance(self.current_context, dict) else self.current_context
            insights_count = len(self.insights_queue)
            people_count = len(self.people)
            decisions_count = len(self.decisions)
            patterns_count = len(self.patterns)
            domain_counts = {k: len(v.records) for k, v in self.domains.items()}
        return {
            "memory": memory_stats,
            "domains": domain_counts,
            "current_context": context_copy,
            "insights_queued": insights_count,
            "active_domain": context_copy.get("domain") if isinstance(context_copy, dict) else None,
            "people_tracked": people_count,
            "decisions_learned": decisions_count,
            "patterns_known": patterns_count,
        }
    
    async def run_ambient_monitor(self):
        """
        Background ambient monitoring (LifeBuddy style).
        
        Continuously:
        - Monitors context
        - Detects patterns
        - Generates insights
        - Updates domains
        """
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            try:
                # Get current context
                devices = await self.registry._scan_bt()
                networks = await self.registry._scan_wifi()
                
                # Update location
                location = "unknown"
                for n in networks[:3]:
                    ssid = _sanitize_content(str(n.get("ssid", "")), max_length=100)
                    if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', ssid.lower()) for w in ["home", "my"]):
                        location = "home"
                    elif any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', ssid.lower()) for w in ["office", "corp"]):
                        location = "office"
                
                with self._lock:
                    self.current_context["location"] = location
                
                # Identify people
                present = []
                with self._lock:
                    for name, data in self.people.items():
                        device_list = data.get("devices", [])
                        if isinstance(device_list, list) and len(device_list) <= _MAX_PEOPLE_DEVICES:
                            for d in devices:
                                mac = _sanitize(str(d.get("mac", "")), max_length=17)
                                if mac and mac in device_list:
                                    present.append(_sanitize(name, max_length=100))
                    self.current_context["people"] = present
                
                # Check for insights
                if present:
                    with self._lock:
                        self.insights_queue.append({
                            "priority": "low",
                            "type": "presence",
                            "message": _sanitize_content(f"{', '.join(present[:5])} nearby"),
                            "timestamp": datetime.now(timezone.utc)
                        })
                        self._cap_insights_queue()
                
                safe_location = _sanitize(location, max_length=50)
                logger.info("Ambient monitor: %s, %d people, %d insights", safe_location, len(present), len(self.insights_queue))
                
            except (ValueError, TypeError, OSError, RuntimeError, AttributeError) as e:
                logger.error("Ambient monitor error: %s", type(e).__name__)