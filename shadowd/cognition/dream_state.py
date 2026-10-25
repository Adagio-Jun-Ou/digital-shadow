"""
DreamState — Offline Processing & Memory Consolidation Engine

Like REM sleep for AI: consolidates memories during idle periods,
strengthens patterns, forgets noise, and discovers hidden connections.
"""

import copy
import logging
import re
import threading
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DreamPhase(str, Enum):
    PATTERN_STRENGTHENING = "pattern_strengthening"
    MEMORY_PRUNING = "memory_pruning"
    INSIGHT_GENERATION = "insight_generation"
    FORGETTING = "forgetting"


@dataclass
class DreamMemory:
    memory_id: str
    content: str
    domain: str
    importance: float
    tags: List[str]
    access_count: int
    created_at: str
    last_accessed: str
    strengthened: bool

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "domain": self.domain,
            "importance": round(self.importance, 3),
            "tags": list(self.tags),
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "strengthened": self.strengthened,
        }


@dataclass
class DreamPattern:
    pattern_id: str
    theme: str
    domains: List[str]
    frequency: int
    avg_importance: float
    related_memories: List[str]

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "theme": self.theme,
            "domains": list(self.domains),
            "frequency": self.frequency,
            "avg_importance": round(self.avg_importance, 3),
            "related_memories": list(self.related_memories),
        }


@dataclass
class DreamInsight:
    insight_id: str
    description: str
    domains_connected: List[str]
    confidence: float
    memory_ids: List[str]
    created_at: str

    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "description": self.description,
            "domains_connected": list(self.domains_connected),
            "confidence": round(self.confidence, 3),
            "memory_ids": list(self.memory_ids),
            "created_at": self.created_at,
        }


@dataclass
class ConsolidationResult:
    memories_consolidated: int
    memories_forgotten: int
    insights_generated: int
    patterns_strengthened: int
    phases_completed: List[str]
    duration_ms: float

    def to_dict(self) -> Dict:
        return {
            "memories_consolidated": self.memories_consolidated,
            "memories_forgotten": self.memories_forgotten,
            "insights_generated": self.insights_generated,
            "patterns_strengthened": self.patterns_strengthened,
            "phases_completed": list(self.phases_completed),
            "duration_ms": round(self.duration_ms, 2),
        }


class DreamState:
    """
    Offline Processing & Memory Consolidation Engine

    Like REM sleep for AI: consolidates memories during idle periods,
    strengthens patterns, forgets noise, and discovers hidden connections.
    """

    _MAX_MEMORIES = 5000
    _MAX_PATTERNS = 500
    _MAX_INSIGHTS = 1000
    _FORGET_IMPORTANCE_THRESHOLD = 0.2
    _FORGET_ACCESS_THRESHOLD = 1
    _PRUNE_OVERLAP_THRESHOLD = 0.8
    _PATTERN_SHARED_WORDS_MIN = 3
    _PATTERN_IMPORTANCE_BOOST = 0.1
    _INSIGHT_KEYWORD_OVERLAP = 0.4
    _MAX_FORGET_RATIO = 0.1
    _MAX_TAGS = 20

    def __init__(self):
        self._memories: Dict[str, DreamMemory] = {}
        self._patterns: Dict[str, DreamPattern] = {}
        self._insights: deque = deque(maxlen=self._MAX_INSIGHTS)
        self._insertion_order: deque = deque(maxlen=self._MAX_MEMORIES)
        self._pattern_order: List[str] = []
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "memories_added": 0,
            "consolidations_run": 0,
            "total_patterns_found": 0,
        }

    def add_memory(
        self,
        content: str,
        domain: str = "general",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content.strip())[:10000]
        if importance != importance:
            importance = 0.5
        importance = max(0.0, min(1.0, importance))
        domain = re.sub(r'[<>\x00-\x1f\x7f]', '', domain.strip())[:100] if domain else "general"
        if tags is None:
            tags = []
        tags = [re.sub(r'[<>\x00-\x1f\x7f]', '', str(t).strip())[:50] for t in tags if str(t).strip()]
        tags = tags[:20]

        memory_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        memory = DreamMemory(
            memory_id=memory_id,
            content=content,
            domain=domain,
            importance=importance,
            tags=tags,
            access_count=0,
            created_at=now,
            last_accessed=now,
            strengthened=False,
        )

        with self._lock:
            if len(self._memories) >= self._MAX_MEMORIES:
                oldest_id = self._insertion_order.popleft()
                self._memories.pop(oldest_id, None)
                logger.warning("Evicted memory %s to make room for new memory", oldest_id)
            self._memories[memory_id] = memory
            self._insertion_order.append(memory_id)
            with self._stats_lock:
                self._stats["memories_added"] += 1

        return memory_id

    def consolidate(self) -> ConsolidationResult:
        start = datetime.now(timezone.utc)
        phases_completed: List[str] = []
        memories_consolidated = 0
        memories_forgotten = 0
        insights_generated = 0
        patterns_strengthened = 0

        # Phase 1: Pattern strengthening
        with self._lock:
            memories_snapshot = dict(self._memories)
            patterns_strengthened, consolidated_ids = self._phase_pattern_strengthening(memories_snapshot)
            if consolidated_ids:
                memories_consolidated = len(consolidated_ids)
            phases_completed.append(DreamPhase.PATTERN_STRENGTHENING.value)

        # Phase 2: Memory pruning
        with self._lock:
            memories_snapshot = dict(self._memories)
            pruned = self._phase_memory_pruning(memories_snapshot)
            phases_completed.append(DreamPhase.MEMORY_PRUNING.value)

        # Phase 3: Insight generation
        with self._lock:
            memories_snapshot = dict(self._memories)
            insights_generated = self._phase_insight_generation(memories_snapshot)
            phases_completed.append(DreamPhase.INSIGHT_GENERATION.value)

        # Phase 4: Forgetting
        with self._lock:
            max_forget = max(1, int(len(self._memories) * self._MAX_FORGET_RATIO))
            forgotten = self._phase_forgetting(max_forget)
            memories_forgotten = forgotten
            phases_completed.append(DreamPhase.FORGETTING.value)

        with self._stats_lock:
            self._stats["consolidations_run"] += 1
            self._stats["total_patterns_found"] += patterns_strengthened

        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return ConsolidationResult(
            memories_consolidated=memories_consolidated,
            memories_forgotten=memories_forgotten,
            insights_generated=insights_generated,
            patterns_strengthened=patterns_strengthened,
            phases_completed=phases_completed,
            duration_ms=duration_ms,
        )

    def _phase_pattern_strengthening(self, memories: Dict[str, DreamMemory]):
        word_groups: Dict[str, List[str]] = {}
        for mid, mem in memories.items():
            words = set(re.findall(r'\b\w{3,}\b', mem.content.lower()))
            for w in words:
                if len(w) < 3:
                    continue
                if w not in word_groups:
                    word_groups[w] = []
                word_groups[w].append(mid)

        patterns_found = 0
        consolidated_ids = set()

        for word, mids in word_groups.items():
            if len(mids) < 2:
                continue
            relevant_memories = [memories[mid] for mid in mids if mid in memories]
            if len(relevant_memories) < 2:
                continue
            shared_count = len(mids)
            meets_threshold = shared_count >= self._PATTERN_SHARED_WORDS_MIN or any(
                len(set(re.findall(r'\b\w{3,}\b', m.content.lower())) & set(re.findall(r'\b\w{3,}\b', memories[mids[0]].content.lower()))) >= self._PATTERN_SHARED_WORDS_MIN
                for m in relevant_memories[1:]
            )
            if meets_threshold:
                for mid in mids:
                    if mid in self._memories:
                        self._memories[mid].importance = min(1.0, self._memories[mid].importance + self._PATTERN_IMPORTANCE_BOOST)
                        self._memories[mid].strengthened = True
                        consolidated_ids.add(mid)

                pattern_id = uuid.uuid4().hex[:12]
                domains = list(set(m.domain for m in relevant_memories))
                avg_imp = sum(m.importance for m in relevant_memories) / len(relevant_memories) if relevant_memories else 0.0
                pattern = DreamPattern(
                    pattern_id=pattern_id,
                    theme=word,
                    domains=domains,
                    frequency=shared_count,
                    avg_importance=avg_imp,
                    related_memories=mids[:20],
                )

                if len(self._patterns) < self._MAX_PATTERNS:
                    if word not in self._patterns:
                        self._pattern_order.append(word)
                    self._patterns[word] = pattern
                    patterns_found += 1
                elif word in self._patterns:
                    self._patterns[word] = pattern
                else:
                    if self._pattern_order:
                        oldest_key = self._pattern_order.pop(0)
                        del self._patterns[oldest_key]
                        logger.warning("Evicted pattern '%s' to make room for '%s'", oldest_key, word)
                    self._patterns[word] = pattern
                    self._pattern_order.append(word)

        return patterns_found, consolidated_ids

    def _phase_memory_pruning(self, memories: Dict[str, DreamMemory]) -> int:
        pruned = 0
        seen_content: Dict[str, str] = {}

        for mid, mem in list(memories.items()):
            if mid not in self._memories:
                continue

            content_words = set(re.findall(r'\b\w{3,}\b', mem.content.lower()))
            content_key = " ".join(sorted(content_words))

            duplicate_of = None
            if content_key in seen_content:
                duplicate_of = seen_content[content_key]
            else:
                for existing_key, existing_mid in seen_content.items():
                    existing_words = set(existing_key.split())
                    if not existing_words or not content_words:
                        continue
                    overlap = len(content_words & existing_words) / max(len(content_words | existing_words), 1)
                    if overlap > self._PRUNE_OVERLAP_THRESHOLD:
                        duplicate_of = existing_mid
                        break

            if duplicate_of is None:
                seen_content[content_key] = mid
            else:
                existing_mem = self._memories.get(duplicate_of)
                current_mem = self._memories.get(mid)
                if existing_mem and current_mem:
                    existing_mem.importance = min(1.0, existing_mem.importance + current_mem.importance * 0.5)
                    del self._memories[mid]
                    try:
                        self._insertion_order.remove(mid)
                    except (ValueError, KeyError):
                        pass
                    pruned += 1

        return pruned

    def _phase_insight_generation(self, memories: Dict[str, DreamMemory]) -> int:
        domain_memories: Dict[str, List[DreamMemory]] = {}
        for mem in memories.values():
            domain_memories.setdefault(mem.domain, []).append(mem)

        domains = list(domain_memories.keys())
        insights = 0

        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                m1_list = domain_memories[d1]
                m2_list = domain_memories[d2]
                if not m1_list or not m2_list:
                    continue

                def _top_keywords(memories_list: List[DreamMemory]) -> Counter:
                    counter = Counter()
                    for m in memories_list:
                        words = re.findall(r'\b\w{4,}\b', m.content.lower())
                        counter.update(words)
                    return counter

                kw1 = _top_keywords(m1_list)
                kw2 = _top_keywords(m2_list)

                set1 = set(dict(kw1.most_common(20)).keys())
                set2 = set(dict(kw2.most_common(20)).keys())

                if not set1 or not set2:
                    continue

                overlap = len(set1 & set2) / max(len(set1 | set2), 1)
                if overlap >= self._INSIGHT_KEYWORD_OVERLAP:
                    shared_words = set1 & set2
                    insight_id = uuid.uuid4().hex[:12]
                    relevant_ids = [m.memory_id for m in m1_list[:3] + m2_list[:3]]
                    now = datetime.now(timezone.utc).isoformat()
                    description = (
                        f"Cross-domain connection: {d1} and {d2} share themes around "
                        f"{', '.join(list(shared_words)[:5])}"
                    )
                    insight = DreamInsight(
                        insight_id=insight_id,
                        description=description,
                        domains_connected=[d1, d2],
                        confidence=min(0.9, overlap + 0.1),
                        memory_ids=relevant_ids,
                        created_at=now,
                    )
                    if len(self._insights) >= self._MAX_INSIGHTS:
                        evicted = self._insights[0]
                        logger.warning("Evicted insight %s to make room for new insight", evicted.insight_id)
                    self._insights.append(insight)
                    insights += 1

        return insights

    def _phase_forgetting(self, max_forget: int) -> int:
        forgotten = 0
        to_remove = []

        for mid, mem in self._memories.items():
            if mem.importance < self._FORGET_IMPORTANCE_THRESHOLD and mem.access_count <= self._FORGET_ACCESS_THRESHOLD:
                to_remove.append(mid)
                if len(to_remove) >= max_forget:
                    break

        for mid in to_remove:
            del self._memories[mid]
            try:
                self._insertion_order.remove(mid)
            except (ValueError, KeyError):
                pass
            forgotten += 1
            logger.warning("Forgot memory %s (low importance/access)", mid)

        return forgotten

    def get_insights(self) -> List[DreamInsight]:
        with self._lock:
            return copy.deepcopy(list(self._insights))

    def get_patterns(self) -> List[DreamPattern]:
        with self._lock:
            return copy.deepcopy(list(self._patterns.values()))

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id in self._memories:
                del self._memories[memory_id]
                try:
                    self._insertion_order.remove(memory_id)
                except (ValueError, KeyError):
                    pass
                return True
            return False

    def recall(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[DreamMemory]:
        limit = max(1, min(100, limit))
        if not query or not query.strip():
            return []
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query.strip())[:1000]
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            query_words = {query.lower()}

        with self._lock:
            results = []
            for mem in self._memories.values():
                if domain and mem.domain != domain:
                    continue
                mem_words = set(re.findall(r'\b\w{3,}\b', mem.content.lower()))
                if query_words & mem_words or query.lower() in mem.content.lower():
                    mem.access_count += 1
                    mem.last_accessed = datetime.now(timezone.utc).isoformat()
                    results.append(mem)

            results.sort(key=lambda m: (m.importance, m.access_count), reverse=True)
            results = results[:limit]
            results = [copy.deepcopy(m) for m in results]
            return results

    def get_stats(self) -> Dict:
        with self._lock:
            total_memories = len(self._memories)
            domains = Counter(m.domain for m in self._memories.values())
            avg_importance = (
                sum(m.importance for m in self._memories.values()) / total_memories
                if total_memories > 0
                else 0.0
            )
            total_patterns = len(self._patterns)
            total_insights = len(self._insights)
        with self._stats_lock:
            stats_snapshot = dict(self._stats)
        return {
            "total_memories": total_memories,
            "total_patterns": total_patterns,
            "total_insights": total_insights,
            "consolidation_count": stats_snapshot["consolidations_run"],
            "domains": dict(domains.most_common(20)),
            "avg_importance": round(avg_importance, 3),
            "max_memories": self._MAX_MEMORIES,
            "max_patterns": self._MAX_PATTERNS,
            "max_insights": self._MAX_INSIGHTS,
        }