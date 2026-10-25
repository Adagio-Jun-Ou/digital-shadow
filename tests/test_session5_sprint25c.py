"""
Session 5 Sprint 25c — Remaining Audit Findings (3 findings)

Fixes:
  AM1: Entity.facts list capped at 1000 (verify in place)
  AM2: TemporalKnowledgeGraph._save() debounced — _save() marks dirty
       and calls flush() when auto_save=True; flush() does the actual I/O.
       Callers can set auto_save=False for batched operations, then flush().
  AM4: TemporalKnowledgeGraph._load() now restores TypedRelation objects
       from persisted data, and _save() (via flush()) includes relations.
"""

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ─── AM1: Entity.facts capped at 1000 ───

class TestEntityFactsCap:
    def test_add_fact_caps_at_1000(self):
        from shadowd.knowledge_graph import Entity, TemporalFact
        entity = Entity(name="test", entity_type="person")
        now = datetime.now(timezone.utc)
        for i in range(1005):
            fact = TemporalFact(
                fact_id=f"f_{i}",
                content=f"Fact {i}",
                entities=["test"],
                valid_from=now,
            )
            entity.add_fact(fact)
        assert len(entity.facts) <= 1000, f"Facts should be capped at 1000, got {len(entity.facts)}"

    def test_add_fact_cap_preserves_latest(self):
        from shadowd.knowledge_graph import Entity, TemporalFact
        entity = Entity(name="test", entity_type="person")
        now = datetime.now(timezone.utc)
        for i in range(1050):
            fact = TemporalFact(
                fact_id=f"f_{i}",
                content=f"Fact {i}",
                entities=["test"],
                valid_from=now,
            )
            entity.add_fact(fact)
        assert len(entity.facts) == 1000
        assert entity.facts[0].content == "Fact 50"
        assert entity.facts[-1].content == "Fact 1049"

    def test_add_fact_source_code_has_1000_cap(self):
        from shadowd.knowledge_graph import Entity
        source = inspect.getsource(Entity.add_fact)
        assert "1000" in source, "add_fact should reference the 1000 cap"


# ─── AM2: _save() debounced with dirty flag and flush() ───

class TestKnowledgeGraphDebouncedSave:
    def test_save_marks_dirty_and_auto_flushes(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert kg._auto_save is True
        kg.add_episode(
            content="test episode",
            source="test",
            facts=[{"content": "test fact", "entities": ["Alice"]}],
            entities=["Alice"],
        )
        assert kg._dirty is False, "After auto-flush, dirty should be False"
        save_file = tmp_path / "knowledge_graph.json"
        assert save_file.exists(), "Auto-save should have flushed to disk"

    def test_auto_save_false_defers_disk_write(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg._auto_save = False
        save_file = tmp_path / "knowledge_graph.json"
        if save_file.exists():
            save_file.unlink()
        kg.add_episode(
            content="deferred episode",
            source="test",
            facts=[{"content": "deferred fact", "entities": ["Bob"]}],
            entities=["Bob"],
        )
        assert kg._dirty is True, "Should be marked dirty after add_episode"
        assert not save_file.exists(), "Disk write should be deferred when auto_save=False"
        kg.flush()
        assert kg._dirty is False, "After flush, dirty should be False"
        assert save_file.exists(), "flush() should write to disk"

    def test_batched_operations_no_per_fact_disk_io(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg._auto_save = False
        for i in range(50):
            kg.add_episode(
                content=f"batch episode {i}",
                source="test",
                facts=[{"content": f"fact {i}", "entities": [f"Entity{i}"]}],
                entities=[f"Entity{i}"],
            )
        assert kg._dirty is True
        save_file = tmp_path / "knowledge_graph.json"
        assert not save_file.exists(), "No disk I/O during batch with auto_save=False"
        kg.flush()
        assert save_file.exists(), "flush() writes to disk"

    def test_has_flush_method(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        assert hasattr(TemporalKnowledgeGraph, "flush"), "TemporalKnowledgeGraph should have flush() method"

    def test_has_dirty_and_auto_save_attrs(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert hasattr(kg, "_dirty"), "Should have _dirty attribute"
        assert hasattr(kg, "_auto_save"), "Should have _auto_save attribute"

    def test_save_method_delegates_to_dirty_flag_and_flush(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph._save)
        assert "_dirty" in source, "_save() should set _dirty"
        assert "flush" in source, "_save() should call flush() when auto_save"

    def test_flush_method_persists_relations(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="Alice works for Acme",
            source="test",
            facts=[{"content": "Alice works for Acme", "entities": ["Alice", "Acme"]}],
            entities=["Alice", "Acme"],
        )
        kg.add_relation("Alice", "Acme", RelationType.WORKS_FOR)
        save_file = tmp_path / "knowledge_graph.json"
        data = json.loads(save_file.read_text())
        assert "relations" in data, "flush() should include relations in persisted data"
        assert len(data["relations"]) == 1
        assert data["relations"][0]["relation_type"] == "works_for"


# ─── AM4: _load() restores TypedRelation objects from disk ───

class TestKnowledgeGraphLoadRelations:
    def test_load_restores_relations(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, TypedRelation, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="Alice met Bob",
            source="test",
            facts=[{"content": "Alice met Bob", "entities": ["Alice", "Bob"]}],
            entities=["Alice", "Bob"],
        )
        kg.add_relation("Alice", "Bob", RelationType.KNOWS, properties={"since": "2024"})
        kg.add_relation("Bob", "Acme", RelationType.WORKS_FOR)
        assert len(kg.relations) == 2

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert len(kg2.relations) == 2, f"Should load 2 relations, got {len(kg2.relations)}"
        rel_alice_bob = kg2.get_relations("Alice", relation_type="knows")
        assert len(rel_alice_bob) == 1
        assert rel_alice_bob[0].from_entity == "Alice"
        assert rel_alice_bob[0].to_entity == "Bob"
        assert rel_alice_bob[0].properties == {"since": "2024"}
        rel_bob_acme = kg2.get_relations("Bob", relation_type="works_for")
        assert len(rel_bob_acme) == 1
        assert rel_bob_acme[0].from_entity == "Bob"
        assert rel_bob_acme[0].to_entity == "Acme"

    def test_load_preserves_relation_index(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test",
            source="test",
            entities=["Alice", "Bob", "Acme"],
        )
        kg.add_relation("Alice", "Bob", RelationType.KNOWS)
        kg.add_relation("Bob", "Acme", RelationType.WORKS_FOR)
        kg.add_relation("Alice", "Acme", RelationType.LIVES_IN)

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        alice_rels = kg2.get_relations("Alice")
        assert len(alice_rels) == 2, f"Alice should have 2 relations, got {len(alice_rels)}"
        bob_rels = kg2.get_relations("Bob")
        assert len(bob_rels) == 2, f"Bob should have 2 relations, got {len(bob_rels)}"
        acme_rels = kg2.get_relations("Acme")
        assert len(acme_rels) == 2, f"Acme should have 2 relations, got {len(acme_rels)}"
        total_rels = len(kg2.relations)
        assert total_rels == 3, f"Total relations should be 3, got {total_rels}"

    def test_load_preserves_relation_between(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test",
            source="test",
            entities=["Alice", "Bob"],
        )
        kg.add_relation("Alice", "Bob", RelationType.KNOWS)

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        between = kg2.get_relation_between("Alice", "Bob", "knows")
        assert len(between) == 1
        assert between[0].relation_type == "knows"

    def test_load_preserves_all_typed_relation_fields(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test",
            source="test",
            entities=["Alice", "Bob"],
        )
        kg.add_relation(
            "Alice", "Bob", RelationType.MENTORED_BY,
            properties={"field": "engineering"},
            confidence=0.85,
            source_episode=kg.episodes[0].episode_id,
        )

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert len(kg2.relations) == 1
        rel = kg2.relations[0]
        assert rel.from_entity == "Alice"
        assert rel.to_entity == "Bob"
        assert rel.relation_type == "mentored_by"
        assert rel.properties == {"field": "engineering"}
        assert abs(rel.confidence - 0.85) < 0.01
        assert rel.source_episode is not None

    def test_load_preserves_entities_and_episodes(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="Alice likes python",
            source="test",
            facts=[{"content": "Alice likes python", "entities": ["Alice"]}],
            entities=["Alice"],
        )

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert "Alice" in kg2.entities
        assert len(kg2.entities["Alice"].facts) == 1
        assert kg2.entities["Alice"].facts[0].content == "Alice likes python"
        assert len(kg2.episodes) == 1

    def test_round_trip_with_empty_relations(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test",
            source="test",
            entities=["Alice"],
        )
        assert len(kg.relations) == 0

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert len(kg2.relations) == 0, "Should load 0 relations when none saved"

    def test_stats_includes_relations_count(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test",
            source="test",
            entities=["Alice", "Bob"],
        )
        kg.add_relation("Alice", "Bob", RelationType.KNOWS)
        stats = kg.get_stats()
        assert "relations" in stats, "get_stats() should include relations count"
        assert stats["relations"] == 1

    def test_save_file_contains_relations_key(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test",
            source="test",
            entities=["Alice", "Bob"],
        )
        kg.add_relation("Alice", "Bob", RelationType.KNOWS)
        save_file = tmp_path / "knowledge_graph.json"
        data = json.loads(save_file.read_text())
        assert "relations" in data, "Persisted data should have 'relations' key"
        assert isinstance(data["relations"], list)
        assert len(data["relations"]) == 1
        rel = data["relations"][0]
        assert "relation_id" in rel
        assert "from_entity" in rel
        assert "to_entity" in rel
        assert "relation_type" in rel
        assert "properties" in rel
        assert "confidence" in rel
        assert "valid_from" in rel