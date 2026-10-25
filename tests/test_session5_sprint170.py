"""Sprint 170: datetime.fromisoformat() crash guard regression tests."""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


class TestSuperMemoryParseTs:
    def test_valid_timestamp(self):
        from shadowd.memory.super_memory import _parse_ts
        result = _parse_ts({"created": "2024-01-15T10:30:00+00:00"}, "created")
        assert isinstance(result, datetime)

    def test_bad_timestamp_returns_default(self):
        from shadowd.memory.super_memory import _parse_ts
        default = datetime.now(timezone.utc)
        result = _parse_ts({"created": "garbage"}, "created", default)
        assert result == default

    def test_missing_key_returns_default(self):
        from shadowd.memory.super_memory import _parse_ts
        default = datetime.now(timezone.utc)
        result = _parse_ts({}, "created", default)
        assert result == default

    def test_no_default_returns_utc_now(self):
        from shadowd.memory.super_memory import _parse_ts
        result = _parse_ts({"created": "garbage"}, "created")
        assert isinstance(result, datetime)

    def test_memory_block_from_dict_bad_timestamp(self):
        from shadowd.memory.super_memory import MemoryBlock
        data = {"label": "test", "value": "hello", "last_modified": "NOT_A_DATE"}
        block = MemoryBlock.from_dict(data)
        assert block.label == "test"
        assert isinstance(block.last_modified, datetime)

    def test_entity_from_dict_bad_timestamp(self):
        from shadowd.memory.super_memory import Entity
        data = {"id": "e1", "type": "person", "name": "Alice", "created": "NOT_A_DATE"}
        entity = Entity.from_dict(data)
        assert entity.name == "Alice"
        assert isinstance(entity.created, datetime)

    def test_relation_from_dict_bad_timestamp(self):
        from shadowd.memory.super_memory import Relation
        data = {"id": "r1", "subject": "a", "predicate": "knows", "object": "b", "timestamp": "NOT_A_DATE"}
        rel = Relation.from_dict(data)
        assert isinstance(rel.timestamp, datetime)


class TestSecuritySafeTimestamp:
    def test_valid_timestamp(self):
        from shadowd.security import _safe_ts
        result = _safe_ts({"created_at": "2024-01-15T10:30:00+00:00"}, "created_at", datetime.now(timezone.utc))
        assert isinstance(result, datetime)

    def test_bad_timestamp_returns_default(self):
        from shadowd.security import _safe_ts
        default = datetime.now(timezone.utc)
        result = _safe_ts({"created_at": "garbage"}, "created_at", default)
        assert result == default

    def test_missing_key_returns_none(self):
        from shadowd.security import _safe_ts
        result = _safe_ts({}, "created_at")
        assert result is None

    def test_user_from_dict_bad_timestamp(self):
        from shadowd.security import User
        data = {
            "user_id": "u1", "email": "a@b.com", "username": "test",
            "hashed_password": "hash", "created_at": "NOT_A_DATE"
        }
        user = User.from_dict(data)
        assert isinstance(user.created_at, datetime)

    def test_api_key_from_dict_bad_timestamp(self):
        from shadowd.security import APIKey
        data = {
            "key_id": "k1", "key_hash": "h", "user_id": "u1", "name": "test",
            "created_at": "NOT_A_DATE"
        }
        key = APIKey.from_dict(data)
        assert isinstance(key.created_at, datetime)


class TestKnowledgeGraphTimestamp:
    def test_valid_timestamp(self):
        from shadowd.knowledge_graph import _kg_ts
        result = _kg_ts({"valid_from": "2024-01-15T10:30:00+00:00"}, "valid_from")
        assert isinstance(result, datetime)

    def test_bad_timestamp_returns_none(self):
        from shadowd.knowledge_graph import _kg_ts
        result = _kg_ts({"valid_from": "garbage"}, "valid_from")
        assert result is None

    def test_missing_key_returns_none(self):
        from shadowd.knowledge_graph import _kg_ts
        result = _kg_ts({}, "valid_from")
        assert result is None


class TestDecisionLearningTimestamp:
    def test_valid_timestamp(self):
        from shadowd.decision_learning_v2 import _dl_ts
        default = datetime.now(timezone.utc)
        result = _dl_ts({"created_at": "2024-01-15T10:30:00+00:00"}, "created_at", default)
        assert isinstance(result, datetime)

    def test_bad_timestamp_returns_default(self):
        from shadowd.decision_learning_v2 import _dl_ts
        default = datetime.now(timezone.utc)
        result = _dl_ts({"created_at": "garbage"}, "created_at", default)
        assert result == default

    def test_missing_key_returns_none(self):
        from shadowd.decision_learning_v2 import _dl_ts
        result = _dl_ts({}, "deadline")
        assert result is None


class TestReasoningStateGraphTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.reasoning.state_graph import Checkpoint, State
        data = {
            "id": "cp1", "state": "init", "context": {},
            "created_at": "NOT_A_DATE"
        }
        cp = Checkpoint.from_dict(data)
        assert isinstance(cp.created_at, datetime)

    def test_from_dict_missing_timestamp(self):
        from shadowd.reasoning.state_graph import Checkpoint, State
        data = {"id": "cp1", "state": "init", "context": {}}
        cp = Checkpoint.from_dict(data)
        assert isinstance(cp.created_at, datetime)


class TestStateGraphTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.state_graph import StateCheckpoint, NodeStatus
        data = {
            "node_id": "n1", "timestamp": "NOT_A_DATE",
            "state": {}, "status": "pending"
        }
        cp = StateCheckpoint.from_dict(data)
        assert isinstance(cp.timestamp, datetime)

    def test_from_dict_missing_timestamp(self):
        from shadowd.state_graph import StateCheckpoint, NodeStatus
        data = {"node_id": "n1", "state": {}, "status": "pending"}
        cp = StateCheckpoint.from_dict(data)
        assert isinstance(cp.timestamp, datetime)


class TestWorkflowEngineTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.workflow.engine import Checkpoint, BlockStatus
        data = {
            "workflow_id": "w1", "block_id": "b1",
            "timestamp": "NOT_A_DATE", "state": {},
            "status": "pending"
        }
        cp = Checkpoint.from_dict(data)
        assert isinstance(cp.timestamp, datetime)

    def test_from_dict_missing_timestamp(self):
        from shadowd.workflow.engine import Checkpoint, BlockStatus
        data = {
            "workflow_id": "w1", "block_id": "b1",
            "state": {}, "status": "pending"
        }
        cp = Checkpoint.from_dict(data)
        assert isinstance(cp.timestamp, datetime)


class TestBlockMemoryTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.block_memory import MemoryBlock
        data = {"label": "test", "value": "hello", "last_modified": "NOT_A_DATE"}
        block = MemoryBlock.from_dict(data)
        assert isinstance(block.last_modified, datetime)

    def test_from_dict_missing_timestamp(self):
        from shadowd.block_memory import MemoryBlock
        data = {"label": "test", "value": "hello"}
        block = MemoryBlock.from_dict(data)
        assert isinstance(block.last_modified, datetime)


class TestAgentCoordinationTimestamp:
    def test_agent_message_from_dict_bad_timestamp(self):
        from shadowd.agent_coordination import AgentMessage
        data = {
            "sender": "s1", "receiver": "r1", "content": "hello",
            "timestamp": "NOT_A_DATE"
        }
        msg = AgentMessage.from_dict(data)
        assert isinstance(msg.timestamp, datetime)

    def test_shadow_task_from_dict_bad_timestamp(self):
        from shadowd.agent_coordination import ShadowTask, TaskStatus
        data = {
            "id": "t1", "description": "test", "goal": "test goal",
            "status": "pending", "created_at": "NOT_A_DATE"
        }
        task = ShadowTask.from_dict(data)
        assert isinstance(task.created_at, datetime)


class TestContextProtocolTimestamp:
    def test_security_context_from_dict_bad_timestamp(self):
        from shadowd.reasoning.context_protocol import SecurityContext
        data = {"entity_id": "e1", "entity_type": "user", "created_at": "NOT_A_DATE"}
        ctx = SecurityContext.from_dict(data)
        assert isinstance(ctx.created_at, datetime)

    def test_context_message_from_dict_bad_timestamp(self):
        from shadowd.reasoning.context_protocol import ContextMessage
        data = {"sender_id": "s1", "receiver_id": "r1", "message_type": "test", "timestamp": "NOT_A_DATE"}
        msg = ContextMessage.from_dict(data)
        assert isinstance(msg.timestamp, datetime)


class TestFunctionDbTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.self_improvement.function_db import GeneratedFunction, FunctionStatus
        data = {
            "function_id": "f1", "name": "test_func", "description": "test",
            "parameters": {}, "code": "pass", "returns": "None",
            "status": "draft", "created_at": "NOT_A_DATE"
        }
        func = GeneratedFunction.from_dict(data)
        assert isinstance(func.created_at, datetime)


class TestAttentionEngineTimestamp:
    def test_corrupt_started_at_no_crash(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusSession
        ae = AttentionEngine()
        session = FocusSession(
            session_id="test", topic="test", started_at="CORRUPT_TIMESTAMP",
            ended_at=None, duration_minutes=0.0, interruptions=0, depth_score=0.0
        )
        ae._current_session = session
        result = ae._end_focus_unlocked(datetime.now(timezone.utc))
        assert session.duration_minutes == 0.0


class TestAdvancedMemoryTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.advanced_memory import _am_ts
        result = _am_ts({"last_accessed": "NOT_A_DATE"}, "last_accessed")
        assert isinstance(result, datetime) or result is None

    def test_from_dict_valid_timestamp(self):
        from shadowd.advanced_memory import _am_ts
        result = _am_ts({"last_accessed": "2024-01-15T10:30:00+00:00"}, "last_accessed")
        assert isinstance(result, datetime)


class TestLearningSystemTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.learning_system import _ls_ts
        result = _ls_ts({"timestamp": "NOT_A_DATE"}, "timestamp")
        assert result is None

    def test_from_dict_valid_timestamp(self):
        from shadowd.learning_system import _ls_ts
        result = _ls_ts({"timestamp": "2024-01-15T10:30:00+00:00"}, "timestamp")
        assert isinstance(result, datetime)


class TestMemoryRagTimestamp:
    def test_from_dict_bad_timestamp(self):
        from shadowd.memory_rag import _mr_ts
        result = _mr_ts({"timestamp": "NOT_A_DATE"}, "timestamp")
        assert isinstance(result, datetime)

    def test_from_dict_valid_timestamp(self):
        from shadowd.memory_rag import _mr_ts
        result = _mr_ts({"timestamp": "2024-01-15T10:30:00+00:00"}, "timestamp")
        assert isinstance(result, datetime)