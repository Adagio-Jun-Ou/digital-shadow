#!/usr/bin/env python3
"""Sprint 137 regression tests — metadata deepcopy + except narrowing."""


def test_emotional_intelligence_metadata_deepcopy():
    from shadowd.cognition.emotional_intelligence import EmotionState, EmotionCategory
    es = EmotionState(primary=EmotionCategory.JOY)
    es.metadata["a"] = "1"
    es.metadata["b"] = "2"
    d = es.to_dict()
    d["metadata"]["c"] = "3"
    assert "c" not in es.metadata


def test_emotional_intelligence_regulation_metadata_deepcopy():
    from shadowd.cognition.emotional_intelligence import RegulationStrategy, RegulationApproach, EmotionCategory
    rs = RegulationStrategy(approach=RegulationApproach.MINDFULNESS, target_emotion=EmotionCategory.ANGER)
    rs.metadata["a"] = "1"
    d = rs.to_dict()
    d["metadata"]["b"] = "2"
    assert "b" not in rs.metadata


def test_metacognition_metadata_deepcopy():
    from shadowd.cognition.metacognition_engine import ThinkingEpisode
    te = ThinkingEpisode()
    te.metadata["a"] = "1"
    d = te.to_dict()
    d["metadata"]["b"] = "2"
    assert "b" not in te.metadata


def test_moral_reasoning_metadata_deepcopy():
    from shadowd.cognition.moral_reasoning import EthicalEvaluation
    ee = EthicalEvaluation()
    ee.metadata["a"] = "1"
    d = ee.to_dict()
    d["metadata"]["b"] = "2"
    assert "b" not in ee.metadata


def test_shadow_voice_metadata_deepcopy():
    from shadowd.cognition.shadow_voice import VoiceSession
    vs = VoiceSession()
    vs.metadata["a"] = "1"
    d = vs.to_dict()
    d["metadata"]["b"] = "2"
    assert "b" not in vs.metadata


def test_shadow_voice_metadata_none():
    from shadowd.cognition.shadow_voice import VoiceSession
    vs = VoiceSession()
    d = vs.to_dict()
    assert d["metadata"] == {}


def test_metacognition_metadata_empty():
    from shadowd.cognition.metacognition_engine import ThinkingEpisode
    te = ThinkingEpisode()
    d = te.to_dict()
    assert d["metadata"] == {}


def test_moral_reasoning_metadata_empty():
    from shadowd.cognition.moral_reasoning import EthicalEvaluation
    ee = EthicalEvaluation()
    d = ee.to_dict()
    assert d["metadata"] == {}


def test_shadow_graph_condition_narrowed_except():
    import inspect
    from shadowd import cognition
    source = inspect.getsource(cognition.shadow_graph)
    assert "except (TypeError, ValueError, AttributeError, KeyError):" in source


def test_shadow_graph_escalation_narrowed_except():
    import inspect
    from shadowd import cognition
    source = inspect.getsource(cognition.shadow_graph)
    assert "except (ValueError, KeyError, TypeError, AttributeError, RuntimeError):" in source


def test_stealth_layer_is_recent_narrowed():
    import inspect
    from shadowd import cognition
    source = inspect.getsource(cognition.stealth_layer)
    assert "except (ValueError, TypeError):" in source


def test_neural_mesh_is_alive_narrowed():
    import inspect
    from shadowd import cognition
    source = inspect.getsource(cognition.neural_mesh)
    assert "except (ValueError, TypeError):" in source


def test_dark_memory_exception_narrowed():
    import inspect
    from shadowd import cognition
    source = inspect.getsource(cognition.dark_memory)
    assert "except (ValueError, KeyError):" in source
    assert "except (OSError, ValueError)" in source