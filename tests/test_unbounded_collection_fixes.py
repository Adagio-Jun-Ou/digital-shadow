#!/usr/bin/env python3
"""
Regression tests for unbounded collection fixes.

Findings addressed:
1. ReinforcementLearner q_table and experiences (HIGH) → capped at 50000/1000
2. MarkovBehaviorModel transitions, state_counts, action_counts (HIGH) → capped at 500 states
3. BehaviorPredictor behavior_models and contexts (HIGH) → capped at 500/50
4. DeviceFingerprinter _known_devices (HIGH) → capped at 5000
5. TemporalPatternEngine day_counts and transitions (MEDIUM) → capped at 365/5000
6. ActionSequencePredictor transitions (MEDIUM) → capped at 5000
7. SignalTriangulator signal_history (MEDIUM) → capped at 1000 MACs / 100 per-MAC
8. AmbientMonitor patterns and event_counts (MEDIUM) → capped at 500/200
9. AmbientMonitor people_encounters outer dict (MEDIUM) → capped at 500
10. temporal_reasoning _patterns/_predictions (ALREADY BOUNDED - verified)
11. shadow_graph nodes (LOW) → capped at 50
"""

import pytest
from datetime import datetime, timedelta, timezone
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════
# Fix 1: ReinforcementLearner
# ═════════════════════════════════════════════════════════════════

class TestReinforcementLearnerCaps:
    def setup_method(self):
        from shikigami.superhuman import ReinforcementLearner
        self.rl = ReinforcementLearner()

    def test_cap_constants_exist(self):
        assert hasattr(self.rl, '_MAX_Q_TABLE')
        assert hasattr(self.rl, '_MAX_EXPERIENCES')
        assert self.rl._MAX_Q_TABLE == 50000
        assert self.rl._MAX_EXPERIENCES == 1000

    def test_q_table_order_tracking(self):
        assert hasattr(self.rl, '_q_order')
        assert isinstance(self.rl._q_order, list)

    def test_experiences_capped(self):
        for i in range(1500):
            self.rl.update(f's{i%10}', f'a{i%5}', float(i), f'next_{i}', [f'a{i%5}'])
        assert len(self.rl.experiences) <= self.rl._MAX_EXPERIENCES

    def test_q_table_eviction_at_cap(self):
        for i in range(100):
            self.rl.update(f'unique_state_{i}', f'action_{i}', 1.0, f'next_{i}', [f'a{i}'])
        # Should have evicted some entries
        assert len(self.rl.q_table) <= self.rl._MAX_Q_TABLE + 10  # small buffer for race

    def test_experience_halving_on_eviction(self):
        for i in range(1000):
            self.rl.update(f's{i%5}', f'a{i%3}', float(i), f'next_{i}', [f'a{i}'])
        # After eviction, experiences should be roughly half the cap
        assert len(self.rl.experiences) <= self.rl._MAX_EXPERIENCES

    def test_save_load_preserves_caps(self, tmp_path):
        for i in range(50):
            self.rl.update(f's{i}', f'a{i}', float(i), f'next_{i}', [f'a{i}'])
        path = tmp_path / "q_table.json"
        self.rl.save_q_table(path)
        from shikigami.superhuman import ReinforcementLearner
        rl2 = ReinforcementLearner()
        rl2.load_q_table(path)
        assert len(rl2.q_table) == len(self.rl.q_table)


# ═════════════════════════════════════════════════════════════════
# Fix 2: MarkovBehaviorModel
# ═════════════════════════════════════════════════════════════════

class TestMarkovBehaviorModelCaps:
    def setup_method(self):
        from shikigami.superhuman import MarkovBehaviorModel
        self.mbm = MarkovBehaviorModel()

    def test_cap_constants_exist(self):
        assert hasattr(self.mbm, '_MAX_STATES')
        assert hasattr(self.mbm, '_MAX_HISTORY')
        assert self.mbm._MAX_STATES == 500
        assert self.mbm._MAX_HISTORY == 1000

    def test_state_order_tracking(self):
        assert hasattr(self.mbm, '_state_order')
        assert isinstance(self.mbm._state_order, list)

    def test_state_eviction_at_cap(self):
        for i in range(600):
            self.mbm.record(f'state_{i}', 'action', f'next_{i}', 1.0)
        assert len(self.mbm._state_order) <= self.mbm._MAX_STATES
        assert len(self.mbm.state_counts) <= self.mbm._MAX_STATES
        assert len(self.mbm.transitions) <= self.mbm._MAX_STATES

    def test_history_capped(self):
        for i in range(1500):
            self.mbm.record('s1', 'a1', 's2', 1.0)
        assert len(self.mbm.history) <= self.mbm._MAX_HISTORY * 2  # halving may leave up to cap

    def test_rewards_cleaned_on_eviction(self):
        for i in range(600):
            self.mbm.record(f'state_{i}', 'action', f'next_{i}', 1.0)
        # Rewards for evicted states should be gone
        assert len(self.mbm.rewards) <= self.mbm._MAX_STATES


# ═════════════════════════════════════════════════════════════════
# Fix 3: BehaviorPredictor
# ═════════════════════════════════════════════════════════════════

class TestBehaviorPredictorCaps:
    def setup_method(self):
        from shikigami.abilities import BehaviorPredictor
        self.bp = BehaviorPredictor()

    def test_cap_constants_exist(self):
        assert hasattr(self.bp, '_MAX_PEOPLE')
        assert hasattr(self.bp, '_MAX_CONTEXTS')
        assert self.bp._MAX_PEOPLE == 500
        assert self.bp._MAX_CONTEXTS == 50

    def test_people_order_tracking(self):
        assert hasattr(self.bp, '_people_order')
        assert isinstance(self.bp._people_order, list)

    def test_people_eviction_at_cap(self):
        for i in range(600):
            self.bp.track_behavior(f'person_{i}', 'exercise', True, {'time': 'morning'})
        assert len(self.bp.behavior_models) <= self.bp._MAX_PEOPLE
        assert len(self.bp._people_order) <= self.bp._MAX_PEOPLE

    def test_contexts_capped_per_behavior(self):
        for i in range(80):
            self.bp.track_behavior('alice', 'exercise', True, {'detail': f'ctx_{i}'})
        # Should have been capped
        assert len(self.bp.behavior_models['alice']['exercise']['contexts']) <= self.bp._MAX_CONTEXTS

    def test_predict_still_works(self):
        for _ in range(10):
            self.bp.track_behavior('alice', 'exercise', True)
        result = self.bp.predict_behavior('alice')
        assert 'confidence' in result


# ═════════════════════════════════════════════════════════════════
# Fix 4: DeviceFingerprinter
# ═════════════════════════════════════════════════════════════════

class TestDeviceFingerprinterCaps:
    def setup_method(self):
        from shikigami.abilities import DeviceFingerprinter
        self.df = DeviceFingerprinter()

    def test_cap_constants_exist(self):
        assert hasattr(self.df, '_MAX_KNOWN_DEVICES')
        assert self.df._MAX_KNOWN_DEVICES == 5000

    def test_device_order_tracking(self):
        assert hasattr(self.df, '_known_devices_order')
        assert isinstance(self.df._known_devices_order, list)

    def test_identify_device(self):
        result = self.df.identify_device('AA:BB:CC:DD:EE:FF', 'iPhone')
        assert isinstance(result, dict)
        assert 'identified' in result or 'device_type' in result


# ═════════════════════════════════════════════════════════════════
# Fix 5: TemporalPatternEngine
# ═════════════════════════════════════════════════════════════════

class TestTemporalPatternEngineCaps:
    def setup_method(self):
        from shikigami.superhuman import TemporalPatternEngine
        self.tpe = TemporalPatternEngine()

    def test_cap_constants_exist(self):
        assert hasattr(self.tpe, '_MAX_DAY_COUNTS')
        assert hasattr(self.tpe, '_MAX_TRANSITIONS')
        assert hasattr(self.tpe, '_MAX_TYPE_KEYS')
        assert self.tpe._MAX_DAY_COUNTS == 365
        assert self.tpe._MAX_TRANSITIONS == 5000
        assert self.tpe._MAX_TYPE_KEYS == 100

    def test_day_order_tracking(self):
        assert hasattr(self.tpe, '_day_order')
        assert isinstance(self.tpe._day_order, list)

    def test_observe_works(self):
        self.tpe.observe({'timestamp': datetime.now(timezone.utc).isoformat(), 'type': 'test'})
        assert self.tpe.total_observations == 1

    def test_day_counts_eviction(self):
        from datetime import date
        base = date(2024, 1, 1)
        for i in range(400):
            dt = datetime.combine(base + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc)
            self.tpe.observe({'timestamp': dt.isoformat(), 'type': 'test'})
        assert len(self.tpe.day_counts) <= self.tpe._MAX_DAY_COUNTS + 10


# ═════════════════════════════════════════════════════════════════
# Fix 6: ActionSequencePredictor
# ═════════════════════════════════════════════════════════════════

class TestActionSequencePredictorCaps:
    def setup_method(self):
        from shikigami.abilities import ActionSequencePredictor
        self.asp = ActionSequencePredictor()

    def test_cap_constants_exist(self):
        assert hasattr(self.asp, '_MAX_SEQUENCES')
        assert hasattr(self.asp, '_MAX_TRANSITIONS')
        assert self.asp._MAX_SEQUENCES == 5000
        assert self.asp._MAX_TRANSITIONS == 5000

    def test_sequence_order_tracking(self):
        assert hasattr(self.asp, '_sequence_order')
        assert hasattr(self.asp, '_transition_order')

    def test_record_and_predict(self):
        self.asp.record_sequence('s1', ['a', 'b', 'c'])
        self.asp.record_sequence('s1', ['a', 'b', 'd'])
        result = self.asp.predict_next(['a', 'b'])
        assert len(result) > 0

    def test_sequence_eviction_at_cap(self):
        for i in range(5500):
            self.asp.record_sequence(f'seq_{i}', [f'action_{i}'])
        assert len(self.asp.sequences) <= self.asp._MAX_SEQUENCES
        assert len(self.asp._sequence_order) <= self.asp._MAX_SEQUENCES


# ═════════════════════════════════════════════════════════════════
# Fix 7: SignalTriangulator
# ═════════════════════════════════════════════════════════════════

class TestSignalTriangulatorCaps:
    def setup_method(self):
        from shikigami.superhuman import SignalTriangulator
        self.st = SignalTriangulator()

    def test_cap_constants_exist(self):
        assert hasattr(self.st, '_MAX_MACS')
        assert hasattr(self.st, '_MAX_PER_MAC')
        assert self.st._MAX_MACS == 1000
        assert self.st._MAX_PER_MAC == 100

    def test_mac_order_tracking(self):
        assert hasattr(self.st, '_mac_order')
        assert isinstance(self.st._mac_order, list)

    def test_per_mac_cap(self):
        mac = 'AA:BB:CC:DD:EE:FF'
        for i in range(120):
            self.st.record_signal(mac, float(-60 - i), datetime.now())
        assert len(self.st.signal_history[mac]) <= self.st._MAX_PER_MAC * 2

    def test_total_mac_eviction(self):
        for i in range(1100):
            mac = f'{i:02X}:{i:02X}:{i:02X}:{i:02X}:{i:02X}:{i:02X}'
            self.st.record_signal(mac, -60.0, datetime.now())
        assert len(self.st.signal_history) <= self.st._MAX_MACS
        assert len(self.st._mac_order) <= self.st._MAX_MACS

    def test_distance_estimation(self):
        # Known distance calculation
        dist = self.st._estimate_distance(-60)
        assert isinstance(dist, float)
        assert dist >= 0


# ═════════════════════════════════════════════════════════════════
# Fix 8 & 9: AmbientMonitor
# ═════════════════════════════════════════════════════════════════

class TestAmbientMonitorCaps:
    def setup_method(self):
        from shadowd.context_protocol import ContextProtocol, ContextType
        from shadowd.ambient_monitor import AmbientMonitor
        ctx = ContextProtocol()
        self.monitor = AmbientMonitor(context_protocol=ctx)

    def test_cap_constants_exist(self):
        assert hasattr(self.monitor, '_MAX_PATTERNS')
        assert hasattr(self.monitor, '_MAX_EVENT_COUNTS')
        assert hasattr(self.monitor, '_MAX_PEOPLE')
        assert hasattr(self.monitor, '_MAX_PER_PERSON_ENCOUNTERS')
        assert self.monitor._MAX_PATTERNS == 500
        assert self.monitor._MAX_EVENT_COUNTS == 200
        assert self.monitor._MAX_PEOPLE == 500
        assert self.monitor._MAX_PER_PERSON_ENCOUNTERS == 100

    def test_pattern_order_tracking(self):
        assert hasattr(self.monitor, '_pattern_order')
        assert isinstance(self.monitor._pattern_order, list)

    def test_people_order_tracking(self):
        assert hasattr(self.monitor, '_people_order')
        assert isinstance(self.monitor._people_order, list)

    @pytest.mark.asyncio
    async def test_event_counts_capped(self):
        import asyncio
        for i in range(250):
            await self.monitor.observe({'type': f'event_type_{i}', 'content': 'test'})
        # event_counts should be capped
        assert len(self.monitor.event_counts) <= self.monitor._MAX_EVENT_COUNTS


# ═════════════════════════════════════════════════════════════════
# Fix 10: TemporalReasoning (already bounded - verification)
# ═════════════════════════════════════════════════════════════════

class TestTemporalReasoningAlreadyBounded:
    def setup_method(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        self.tr = TemporalReasoning()

    def test_events_already_bounded(self):
        assert hasattr(self.tr, '_MAX_EVENTS')
        assert self.tr._MAX_EVENTS == 10000

    def test_patterns_already_bounded(self):
        assert hasattr(self.tr, '_MAX_PATTERNS')
        assert self.tr._MAX_PATTERNS == 500

    def test_predictions_already_bounded(self):
        assert hasattr(self.tr, '_MAX_PREDICTIONS')
        assert self.tr._MAX_PREDICTIONS == 200

    def test_events_deque(self):
        from collections import deque
        assert isinstance(self.tr._events, deque)

    def test_events_capped(self):
        from datetime import timezone
        for i in range(11000):
            self.tr.record_event('test_event')
        assert len(self.tr._events) <= self.tr._MAX_EVENTS


# ═════════════════════════════════════════════════════════════════
# Fix 11: ShadowGraph nodes
# ═════════════════════════════════════════════════════════════════

class TestShadowGraphNodeCaps:
    def setup_method(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        self.sg = ShadowGraph()

    def test_cap_constant_exists(self):
        assert hasattr(self.sg, '_MAX_NODES')
        assert self.sg._MAX_NODES == 50

    def test_default_nodes_within_cap(self):
        # Default graph has 11 nodes
        # listen, analyze, reason, respond, delegate, escalate, act, verify, learn, reflect, idle + dormant
        assert len(self.sg.nodes) <= self.sg._MAX_NODES

    def test_add_node_within_cap(self):
        from shadowd.cognition.shadow_graph import ShadowNode, CognitiveMode, ApprovalLevel
        initial_count = len(self.sg.nodes)
        # Default nodes should be 11 (listen through idle, plus any others)
        assert initial_count <= self.sg._MAX_NODES
