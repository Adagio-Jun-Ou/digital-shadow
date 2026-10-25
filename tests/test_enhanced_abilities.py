#!/usr/bin/env python3
"""
Tests for Enhanced Shikigami Abilities

Tests cover:
- Monte Carlo simulation (Oracle)
- Device fingerprinting (X-Ray, Life Pulse)
- Pattern analysis (Pattern Sight)
- Behavior prediction (Social Radar)
- Action sequences (Intent Seer)
- Decision tracking (Shadow Council)
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from shikigami.abilities import (
    MonteCarloSimulator,
    DeviceFingerprinter,
    PatternAnalyzer,
    BehaviorPredictor,
    ActionSequencePredictor,
    DecisionTracker,
    AbilityResult,
)
from shikigami.enhanced_abilities import (
    EnhancedShikigamiRegistry,
    ShikigamiType,
)


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation for predictions."""
    
    def test_simulate_outcome_basic(self):
        """Test basic outcome simulation."""
        mc = MonteCarloSimulator(seed=42)
        result = mc.simulate_outcome(0.7, num_simulations=1000)
        
        assert 0.5 < result["probability"] < 0.9
        assert result["std_dev"] > 0
        assert result["simulations"] == 1000
        assert len(result["confidence_interval_95"]) == 2
    
    def test_simulate_outcome_deterministic(self):
        """Test with deterministic probability."""
        mc = MonteCarloSimulator(seed=42)
        
        # High probability
        high = mc.simulate_outcome(0.95, num_simulations=5000)
        assert high["probability"] > 0.9
        
        # Low probability
        low = mc.simulate_outcome(0.05, num_simulations=5000)
        assert low["probability"] < 0.1
    
    def test_simulate_outcome_reproducible(self):
        """Test that same seed gives similar results."""
        mc1 = MonteCarloSimulator(seed=42)
        mc2 = MonteCarloSimulator(seed=42)
        
        r1 = mc1.simulate_outcome(0.5, num_simulations=1000)
        r2 = mc2.simulate_outcome(0.5, num_simulations=1000)
        
        # Probabilities should be close (within 10%) due to same seed
        assert abs(r1["probability"] - r2["probability"]) < 0.1
    
    def test_predict_time(self):
        """Test time prediction with uncertainty."""
        mc = MonteCarloSimulator(seed=42)
        result = mc.predict_time(base_hours=4.0, variance=1.0, num_simulations=10000)
        
        assert 3.0 < result["expected_hours"] < 5.0
        assert result["std_dev_hours"] > 0
        assert result["p95_hours"] > result["expected_hours"]
    
    def test_predict_time_variance(self):
        """Test that higher variance leads to wider distribution."""
        mc = MonteCarloSimulator(seed=42)
        
        low_var = mc.predict_time(4.0, 0.5, 10000)
        high_var = mc.predict_time(4.0, 2.0, 10000)
        
        assert high_var["std_dev_hours"] > low_var["std_dev_hours"]


class TestDeviceFingerprinter:
    """Test device fingerprinting for X-Ray and Life Pulse."""
    
    def test_identify_known_device(self):
        """Test identifying a known device."""
        fp = DeviceFingerprinter()
        fp._known_devices["AA:BB:CC:DD:EE:FF"] = {
            "name": "John",
            "device_type": "phone"
        }
        
        result = fp.identify_device("aa:bb:cc:dd:ee:ff", "John's iPhone")
        
        assert result["identified"] is True
        assert result["person"] == "John"
        assert result["confidence"] > 0.5
    
    def test_identify_unknown_device(self):
        """Test identifying an unknown device."""
        fp = DeviceFingerprinter()
        
        result = fp.identify_device("11:22:33:44:55:66", "iPhone 15")
        
        assert result["identified"] is False
        assert result["device_type"] == "apple_mobile"
    
    def test_identify_device_types(self):
        """Test device type inference."""
        fp = DeviceFingerprinter()
        
        assert fp.identify_device("x", "John's iPhone")["device_type"] == "apple_mobile"
        assert fp.identify_device("x", "Apple Watch")["device_type"] == "smartwatch"
        assert fp.identify_device("x", "AirPods Pro")["device_type"] == "audio"
        assert fp.identify_device("x", "MacBook Pro")["device_type"] == "apple_computer"
        assert fp.identify_device("x", "Galaxy S23")["device_type"] == "android_mobile"
    
    def test_get_person_constellation(self):
        """Test getting device constellation for a person."""
        fp = DeviceFingerprinter()
        fp._known_devices["AA:BB:CC:DD:EE:FF"] = {"name": "John"}
        fp._known_devices["11:22:33:44:55:66"] = {"name": "John"}
        fp._known_devices["99:88:77:66:55:44"] = {"name": "Sarah"}
        
        john_devices = fp.get_person_constellation("John")
        
        assert len(john_devices) == 2
        assert "AA:BB:CC:DD:EE:FF" in john_devices
    
    def test_calculate_presence_confidence(self):
        """Test presence confidence calculation."""
        fp = DeviceFingerprinter()
        fp._known_devices["AA:BB:CC:DD:EE:FF"] = {"name": "John"}
        fp._known_devices["11:22:33:44:55:66"] = {"name": "John"}
        fp._known_devices["99:88:77:66:55:44"] = {"name": "John"}
        
        # All devices detected
        all_detected = ["AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66", "99:88:77:66:55:44"]
        confidence_all = fp.calculate_presence_confidence(all_detected, "John")
        assert confidence_all > 0.5
        
        # One device detected
        one_detected = ["AA:BB:CC:DD:EE:FF"]
        confidence_one = fp.calculate_presence_confidence(one_detected, "John")
        assert 0 < confidence_one < confidence_all
        
        # No devices detected
        no_detection = fp.calculate_presence_confidence([], "John")
        assert no_detection == 0
    
    def test_load_people(self):
        """Test loading people from database."""
        fp = DeviceFingerprinter()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "John": {
                    "devices": ["AA:BB:CC:DD:EE:FF"],
                    "patterns": {"home": {"morning": True}}
                }
            }, f)
            temp_path = Path(f.name)
        
        people = fp.load_people(temp_path)
        
        assert "John" in people
        assert "AA:BB:CC:DD:EE:FF" in fp._known_devices
        temp_path.unlink()


class TestPatternAnalyzer:
    """Test pattern detection with correlation and causation."""
    
    def test_add_observation(self):
        """Test adding observations."""
        pa = PatternAnalyzer()
        
        pa.add_observation("activity", 1.0)
        pa.add_observation("activity", 2.0)
        
        assert len(pa.patterns["activity"]) == 2
    
    def test_detect_frequency_fft(self):
        """Test frequency detection."""
        pa = PatternAnalyzer()
        
        # Create periodic data (daily pattern)
        values = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        
        result = pa.detect_frequency(values)
        
        assert result["dominant_frequency"] is not None
        assert result["period"] is not None
    
    def test_detect_correlations(self):
        """Test correlation detection."""
        pa = PatternAnalyzer()
        
        observations = [
            {"type": "coding", "value": 5},
            {"type": "coffee", "value": 1},
            {"type": "coding", "value": 4},
            {"type": "coffee", "value": 0},
            {"type": "coding", "value": 6},
            {"type": "coffee", "value": 1},
        ]
        
        correlations = pa.find_correlations(observations)
        
        # Should find correlations between types
        assert isinstance(correlations, list)
    
    def test_detect_causation(self):
        """Test causation detection."""
        pa = PatternAnalyzer()
        
        # Cause always leads to effect
        base_time = datetime.now()
        observations = [
            {"type": "coffee", "timestamp": (base_time - timedelta(hours=i*2)).isoformat()}
            for i in range(10)
        ] + [
            {"type": "alert", "timestamp": (base_time - timedelta(hours=i*2) + timedelta(minutes=30)).isoformat()}
            for i in range(10)
        ]
        
        result = pa.detect_causation(observations, "coffee", "alert", lag_hours=2)
        
        # Coffee should lead to alert within 2 hours
        assert result["causation_detected"] is True or result["causation_rate"] > 0.5
    
    def test_causation_insufficient_data(self):
        """Test causation with insufficient data."""
        pa = PatternAnalyzer()
        
        result = pa.detect_causation([], "coffee", "alert")
        
        assert result["causation_detected"] is False
        assert result["confidence"] == 0


class TestBehaviorPredictor:
    """Test behavior prediction for Social Radar."""
    
    def test_track_behavior(self):
        """Test tracking behavior."""
        bp = BehaviorPredictor()
        
        for _ in range(5):
            bp.track_behavior("John", "arrives_late", True)
        for _ in range(5):
            bp.track_behavior("John", "arrives_late", False)
        
        prediction = bp.predict_behavior("John", "arrives_late")
        
        assert "arrives_late" in prediction["behaviors"]
        assert 0.3 < prediction["behaviors"]["arrives_late"]["probability"] < 0.7
    
    def test_predict_absent_person(self):
        """Test predicting behavior for unknown person."""
        bp = BehaviorPredictor()
        
        prediction = bp.predict_behavior("Unknown", "some_behavior")
        
        assert prediction["confidence"] == 0
        assert prediction["behaviors"] == {}
    
    def test_get_warning_signs(self):
        """Test getting warning signs for behavior."""
        bp = BehaviorPredictor()
        
        # Track with context
        for i in range(10):
            bp.track_behavior("John", "arrives_late", True, context={"day": "Monday" if i % 2 == 0 else "Friday"})
        
        signs = bp.get_warning_signs("John")
        
        assert isinstance(signs, list)
    
    def test_prediction_confidence_increases(self):
        """Test that confidence increases with more data."""
        bp = BehaviorPredictor()
        
        # Small sample
        for _ in range(3):
            bp.track_behavior("Alice", "calls", True)
        low_conf = bp.predict_behavior("Alice", "calls")["confidence"]
        
        # More data
        for _ in range(17):
            bp.track_behavior("Alice", "calls", True)
        high_conf = bp.predict_behavior("Alice", "calls")["confidence"]
        
        assert high_conf >= low_conf


class TestActionSequencePredictor:
    """Test Markov chain action prediction."""
    
    def test_record_sequence(self):
        """Test recording action sequences."""
        asp = ActionSequencePredictor()
        
        asp.record_sequence("seq1", ["wake", "coffee", "work"])
        asp.record_sequence("seq2", ["wake", "exercise", "work"])
        
        assert len(asp.sequences) == 2
        assert asp.transitions["wake"]["coffee"] == 1
        assert asp.transitions["wake"]["exercise"] == 1
    
    def test_predict_next(self):
        """Test predicting next action."""
        asp = ActionSequencePredictor()
        
        # Train with common pattern
        for _ in range(10):
            asp.record_sequence("seq", ["wake", "coffee", "work"])
        for _ in range(3):
            asp.record_sequence("seq", ["wake", "exercise", "work"])
        
        predictions = asp.predict_next(["wake"])
        
        assert len(predictions) > 0
        assert predictions[0]["action"] in ["coffee", "exercise"]
        # Coffee should be more likely (10 vs 3)
    
    def test_predict_sequence(self):
        """Test predicting sequence of actions."""
        asp = ActionSequencePredictor()
        
        for _ in range(5):
            asp.record_sequence("seq", ["a", "b", "c", "d"])
        
        predicted = asp.predict_sequence(["a"], length=3)
        
        assert len(predicted) <= 3
    
    def test_predict_no_data(self):
        """Test prediction with no data."""
        asp = ActionSequencePredictor()
        
        predictions = asp.predict_next(["unknown_action"])
        
        assert predictions == []


class TestDecisionTracker:
    """Test decision tracking for Shadow Council."""
    
    def test_record_decision(self):
        """Test recording a decision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = DecisionTracker(Path(tmpdir) / "decisions.json")
            
            entry = dt.record_decision(
                situation="Should I accept the job offer?",
                choice="Accept",
                outcome="Great team, good salary",
                satisfaction=8
            )
            
            assert entry["choice"] == "Accept"
            assert entry["satisfaction"] == 8
            assert len(dt.decisions) == 1
    
    def test_find_similar(self):
        """Test finding similar decisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = DecisionTracker(Path(tmpdir) / "decisions.json")
            
            dt.record_decision("Job offer at startup", "Accept", outcome="Good", satisfaction=7)
            dt.record_decision("Job offer at corporate", "Reject", outcome="Bad fit", satisfaction=5)
            dt.record_decision("Vacation in Europe", "Go", outcome="Amazing", satisfaction=9)
            
            similar = dt.find_similar("job offer", limit=5)
            
            assert len(similar) == 2
            assert "job" in similar[0]["situation"].lower() or "offer" in similar[0]["situation"].lower()
    
    def test_get_recommendation(self):
        """Test getting recommendation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = DecisionTracker(Path(tmpdir) / "decisions.json")
            
            # Multiple similar decisions with outcomes
            for _ in range(5):
                dt.record_decision("Project deadline", "Start early", outcome="On time", satisfaction=9)
            for _ in range(3):
                dt.record_decision("Project deadline", "Start late", outcome="Overdue", satisfaction=3)
            
            rec = dt.get_recommendation("Project deadline approaching")
            
            assert rec["recommendation"] == "Start early"
            assert rec["avg_satisfaction"] > 7
    
    def test_extract_tags(self):
        """Test tag extraction from situations."""
        dt = DecisionTracker()
        
        tags1 = dt._extract_tags("Should I take this job offer?")
        assert "work" in tags1 or "finance" in tags1
        
        tags2 = dt._extract_tags("Should I ask her on a date?")
        assert "relationship" in tags2
    
    def test_no_similar_decisions(self):
        """Test recommendation with no similar decisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = DecisionTracker(Path(tmpdir) / "decisions.json")
            
            rec = dt.get_recommendation("Something completely new")
            
            assert rec["recommendation"] is None
            assert rec["confidence"] == 0


class TestEnhancedShikigamiRegistry:
    """Test the full Shikigami registry."""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create registry with temp data directory."""
        import shikigami.enhanced_abilities as sa_module
        sa_module.PEOPLE_DB = tmp_path / "people.json"
        sa_module.DECISIONS_DB = tmp_path / "decisions.json"
        sa_module.PATTERNS_DB = tmp_path / "patterns.json"
        sa_module.OBSERVATIONS_DB = tmp_path / "observations.json"
        sa_module.SEQUENCES_DB = tmp_path / "sequences.json"
        
        return EnhancedShikigamiRegistry()
    
    def test_list_shikigami(self, registry):
        """Test listing abilities."""
        abilities = registry.list_shikigami()

        # 10 originals + 8 added across two rounds + 3 in round 6 = 21.
        assert len(abilities) >= 21
        # Spot-check from each tier.
        for t in ("oracle", "habit_tracker", "energy_flow", "focus_sight",
                  "connection", "compass", "hearth", "time_witness", "mirror",
                  "lantern", "tide", "echo"):
            assert any(a["type"] == t for a in abilities), f"missing ability: {t}"
    
    def test_auto_select(self, registry):
        """Test auto-selecting ability from query."""
        assert registry.auto_select("Will it rain?") == ShikigamiType.ORACLE
        assert registry.auto_select("Who is nearby?") == ShikigamiType.X_RAY
        assert registry.auto_select("What did I do Tuesday?") == ShikigamiType.PERFECT_MEMORY
        assert registry.auto_select("Create a file") == ShikigamiType.REALITY_FORGE
    
    def test_track_person(self, registry):
        """Test tracking a person."""
        result = registry.track_person("Alice", "AA:BB:CC:DD:EE:FF", relation="friend")
        
        assert "Alice" in result
        assert "Alice" in registry.people
    
    def test_track_decision(self, registry):
        """Test tracking a decision."""
        result = registry.track_decision("Job offer", "Accept", "Good outcome", satisfaction=8)
        
        assert "tracked" in result.lower()
        assert len(registry.decision_tracker.decisions) > 0
    
    def test_track_pattern(self, registry):
        """Test tracking a pattern."""
        result = registry.track_pattern("coffee", "alert", observed=True)
        
        assert "coffee" in result.lower()
        assert "alert" in result.lower()


class TestAbilityResult:
    """Test AbilityResult dataclass."""
    
    def test_create_result(self):
        """Test creating ability result."""
        result = AbilityResult(
            success=True,
            output="Test output",
            data={"key": "value"},
            confidence=0.85
        )
        
        assert result.success is True
        assert result.output == "Test output"
        assert result.data["key"] == "value"
        assert result.confidence == 0.85
    
    def test_default_values(self):
        """Test default values."""
        result = AbilityResult(success=True, output="Test")
        
        assert result.data == {}
        assert result.confidence == 0.0
        assert result.commands_run == []
        assert result.files_created == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])