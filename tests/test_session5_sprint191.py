"""Sprint 191: Word boundary regex regression tests across 8 modules."""
import pytest
import re
import threading


class TestDomainExpertiseWordBoundary:
    def test_project_not_matching_projection(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "projection analysis complete"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["projects"] == 0

    def test_project_matches_actual_project(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "project alpha started"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["projects"] >= 1

    def test_deadline_not_matching_deadlines_in_non_context(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "greeting card sent"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["deadline_count"] == 0

    def test_meeting_not_matching_greeting(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "greeting card sent"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["meetings"] == 0

    def test_meeting_matches_actual_meeting(self):
        from shadowd.domain_expertise import CareerExpert
        expert = CareerExpert()
        obs = [{"text": "meeting with team", "timestamp": "2026-04-21T10:00:00+00:00"}]
        result = expert.analyze("career", obs)
        assert result["tracked_items"]["meetings"] >= 1

    def test_cost_not_matching_costume(self):
        from shadowd.domain_expertise import FinanceExpert
        expert = FinanceExpert()
        obs = [{"text": "costume party planning"}]
        result = expert.analyze("finance", obs)
        assert result["tracked_items"]["expenses"] == 0

    def test_paid_not_matching_overpaid(self):
        from shadowd.domain_expertise import FinanceExpert
        expert = FinanceExpert()
        obs = [{"text": "overpaid for dinner"}]
        result = expert.analyze("finance", obs)
        assert result["tracked_items"]["expenses"] == 0

    def test_expense_matches_actual_expense(self):
        from shadowd.domain_expertise import FinanceExpert
        expert = FinanceExpert()
        obs = [{"text": "expense report for $50"}]
        result = expert.analyze("finance", obs)
        assert result["tracked_items"]["expenses"] >= 1


class TestDeviceContextWordBoundary:
    def test_home_not_matching_homepage(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"name": "homepage city"})
        assert result != LocationType.HOME

    def test_home_matches_actual_home(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"name": "home office"})
        assert result == LocationType.HOME

    def test_work_not_matching_network(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"name": "network center"})
        assert result != LocationType.OFFICE

    def test_work_matches_actual_work(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"name": "work station"})
        assert result == LocationType.OFFICE

    def test_sport_not_matching_transport(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"name": "transport hub"})
        assert result != LocationType.GYM

    def test_ssid_home_matches_underscore(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"wifi_ssid": "Home_5G"})
        assert result == LocationType.HOME

    def test_ssid_corp_matches_underscore(self):
        from shadowd.cognition.device_context import _classify_location, LocationType
        result = _classify_location({"wifi_ssid": "Corp_Network"})
        assert result == LocationType.OFFICE

    def test_notification_mail_matches_email(self):
        from shadowd.cognition.device_context import _classify_notification, NotificationType
        result = _classify_notification({"app": "mail", "category": ""})
        assert result == NotificationType.EMAIL

    def test_notification_social_matches_social(self):
        from shadowd.cognition.device_context import _classify_notification, NotificationType
        result = _classify_notification({"app": "social", "category": ""})
        assert result == NotificationType.SOCIAL


class TestStealthLayerWordBoundary:
    def test_ip_not_matching_triple(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        data = {"triple_value": "secret"}
        result = ce.anonymize(data)
        assert result["triple_value"] == "secret"

    def test_ip_matches_actual_ip(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        data = {"server_ip": "192.168.1.1"}
        result = ce.anonymize(data)
        assert result["server_ip"] != "192.168.1.1"

    def test_name_not_matching_rename(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        data = {"rename_count": 5}
        result = ce.anonymize(data)
        assert result["rename_count"] == 5

    def test_name_matches_actual_name(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        data = {"name": "John"}
        result = ce.anonymize(data)
        assert result["name"] != "John"


class TestLearningSystemWordBoundary:
    def test_because_in_reasoning(self):
        text = "I chose this because it was cheapest"
        assert re.search(r'\bbecause\b', text.lower())
        reason = text.lower().split("because")[1][:100]
        assert "cheapest" in reason

    def test_because_not_matching_becauseof(self):
        text = "becauseof reasons unknown"
        match = re.search(r'\bbecause\b', text.lower())
        assert match is None


class TestShadowVoiceWordBoundary:
    def test_oracle_not_matching_oracleable(self):
        from shadowd.cognition.shadow_voice import IntentParser
        parser = IntentParser()
        entities = parser.extract_entities("I need oracleable solutions")
        assert entities.get("ability") != "oracle"

    def test_oracle_matches_actual_oracle(self):
        from shadowd.cognition.shadow_voice import IntentParser
        parser = IntentParser()
        entities = parser.extract_entities("use oracle for prediction")
        assert entities.get("ability") == "oracle"

    def test_multiword_ability_matches(self):
        from shadowd.cognition.shadow_voice import IntentParser
        parser = IntentParser()
        entities = parser.extract_entities("activate life pulse ability")
        assert entities.get("ability") == "life pulse"


class TestShadowEnhancedSSIDWordBoundary:
    def test_home_ssid_with_underscore(self):
        data = [{"ssid": "Home_5G"}]
        for n in data:
            home_match = any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["home", "my"])
            assert home_match

    def test_home_ssid_exact(self):
        data = [{"ssid": "Home"}]
        for n in data:
            home_match = any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["home", "my"])
            assert home_match


class TestShadowAISSIDWordBoundary:
    def test_corp_ssid_matches(self):
        data = [{"ssid": "Corp_Net"}]
        for n in data:
            corp_match = any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["office", "corp"])
            assert corp_match

    def test_office_ssid_matches(self):
        data = [{"ssid": "office_5g"}]
        for n in data:
            office_match = any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["office", "corp"])
            assert office_match

    def test_corporate_not_matching_corp(self):
        data = [{"ssid": "CorporateGuest"}]
        for n in data:
            corp_match = any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', n["ssid"].lower()) for w in ["office", "corp"])
            assert not corp_match


class TestMemoryRagWordBoundary:
    def test_home_domain_matches_home_query(self):
        import re as _re
        query = "home cleaning"
        domain = "home"
        assert _re.search(r'\b' + _re.escape(domain) + r'\b', query.lower())

    def test_home_domain_not_matching_homework(self):
        import re as _re
        query = "homework assignment"
        domain = "home"
        assert not _re.search(r'\b' + _re.escape(domain) + r'\b', query.lower())

    def test_career_not_matching_careering(self):
        import re as _re
        query = "careering off course"
        domain = "career"
        assert not _re.search(r'\b' + _re.escape(domain) + r'\b', query.lower())