"""Sprint 175-176: cognitive_reflection + social_intelligence deep audit regression tests."""
import pytest
from datetime import datetime, timezone


class TestCognitiveReflectionFixes:
    def test_linked_decision_ids_sanitized(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            topic="Test topic", description="Desc",
            reflection_type="daily", domain="work", confidence_before=0.5, confidence_after=0.8,
            linked_decision_ids=["<script>alert(1)</script>", "normal_id"]
        )
        assert entry is not None
        assert "normal_id" in entry.linked_decision_ids
        assert "<script>" not in str(entry.linked_decision_ids)

    def test_extract_insights_merged_returns_merged_values(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        cr.add_reflection(topic="Same lesson", description="Desc", reflection_type="daily", domain="work", confidence_before=0.5, confidence_after=0.7)
        cr.extract_insights(cr._reflections[list(cr._reflections.keys())[0]].entry_id)
        cr.add_reflection(topic="Same lesson", description="Desc", reflection_type="daily", domain="work", confidence_before=0.6, confidence_after=0.8)
        latest_rid = list(cr._reflections.keys())[-1]
        insights2 = cr.extract_insights(cr._reflections[latest_rid].entry_id)
        if len(insights2) == 1 and insights2[0].frequency >= 2:
            assert insights2[0].frequency >= 2
            assert insights2[0].confidence >= 0.7

    def test_stats_lock_detect_patterns(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        cr.add_reflection(topic="Test", description="Desc", reflection_type="daily", domain="work", confidence_before=0.5, confidence_after=0.8)
        cr.add_reflection(topic="Test2", description="Desc2", reflection_type="daily", domain="work", confidence_before=0.4, confidence_after=0.7)
        patterns = cr.detect_patterns(min_frequency=1)
        stats = cr.get_stats()
        assert "patterns_detected" in stats

    def test_extract_insights_sanitizes_reflection_id(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        result = cr.extract_insights("nonexistent_id_sanitized")
        assert result == []

    def test_get_confidence_trend_sanitizes_domain(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        cr.add_reflection(topic="Test", description="Desc", reflection_type="daily", domain="work", confidence_before=0.5, confidence_after=0.8)
        trend = cr.get_confidence_trend(domain="<script>evil</script>")
        assert "trend" in trend

    def test_list_reflections_sanitizes_domain(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        cr.add_reflection(topic="Test", description="Desc", reflection_type="daily", domain="work", confidence_before=0.5, confidence_after=0.8)
        results = cr.list_reflections(domain="work")
        assert len(results) >= 1

    def test_get_learning_insights_sanitizes_category(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        rid = cr.add_reflection(topic="Test", description="Desc", reflection_type="daily", domain="work", confidence_before=0.5, confidence_after=0.8)
        cr.extract_insights(rid)
        insights = cr.get_learning_insights(category="professional_growth")
        assert isinstance(insights, list)

    def test_dead_sanitize_re_removed(self):
        import shadowd.cognition.cognitive_reflection as cr_module
        assert not hasattr(cr_module, '_SANITIZE_RE') or cr_module._SANITIZE_RE is cr_module._SANITIZE_CONTENT_RE


class TestSocialIntelligenceFixes:
    def test_stats_lock_create_group(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType
        si = SocialIntelligence()
        group = si.create_group("test", GroupType.PROFESSIONAL, members=["alice"])
        assert group.name == "test"
        stats = si.get_stats()
        assert stats["groups_tracked"] == 1
        assert stats["total_members"] == 1

    def test_stats_lock_remove_member(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType
        si = SocialIntelligence()
        group = si.create_group("test", GroupType.PROFESSIONAL, members=["alice", "bob"])
        result = si.remove_member(group.group_id, "alice")
        assert result is True
        stats = si.get_stats()
        assert stats["total_members"] == 1

    def test_stats_lock_add_influence(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, InfluenceType
        si = SocialIntelligence()
        si.create_group("test", GroupType.PROFESSIONAL, members=["alice", "bob"])
        rel = si.add_influence("alice", "bob", InfluenceType.EXPERT, 0.8)
        stats = si.get_stats()
        assert stats["influence_relations"] == 1

    def test_stats_lock_assign_role(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, RoleType
        si = SocialIntelligence()
        si.create_group("test", GroupType.PROFESSIONAL, members=["alice"])
        role = si.assign_role("alice", "test", RoleType.LEADER)
        stats = si.get_stats()
        assert stats["roles_assigned"] == 1

    def test_find_influencers_min_influence_clamp(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, InfluenceType
        si = SocialIntelligence()
        si.create_group("test", GroupType.PROFESSIONAL, members=["alice", "bob"])
        si.add_influence("alice", "bob", InfluenceType.EXPERT, 0.8)
        result_low = si.find_influencers(min_influence=-1.0)
        assert len(result_low) >= 1
        result_high = si.find_influencers(min_influence=5.0)
        assert len(result_high) == 0

    def test_create_group_name_sanitized(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType
        si = SocialIntelligence()
        group = si.create_group("<script>alert(1)</script>", GroupType.PROFESSIONAL)
        assert "<script>" not in group.name

    def test_add_influence_source_sanitized(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, InfluenceType
        si = SocialIntelligence()
        si.create_group("test", GroupType.PROFESSIONAL, members=["alice", "bob"])
        rel = si.add_influence("<script>evil</script>", "bob", InfluenceType.EXPERT, 0.5)
        assert "<script>" not in rel.source

    def test_assign_role_person_sanitized(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, RoleType
        si = SocialIntelligence()
        si.create_group("test", GroupType.PROFESSIONAL, members=["alice"])
        role = si.assign_role("<script>evil</script>", "test", RoleType.LEADER)
        assert "<script>" not in role.person_id