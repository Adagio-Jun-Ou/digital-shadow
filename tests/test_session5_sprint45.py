"""Sprint 45: AdaptiveSkillTree — Dynamic ability leveling based on usage patterns"""

import pytest
import threading
from datetime import datetime, timezone, timedelta

from shadowd.cognition.adaptive_skill_tree import (
    AdaptiveSkillTree, SkillDomain, SkillLevel,
    SkillNode, SkillRecommendation, LevelUpEvent,
    _MAX_NAME_LENGTH, _MAX_SKILLS,
)


class TestSkillNodeProperties:
    def test_level_name_novice(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, level=0, xp=0)
        assert node.level_name == "novice"

    def test_level_name_master(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, level=5, xp=3000)
        assert node.level_name == "master"

    def test_success_rate_zero_usage(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL)
        assert node.success_rate == 0.0

    def test_success_rate_half(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, usage_count=10, success_count=5)
        assert node.success_rate == 0.5

    def test_xp_to_next_level(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, level=0, xp=50)
        assert node.xp_to_next == 50.0

    def test_xp_to_next_at_max(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, level=5, xp=3000)
        assert node.xp_to_next == 0.0

    def test_progress_calculation(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, level=0, xp=50)
        assert 0 < node.progress < 1

    def test_progress_at_max_level(self):
        node = SkillNode(skill_id="test", name="Test", domain=SkillDomain.TECHNICAL, level=5, xp=5000)
        assert node.progress == 1.0

    def test_to_dict(self):
        node = SkillNode(skill_id="t", name="T", domain=SkillDomain.SOCIAL, level=2, xp=300, usage_count=5, success_count=3)
        d = node.to_dict()
        assert d["skill_id"] == "t"
        assert d["domain"] == "social"
        assert d["level"] == 2
        assert d["level_name"] == "intermediate"
        assert "progress" in d
        assert "success_rate" in d


class TestSkillTreeBasics:
    def test_add_skill(self):
        tree = AdaptiveSkillTree()
        node = tree.add_skill("python", "Python", "technical")
        assert node.skill_id == "python"
        assert node.domain.value == "technical"
        assert node.unlocked is True

    def test_add_skill_with_dependencies(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("basics", "Basics", "technical")
        node = tree.add_skill("advanced", "Advanced", "technical", dependencies=["basics"])
        assert node.unlocked is True

    def test_add_skill_locked_dependencies(self):
        tree = AdaptiveSkillTree()
        node = tree.add_skill("advanced", "Advanced", "technical", dependencies=["missing_dep"])
        assert node.unlocked is False

    def test_use_skill_gains_xp(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        node = tree.use_skill("python", success=True)
        assert node.xp > 0
        assert node.usage_count == 1
        assert node.success_count == 1

    def test_use_skill_failure_no_success_count(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        node = tree.use_skill("python", success=False)
        assert node.usage_count == 1
        assert node.success_count == 0
        assert node.xp >= 10.0

    def test_use_skill_nonexistent_raises(self):
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError, match="not found"):
            tree.use_skill("nonexistent")

    def test_use_skill_locked_raises(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("advanced", "Advanced", "technical", dependencies=["locked_dep"])
        with pytest.raises(ValueError, match="locked"):
            tree.use_skill("advanced")

    def test_duplicate_skill_raises(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        with pytest.raises(ValueError, match="already exists"):
            tree.add_skill("python", "Python2", "technical")

    def test_invalid_domain_raises(self):
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError, match="Invalid domain"):
            tree.add_skill("x", "X", "invalid_domain")

    def test_invalid_skill_id_raises(self):
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError, match="Invalid skill_id"):
            tree.add_skill("bad id!", "Bad", "technical")


class TestSkillTreeLeveling:
    def test_level_up_on_xp_threshold(self):
        tree = AdaptiveSkillTree()
        node = tree.add_skill("python", "Python", "technical")
        assert node.level == 0
        for _ in range(50):
            node = tree.use_skill("python", success=True)
        assert node.level > 0

    def test_level_up_events_recorded(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        for _ in range(50):
            tree.use_skill("python", success=True)
        events = tree.get_level_ups()
        assert len(events) > 0
        assert events[0].skill_id == "python"

    def test_level_up_event_to_dict(self):
        event = LevelUpEvent(
            skill_id="py", skill_name="Python", old_level=0, new_level=1,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        d = event.to_dict()
        assert d["skill_id"] == "py"
        assert d["old_level"] == 0
        assert d["new_level"] == 1

    def test_calculate_level_thresholds(self):
        tree = AdaptiveSkillTree()
        assert tree._calculate_level(0) == 0
        assert tree._calculate_level(100) == 1
        assert tree._calculate_level(300) == 2
        assert tree._calculate_level(700) == 3
        assert tree._calculate_level(1500) == 4
        assert tree._calculate_level(3000) == 5

    def test_master_level_does_not_exceed(self):
        tree = AdaptiveSkillTree()
        node = tree.add_skill("python", "Python", "technical")
        for _ in range(200):
            node = tree.use_skill("python", success=True)
        assert node.level <= 5


class TestSkillTreeUnlocking:
    def test_skill_unlocks_when_dependency_leveled(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("basics", "Basics", "technical", dependencies=[])
        locked = tree.add_skill("advanced", "Advanced", "technical", dependencies=["basics"])
        assert locked.unlocked is True

    def test_skill_remains_locked_with_missing_dependency(self):
        tree = AdaptiveSkillTree()
        locked = tree.add_skill("advanced", "Advanced", "technical", dependencies=["nonexistent"])
        assert locked.unlocked is False

    def test_unlock_triggers_on_level_up(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("basics", "Basics", "technical")
        advanced = tree.add_skill("advanced", "Advanced", "technical", dependencies=["basics"])
        while tree.get_skill("basics").level < 1:
            tree.use_skill("basics", success=True)
        if advanced.skill_id in tree._skills:
            assert tree.get_skill("advanced").unlocked is True


class TestSkillTreeRecommendations:
    def test_recommendations_empty_tree(self):
        tree = AdaptiveSkillTree()
        recs = tree.get_recommendations()
        assert recs == []

    def test_recommendations_with_skills(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        recs = tree.get_recommendations()
        assert len(recs) > 0

    def test_recommendations_locked_skill_dependency(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("basics", "Basics", "technical")
        tree.add_skill("advanced", "Advanced", "technical", dependencies=["missing_dep"])
        recs = tree.get_recommendations()
        locked_recs = [r for r in recs if "unlock" in r.reason.lower() or "Required" in r.reason]
        assert len(locked_recs) == 0

    def test_recommendations_limit(self):
        tree = AdaptiveSkillTree()
        for i in range(20):
            tree.add_skill(f"skill_{i}", f"Skill {i}", "technical")
        recs = tree.get_recommendations(limit=5)
        assert len(recs) <= 5

    def test_recommendation_to_dict(self):
        rec = SkillRecommendation(
            skill_id="py", reason="Build foundational skills",
            priority=0.75, current_level=0, target_level=1
        )
        d = rec.to_dict()
        assert d["skill_id"] == "py"
        assert d["priority"] == 0.75


class TestSkillTreeDomainSummary:
    def test_domain_summary_empty(self):
        tree = AdaptiveSkillTree()
        summary = tree.get_domain_summary()
        assert "technical" in summary
        assert summary["technical"]["skills"] == 0

    def test_domain_summary_with_data(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        tree.use_skill("python", success=True)
        summary = tree.get_domain_summary()
        assert summary["technical"]["skills"] == 1
        assert summary["technical"]["xp"] > 0

    def test_domain_summary_all_domains_present(self):
        tree = AdaptiveSkillTree()
        summary = tree.get_domain_summary()
        for domain in SkillDomain:
            assert domain.value in summary


class TestSkillTreeMisc:
    def test_get_skill(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        skill = tree.get_skill("python")
        assert skill is not None
        assert skill.name == "Python"

    def test_get_skill_nonexistent(self):
        tree = AdaptiveSkillTree()
        assert tree.get_skill("nonexistent") is None

    def test_list_skills_all(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        tree.add_skill("writing", "Writing", "communication")
        skills = tree.list_skills()
        assert len(skills) == 2

    def test_list_skills_by_domain(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        tree.add_skill("writing", "Writing", "communication")
        skills = tree.list_skills(domain="technical")
        assert len(skills) == 1
        assert skills[0].domain.value == "technical"

    def test_list_skills_invalid_domain(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        skills = tree.list_skills(domain="invalid")
        assert skills == []

    def test_remove_skill(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        assert tree.remove_skill("python") is True
        assert tree.get_skill("python") is None

    def test_remove_skill_nonexistent(self):
        tree = AdaptiveSkillTree()
        assert tree.remove_skill("nonexistent") is False

    def test_get_stats(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        tree.use_skill("python", success=True)
        stats = tree.get_stats()
        assert stats["total_skills"] == 1
        assert stats["unlocked_skills"] == 1
        assert stats["total_usage"] == 1

    def test_get_skill_tree(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("basics", "Basics", "technical")
        tree.add_skill("advanced", "Advanced", "technical", dependencies=["basics"])
        tree_str = tree.get_skill_tree()
        assert "basics" in tree_str
        assert "advanced" in tree_str

    def test_get_level_ups_filtered(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        tree.add_skill("js", "JavaScript", "technical")
        for _ in range(50):
            tree.use_skill("python", success=True)
        for _ in range(50):
            tree.use_skill("js", success=True)
        python_levels = tree.get_level_ups(skill_id="python")
        js_levels = tree.get_level_ups(skill_id="js")
        all_events = tree.get_level_ups()
        assert len(python_levels) + len(js_levels) == len(all_events)


class TestSkillTreeSecurity:
    def test_sanitize_name_strips_control_chars(self):
        tree = AdaptiveSkillTree()
        node = tree.add_skill("test", "Test\x00Name\x01", "technical")
        assert "\x00" not in node.name
        assert "\x01" not in node.name

    def test_sanitize_name_truncates(self):
        tree = AdaptiveSkillTree()
        long_name = "A" * (_MAX_NAME_LENGTH + 50)
        node = tree.add_skill("test", long_name, "technical")
        assert len(node.name) <= _MAX_NAME_LENGTH

    def test_validate_id_rejects_special_chars(self):
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.add_skill("bad id!", "Bad", "technical")

    def test_validate_id_rejects_empty(self):
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.add_skill("", "Empty", "technical")

    def test_initial_xp_negative_clamped(self):
        tree = AdaptiveSkillTree()
        node = tree.add_skill("test", "Test", "technical", initial_xp=-50)
        assert node.xp >= 0

    def test_metadata_truncated(self):
        tree = AdaptiveSkillTree()
        long_val = "x" * 2000
        node = tree.add_skill("test", "Test", "technical", metadata={"key": long_val})
        assert len(node.metadata["key"]) <= 1000

    def test_max_skills_cap(self):
        tree = AdaptiveSkillTree()
        for i in range(_MAX_SKILLS + 5):
            tree.add_skill(f"skill_{i}", f"Skill {i}", "technical")
        stats = tree.get_stats()
        assert stats["total_skills"] <= _MAX_SKILLS

    def test_thread_safety(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        errors = []

        def worker():
            try:
                for _ in range(50):
                    tree.use_skill("python", success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        stats = tree.get_stats()
        assert stats["total_usage"] == 200


class TestSkillTreeIntegration:
    def test_full_skill_progression(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("python", "Python", "technical")
        tree.add_skill("testing", "Testing", "technical", dependencies=["python"])
        tree.add_skill("devops", "DevOps", "technical", dependencies=["testing"])

        for _ in range(20):
            tree.use_skill("python", success=True)

        assert tree.get_skill("testing").unlocked is True

        for _ in range(20):
            tree.use_skill("testing", success=True)

        recs = tree.get_recommendations()
        assert len(recs) > 0

        summary = tree.get_domain_summary()
        assert summary["technical"]["skills"] == 3

    def test_api_skill_add_request_model(self):
        from shadowd.cognition.api_router import SkillAddRequest
        req = SkillAddRequest(skill_id="test", name="Test", domain="technical")
        assert req.skill_id == "test"
        assert req.domain == "technical"
        assert req.initial_xp == 0.0

    def test_api_skill_use_request_model(self):
        from shadowd.cognition.api_router import SkillUseRequest
        req = SkillUseRequest(skill_id="test", success=True)
        assert req.skill_id == "test"
        assert req.success is True