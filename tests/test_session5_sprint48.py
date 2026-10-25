"""Sprint 48: Integration + security hardening of AdaptiveSkillTree and NarrativeMemory"""

import pytest
import threading


class TestSkillTreeIntegration:
    def test_skill_tree_in_shadow_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        assert hasattr(os, 'skills')
        assert os.skills is not None

    def test_skill_tree_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os._running = True
        os._start_time = 0
        health = os.get_health()
        assert "skill_tree" in health["modules"]

    def test_skill_tree_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        status = os.get_status()
        assert "skill_stats" in status

    def test_skill_tree_api_endpoints(self):
        from shadowd.cognition.api_router import router
        routes = [r.path for r in router.routes]
        assert "/os/skills/add" in routes
        assert "/os/skills/use" in routes
        assert "/os/skills/{skill_id}" in routes
        assert "/os/skills" in routes
        assert "/os/skills/recommendations" in routes
        assert "/os/skills/domain-summary" in routes
        assert "/os/skills/level-ups" in routes
        assert "/os/skill-tree" in routes
        assert "/os/skills/stats" in routes

    def test_api_add_skill_request_validation(self):
        from shadowd.cognition.api_router import SkillAddRequest
        req = SkillAddRequest(skill_id="python", name="Python", domain="technical", initial_xp=50.0)
        assert req.skill_id == "python"
        assert req.initial_xp == 50.0

    def test_api_skill_use_request_validation(self):
        from shadowd.cognition.api_router import SkillUseRequest
        req = SkillUseRequest(skill_id="python", success=True)
        assert req.skill_id == "python"
        assert req.success is True


class TestNarrativeMemoryIntegration:
    def test_narrative_in_shadow_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        assert hasattr(os, 'narrative')
        assert os.narrative is not None

    def test_narrative_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os._running = True
        os._start_time = 0
        health = os.get_health()
        assert "narrative_memory" in health["modules"]

    def test_narrative_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        status = os.get_status()
        assert "narrative_stats" in status

    def test_narrative_api_endpoints(self):
        from shadowd.cognition.api_router import router
        routes = [r.path for r in router.routes]
        assert "/os/narrative/event" in routes
        assert "/os/narrative/chapter" in routes
        assert "/os/narrative/chapter/{chapter_id}" in routes
        assert "/os/narrative/arc" in routes
        assert "/os/narrative/arc/{arc_id}" in routes
        assert "/os/narrative/chapters" in routes
        assert "/os/narrative/arcs" in routes
        assert "/os/narrative/themes" in routes
        assert "/os/narrative/search" in routes
        assert "/os/narrative/summary" in routes
        assert "/os/narrative/stats" in routes

    def test_api_narrative_event_request(self):
        from shadowd.cognition.api_router import NarrativeEventRequest
        req = NarrativeEventRequest(content="Started a new project", importance=0.7)
        assert req.content == "Started a new project"
        assert req.importance == 0.7

    def test_api_narrative_chapter_request(self):
        from shadowd.cognition.api_router import NarrativeChapterRequest
        req = NarrativeChapterRequest(title="Career Growth", theme="career")
        assert req.title == "Career Growth"
        assert req.theme == "career"


class TestSecurityAdaptiveSkillTree:
    def test_skill_id_validation(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.add_skill("bad id!", "Bad", "technical")

    def test_skill_id_max_length(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.add_skill("a" * 65, "Long", "technical")

    def test_skill_name_sanitization(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        node = tree.add_skill("test", "Hello\x00World\x01", "technical")
        assert "\x00" not in node.name
        assert "\x01" not in node.name

    def test_metadata_size_cap(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        long_val = "x" * 2000
        node = tree.add_skill("test", "Test", "technical", metadata={"key": long_val})
        assert len(node.metadata["key"]) <= 1000

    def test_domain_validation(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.add_skill("test", "Test", "invalid_domain")

    def test_use_nonexistent_skill_raises(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.use_skill("nonexistent")

    def test_locked_skill_use_raises(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        with pytest.raises(ValueError):
            tree.use_skill("locked_skill")

    def test_level_cap_at_master(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        tree = AdaptiveSkillTree()
        tree.add_skill("test", "Test", "technical", initial_xp=5000)
        node = tree.get_skill("test")
        assert node.level <= 5


class TestSecurityNarrativeMemory:
    def test_event_content_sanitization(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()
        evt = nm.add_event("Test\x00Event\x01Here")
        assert "\x00" not in evt.content
        assert "\x01" not in evt.content

    def test_chapter_title_truncation(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()
        chapter = nm.create_chapter("A" * 300, "career")
        assert len(chapter.title) <= 200

    def test_arc_title_truncation(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()
        arc = nm.create_arc("B" * 300)
        assert len(arc.title) <= 200

    def test_importance_clamped(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()
        evt = nm.add_event("test", importance=5.0)
        assert evt.importance == 1.0
        evt2 = nm.add_event("test2", importance=-1.0)
        assert evt2.importance == 0.0

    def test_chapter_cap(self):
        from shadowd.cognition.narrative_memory import _MAX_CHAPTERS
        assert _MAX_CHAPTERS == 500

    def test_arc_cap(self):
        from shadowd.cognition.narrative_memory import _MAX_ARCS
        assert _MAX_ARCS == 100

    def test_thread_safety_events(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()
        errors = []

        def worker():
            try:
                for i in range(50):
                    nm.add_event(f"Event {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        stats = nm.get_stats()
        assert stats["total_events"] == 200


class TestEndToEndFlow:
    def test_skill_tree_full_progression(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain
        tree = AdaptiveSkillTree()

        tree.add_skill("python", "Python", "technical")
        tree.add_skill("testing", "Testing", "technical", dependencies=["python"])
        tree.add_skill("devops", "DevOps", "technical", dependencies=["testing"])

        python = tree.get_skill("python")
        assert python.unlocked is True
        assert python.level == 0

        for _ in range(30):
            tree.use_skill("python", success=True)

        python = tree.get_skill("python")
        assert python.level >= 1

        testing = tree.get_skill("testing")
        assert testing.unlocked is True

        recs = tree.get_recommendations()
        assert len(recs) > 0

        summary = tree.get_domain_summary()
        assert "technical" in summary
        assert summary["technical"]["skills"] == 3

    def test_narrative_full_story(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()

        ch1 = nm.create_chapter("Early Career", "career", "Starting out")
        e1 = nm.add_event("Got my first job!", importance=0.9)
        nm.add_event_to_chapter(ch1.chapter_id, e1)

        ch2 = nm.create_chapter("Growth Period", "growth", "Learning and growing")
        e2 = nm.add_event("Completed a major project successfully")
        nm.add_event_to_chapter(ch2.chapter_id, e2)

        arc = nm.create_arc("Career Arc", themes=["career", "growth"])
        nm.add_chapter_to_arc(arc.arc_id, ch1.chapter_id)
        nm.add_chapter_to_arc(arc.arc_id, ch2.chapter_id)

        nm.resolve_arc(arc.arc_id, "Built a successful career through persistence")

        summary = nm.generate_summary()
        assert summary.total_events >= 2
        assert summary.total_chapters == 2

    def test_skill_tree_and_narrative_combined(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        from shadowd.cognition.narrative_memory import NarrativeMemory

        skills = AdaptiveSkillTree()
        narrative = NarrativeMemory()

        skills.add_skill("coding", "Coding", "technical")
        narrative.add_event("Started learning to code")

        skills.use_skill("coding", success=True)
        narrative.add_event("Wrote my first program", importance=0.7)

        skill_stats = skills.get_stats()
        narrative_stats = narrative.get_stats()
        assert skill_stats["total_skills"] == 1
        assert narrative_stats["total_events"] == 2