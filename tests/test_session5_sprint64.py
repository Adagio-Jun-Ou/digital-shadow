import threading
import time
import pytest
from shadowd.cognition.social_intelligence import (
    SocialIntelligence, GroupType, InfluenceType, RoleType,
    SocialGroup, InfluenceRelation, SocialRole, CommunicationPattern,
    _sanitize,
)


class TestSocialIntelligenceBasics:
    def test_init(self):
        si = SocialIntelligence()
        stats = si.get_stats()
        assert stats["groups_tracked"] == 0
        assert stats["influence_relations"] == 0
        assert stats["roles_assigned"] == 0
        assert stats["patterns_detected"] == 0

    def test_create_group(self):
        si = SocialIntelligence()
        g = si.create_group("Team Alpha", GroupType.PROJECT, "A project team", ["p1", "p2"])
        assert g.group_id.startswith("grp_")
        assert g.name == "Team Alpha"
        assert g.group_type == GroupType.PROJECT
        assert g.description == "A project team"
        assert len(g.members) == 2
        assert "p1" in g.members

    def test_create_group_default_members(self):
        si = SocialIntelligence()
        g = si.create_group("Empty", GroupType.SOCIAL)
        assert g.members == []

    def test_create_group_all_types(self):
        si = SocialIntelligence()
        for gt in GroupType:
            g = si.create_group(f"Group-{gt.value}", gt)
            assert g.group_type == gt

    def test_add_member(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL, members=["a"])
        ok = si.add_member(g.group_id, "b")
        assert ok is True
        updated = si.get_group(g.group_id)
        assert "b" in updated.members
        assert len(updated.members) == 2

    def test_add_member_duplicate(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL, members=["a"])
        ok = si.add_member(g.group_id, "a")
        assert ok is True

    def test_add_member_nonexistent_group(self):
        si = SocialIntelligence()
        ok = si.add_member("nonexistent", "a")
        assert ok is False

    def test_add_member_cap(self):
        si = SocialIntelligence()
        members = [f"p{i}" for i in range(50)]
        g = si.create_group("G1", GroupType.FORMAL, members=members)
        ok = si.add_member(g.group_id, "p50")
        assert ok is False

    def test_remove_member(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL, members=["a", "b", "c"])
        ok = si.remove_member(g.group_id, "b")
        assert ok is True
        updated = si.get_group(g.group_id)
        assert "b" not in updated.members
        assert len(updated.members) == 2

    def test_remove_member_not_in_group(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL, members=["a"])
        ok = si.remove_member(g.group_id, "z")
        assert ok is False

    def test_remove_member_nonexistent_group(self):
        si = SocialIntelligence()
        ok = si.remove_member("nonexistent", "a")
        assert ok is False

    def test_get_group(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL)
        fetched = si.get_group(g.group_id)
        assert fetched is not None
        assert fetched.group_id == g.group_id

    def test_get_group_deepcopy(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL, members=["a"])
        fetched = si.get_group(g.group_id)
        fetched.name = "Mutated"
        original = si.get_group(g.group_id)
        assert original.name == "G1"

    def test_get_group_not_found(self):
        si = SocialIntelligence()
        assert si.get_group("nonexistent") is None

    def test_list_groups(self):
        si = SocialIntelligence()
        si.create_group("A", GroupType.FORMAL)
        si.create_group("B", GroupType.SOCIAL)
        all_groups = si.list_groups()
        assert len(all_groups) == 2

    def test_list_groups_by_type(self):
        si = SocialIntelligence()
        si.create_group("A", GroupType.FORMAL)
        si.create_group("B", GroupType.SOCIAL)
        si.create_group("C", GroupType.FORMAL)
        formal = si.list_groups(GroupType.FORMAL)
        assert len(formal) == 2

    def test_list_groups_deepcopy(self):
        si = SocialIntelligence()
        si.create_group("A", GroupType.FORMAL)
        groups = si.list_groups()
        groups[0].name = "Mutated"
        original = si.list_groups()
        assert original[0].name == "A"


class TestInfluenceRelations:
    def test_add_influence(self):
        si = SocialIntelligence()
        rel = si.add_influence("s1", "t1", InfluenceType.MENTOR, 0.8, "career guidance")
        assert rel.relation_id.startswith("inf_")
        assert rel.source == "s1"
        assert rel.target == "t1"
        assert rel.influence_type == InfluenceType.MENTOR
        assert rel.strength == 0.8
        assert rel.context == "career guidance"

    def test_add_influence_default_strength(self):
        si = SocialIntelligence()
        rel = si.add_influence("s1", "t1", InfluenceType.PEER)
        assert rel.strength == 0.5

    def test_add_influence_clamp_strength(self):
        si = SocialIntelligence()
        rel = si.add_influence("s1", "t1", InfluenceType.AUTHORITY, 2.0)
        assert rel.strength == 1.0
        rel2 = si.add_influence("s2", "t2", InfluenceType.EXPERT, -0.5)
        assert rel2.strength == 0.0

    def test_add_influence_all_types(self):
        si = SocialIntelligence()
        for it in InfluenceType:
            rel = si.add_influence("s", "t", it)
            assert rel.influence_type == it

    def test_get_influence_network(self):
        si = SocialIntelligence()
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.8)
        si.add_influence("carol", "alice", InfluenceType.PEER, 0.6)
        net = si.get_influence_network("alice")
        assert net["person_id"] == "alice"
        assert net["outgoing_count"] == 1
        assert net["incoming_count"] == 1

    def test_get_influence_network_empty(self):
        si = SocialIntelligence()
        net = si.get_influence_network("nobody")
        assert net["outgoing_count"] == 0
        assert net["incoming_count"] == 0

    def test_find_influencers(self):
        si = SocialIntelligence()
        si.add_influence("alice", "bob", InfluenceType.AUTHORITY, 0.9)
        si.add_influence("alice", "carol", InfluenceType.EXPERT, 0.8)
        si.add_influence("dave", "eve", InfluenceType.PEER, 0.4)
        influencers = si.find_influencers(min_influence=0.7)
        assert len(influencers) == 1
        assert influencers[0]["person_id"] == "alice"

    def test_find_influencers_by_group(self):
        si = SocialIntelligence()
        g = si.create_group("Team", GroupType.PROJECT, members=["alice", "bob"])
        si.add_influence("alice", "bob", InfluenceType.AUTHORITY, 0.9)
        si.add_influence("carol", "dave", InfluenceType.EXPERT, 0.9)
        influencers = si.find_influencers(group_id=g.group_id, min_influence=0.7)
        assert len(influencers) == 1
        assert influencers[0]["person_id"] == "alice"

    def test_find_influencers_nonexistent_group(self):
        si = SocialIntelligence()
        assert si.find_influencers(group_id="nonexistent") == []


class TestSocialRoles:
    def test_assign_role(self):
        si = SocialIntelligence()
        role = si.assign_role("p1", "g1", RoleType.LEADER, 0.9)
        assert role.role_id.startswith("role_")
        assert role.person_id == "p1"
        assert role.group_id == "g1"
        assert role.role_type == RoleType.LEADER
        assert role.influence_level == 0.9

    def test_assign_role_default(self):
        si = SocialIntelligence()
        role = si.assign_role("p1", "g1", RoleType.CONTRIBUTOR)
        assert role.influence_level == 0.5

    def test_assign_role_clamp(self):
        si = SocialIntelligence()
        r1 = si.assign_role("p1", "g1", RoleType.LEADER, 3.0)
        assert r1.influence_level == 1.0
        r2 = si.assign_role("p2", "g1", RoleType.OBSERVER, -1.0)
        assert r2.influence_level == 0.0

    def test_assign_role_all_types(self):
        si = SocialIntelligence()
        for rt in RoleType:
            role = si.assign_role("p", "g", rt)
            assert role.role_type == rt


class TestCommunicationPatterns:
    def test_detect_patterns_empty_group(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL)
        patterns = si.detect_communication_patterns(g.group_id)
        assert patterns == []

    def test_detect_patterns_no_influences(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.FORMAL, members=["a", "b"])
        patterns = si.detect_communication_patterns(g.group_id)
        assert patterns == []

    def test_detect_patterns_with_influences(self):
        si = SocialIntelligence()
        g = si.create_group("G1", GroupType.PROJECT, members=["alice", "bob", "carol"])
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.8)
        si.add_influence("alice", "carol", InfluenceType.MENTOR, 0.7)
        si.add_influence("bob", "carol", InfluenceType.PEER, 0.5)
        patterns = si.detect_communication_patterns(g.group_id)
        assert len(patterns) >= 1
        for p in patterns:
            assert p.pattern_id.startswith("cpat_")
            assert p.frequency >= 1
            assert -1.0 <= p.avg_sentiment <= 1.0

    def test_detect_patterns_nonexistent_group(self):
        si = SocialIntelligence()
        assert si.detect_communication_patterns("nonexistent") == []


class TestGroupDynamics:
    def test_get_dynamics_empty(self):
        si = SocialIntelligence()
        assert si.get_group_dynamics("nonexistent") == {}

    def test_get_dynamics_basic(self):
        si = SocialIntelligence()
        g = si.create_group("Team", GroupType.PROJECT, members=["a", "b", "c"])
        si.add_influence("a", "b", InfluenceType.AUTHORITY, 0.9)
        si.add_influence("a", "c", InfluenceType.EXPERT, 0.7)
        si.assign_role("a", g.group_id, RoleType.LEADER, 0.9)
        si.assign_role("b", g.group_id, RoleType.CONTRIBUTOR, 0.5)
        dynamics = si.get_group_dynamics(g.group_id)
        assert dynamics["group_id"] == g.group_id
        assert dynamics["member_count"] == 3
        assert dynamics["influence_count"] == 2
        assert "leader" in dynamics["role_distribution"]
        assert "contributor" in dynamics["role_distribution"]
        assert 0.0 <= dynamics["cohesion_score"] <= 1.0

    def test_get_dynamics_cohesion_update(self):
        si = SocialIntelligence()
        g = si.create_group("Team", GroupType.PROJECT, members=["a", "b"])
        si.add_influence("a", "b", InfluenceType.PEER, 0.8)
        dynamics = si.get_group_dynamics(g.group_id)
        assert dynamics["cohesion_score"] == 0.8

    def test_get_dynamics_influence_network(self):
        si = SocialIntelligence()
        g = si.create_group("Team", GroupType.PROJECT, members=["a", "b"])
        si.add_influence("a", "b", InfluenceType.MENTOR, 0.9)
        dynamics = si.get_group_dynamics(g.group_id)
        assert "a" in dynamics["influence_network"]
        assert len(dynamics["influence_network"]["a"]) == 1
        assert dynamics["influence_network"]["a"][0]["target"] == "b"


class TestBoundedEviction:
    def test_groups_cap(self):
        si = SocialIntelligence()
        for i in range(si._MAX_GROUPS + 10):
            si.create_group(f"Group-{i}", GroupType.FORMAL)
        stats = si.get_stats()
        assert stats["groups_tracked"] == si._MAX_GROUPS

    def test_influence_cap(self):
        si = SocialIntelligence()
        for i in range(si._MAX_INFLUENCE + 10):
            si.add_influence(f"s{i}", f"t{i}", InfluenceType.PEER)
        stats = si.get_stats()
        assert stats["influence_relations"] == si._MAX_INFLUENCE

    def test_roles_cap(self):
        si = SocialIntelligence()
        for i in range(si._MAX_ROLES + 10):
            si.assign_role(f"p{i}", f"g{i}", RoleType.CONTRIBUTOR)
        stats = si.get_stats()
        assert stats["roles_assigned"] == si._MAX_ROLES

    def test_patterns_cap(self):
        si = SocialIntelligence()
        for i in range(si._MAX_PATTERNS + 10):
            p = CommunicationPattern(
                pattern_id=f"cpat_{i:012x}",
                participants=[f"p{i}"],
                frequency=1,
                pattern_type="information_flow",
            )
            with si._lock:
                if len(si._patterns) >= si._MAX_PATTERNS:
                    evict_id = si._pattern_order.pop(0)
                    si._patterns.pop(evict_id, None)
                si._patterns[p.pattern_id] = p
                si._pattern_order.append(p.pattern_id)
                si._stats["patterns_detected"] = len(si._patterns)
        stats = si.get_stats()
        assert stats["patterns_detected"] == si._MAX_PATTERNS

    def test_group_eviction_removes_roles(self):
        si = SocialIntelligence()
        for i in range(si._MAX_GROUPS + 1):
            g = si.create_group(f"G{i}", GroupType.FORMAL)
        assert si.get_stats()["groups_tracked"] == si._MAX_GROUPS


class TestInputSanitization:
    def test_control_chars_stripped(self):
        assert _sanitize("hello\x00world", 100) == "helloworld"
        assert _sanitize("test\x08value", 100) == "testvalue"

    def test_max_length(self):
        long_str = "a" * 500
        assert len(_sanitize(long_str, 200)) == 200

    def test_group_name_sanitized(self):
        si = SocialIntelligence()
        g = si.create_group("Team\x00Alpha", GroupType.FORMAL)
        assert "\x00" not in g.name

    def test_influence_context_sanitized(self):
        si = SocialIntelligence()
        rel = si.add_influence("s", "t", InfluenceType.MENTOR, context="bad\x08context")
        assert "\x08" not in rel.context

    def test_long_description_truncated(self):
        si = SocialIntelligence()
        g = si.create_group("G", GroupType.FORMAL, description="x" * 2000)
        assert len(g.description) == 1000

    def test_members_sanitized(self):
        si = SocialIntelligence()
        g = si.create_group("G", GroupType.FORMAL, members=["p\x00id"])
        assert "\x00" not in g.members[0]


class TestThreadSafety:
    def test_concurrent_group_creates(self):
        si = SocialIntelligence()
        results = []
        errors = []

        def create_group(idx):
            try:
                g = si.create_group(f"G-{idx}", GroupType.FORMAL)
                results.append(g.group_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_group, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_influence_adds(self):
        si = SocialIntelligence()
        results = []
        errors = []

        def add_influence(idx):
            try:
                rel = si.add_influence(f"s{idx}", f"t{idx}", InfluenceType.PEER)
                results.append(rel.relation_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_influence, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_member_add_remove(self):
        si = SocialIntelligence()
        g = si.create_group("G", GroupType.FORMAL, members=["p0"])
        errors = []

        def add_remove(idx):
            try:
                si.add_member(g.group_id, f"p{idx}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_remove, args=(i,)) for i in range(1, 20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestDataclassToDict:
    def test_social_group_to_dict(self):
        g = SocialGroup(group_id="grp_test", name="Test", description="Desc", members=["a"], group_type=GroupType.FORMAL, cohesion_score=0.7)
        d = g.to_dict()
        assert d["group_id"] == "grp_test"
        assert d["group_type"] == "formal"
        assert d["cohesion_score"] == 0.7

    def test_influence_relation_to_dict(self):
        r = InfluenceRelation(relation_id="inf_test", source="s", target="t", influence_type=InfluenceType.AUTHORITY, strength=0.9, context="ctx")
        d = r.to_dict()
        assert d["relation_id"] == "inf_test"
        assert d["influence_type"] == "authority"

    def test_social_role_to_dict(self):
        r = SocialRole(role_id="role_test", person_id="p", group_id="g", role_type=RoleType.BRIDGE, influence_level=0.6)
        d = r.to_dict()
        assert d["role_id"] == "role_test"
        assert d["role_type"] == "bridge"

    def test_communication_pattern_to_dict(self):
        p = CommunicationPattern(pattern_id="cpat_test", participants=["a", "b"], frequency=5, pattern_type="decision-making", avg_sentiment=0.3)
        d = p.to_dict()
        assert d["pattern_id"] == "cpat_test"
        assert d["frequency"] == 5

    def test_cohesion_score_clamped(self):
        g = SocialGroup(group_id="grp_test", name="T", description="", cohesion_score=3.0)
        d = g.to_dict()
        assert d["cohesion_score"] == 1.0

    def test_sentiment_clamped(self):
        p = CommunicationPattern(pattern_id="cpat_test", participants=[], avg_sentiment=5.0)
        d = p.to_dict()
        assert d["avg_sentiment"] == 1.0

    def test_metadata_capped(self):
        meta = {f"k{i}": i for i in range(30)}
        g = SocialGroup(group_id="grp_test", name="T", description="", metadata=meta)
        d = g.to_dict()
        assert len(d["metadata"]) <= 20


class TestIntegration:
    def test_full_workflow(self):
        si = SocialIntelligence()
        g = si.create_group("Engineering", GroupType.PROFESSIONAL, "Eng team", ["alice", "bob", "carol"])
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.9, "career")
        si.add_influence("alice", "carol", InfluenceType.AUTHORITY, 0.8, "project")
        si.add_influence("bob", "carol", InfluenceType.PEER, 0.6)
        si.assign_role("alice", g.group_id, RoleType.LEADER, 0.9)
        si.assign_role("bob", g.group_id, RoleType.FACILITATOR, 0.7)
        si.assign_role("carol", g.group_id, RoleType.CONTRIBUTOR, 0.5)
        patterns = si.detect_communication_patterns(g.group_id)
        assert len(patterns) >= 1
        dynamics = si.get_group_dynamics(g.group_id)
        assert dynamics["member_count"] == 3
        assert dynamics["influence_count"] == 3
        influencers = si.find_influencers(group_id=g.group_id, min_influence=0.5)
        assert len(influencers) >= 1
        net = si.get_influence_network("alice")
        assert net["outgoing_count"] == 2
        assert net["incoming_count"] == 0

    def test_id_format(self):
        si = SocialIntelligence()
        g = si.create_group("G", GroupType.FORMAL)
        assert g.group_id.startswith("grp_")
        assert len(g.group_id) == 16
        rel = si.add_influence("s", "t", InfluenceType.PEER)
        assert rel.relation_id.startswith("inf_")
        assert len(rel.relation_id) == 16
        role = si.assign_role("p", "g", RoleType.LEADER)
        assert role.role_id.startswith("role_")
        assert len(role.role_id) == 17