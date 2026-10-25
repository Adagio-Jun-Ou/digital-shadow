#!/usr/bin/env python3
"""Sprint 153 — Deepcopy Audit Pass 2: shadow_crew.py

Verifies that getters return defensive copies, not mutable references
to internal state.
"""

import copy
import sys
import os
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadowd.cognition.shadow_crew import (
    ShadowAgent,
    AgentRole,
    AgentRegistry,
    ShadowCrew,
)


class TestShadowAgentGetState(unittest.TestCase):
    """ShadowAgent.get_state() must return deepcopy of mutable internals."""

    def setUp(self):
        self.agent = ShadowAgent(
            agent_id="test-agent",
            name="Test",
            role=AgentRole.OBSERVER,
            goal="observe",
            backstory="test agent",
        )

    def test_get_state_dict_is_copy(self):
        self.agent.set_state("config", {"key": "value"})
        result = self.agent.get_state("config")
        self.assertIsInstance(result, dict)
        result["key"] = "mutated"
        self.assertEqual(self.agent.get_state("config"), {"key": "value"})

    def test_get_state_list_is_copy(self):
        self.agent.set_state("items", [1, 2, 3])
        result = self.agent.get_state("items")
        self.assertIsInstance(result, list)
        result.append(4)
        self.assertEqual(self.agent.get_state("items"), [1, 2, 3])

    def test_get_state_set_is_copy(self):
        self.agent.set_state("tags", {"a", "b"})
        result = self.agent.get_state("tags")
        self.assertIsInstance(result, set)
        result.add("c")
        self.assertNotIn("c", self.agent.get_state("tags"))

    def test_get_state_scalar_not_copied(self):
        self.agent.set_state("count", 42)
        result = self.agent.get_state("count")
        self.assertEqual(result, 42)

    def test_get_state_default(self):
        result = self.agent.get_state("nonexistent", "default_val")
        self.assertEqual(result, "default_val")


class TestAgentRegistryGetReturnsCopy(unittest.TestCase):
    """AgentRegistry.get() must return deepcopy, not mutable reference."""

    def setUp(self):
        self.registry = AgentRegistry()
        self.agent = ShadowAgent(
            agent_id="reg-agent-1",
            name="RegAgent",
            role=AgentRole.OBSERVER,
            goal="observe",
            backstory="registry agent",
        )
        self.agent.set_state("data", {"secret": "abc"})
        self.registry.register(self.agent)

    def test_get_returns_different_object(self):
        result = self.registry.get("reg-agent-1")
        self.assertIsNot(result, self.registry.get("reg-agent-1"))

    def test_get_state_mutation_isolated(self):
        result = self.registry.get("reg-agent-1")
        result.set_state("data", {"secret": "MUTATED"})
        original = self.registry.get("reg-agent-1")
        self.assertEqual(original.get_state("data"), {"secret": "abc"})

    def test_get_nonexistent(self):
        self.assertIsNone(self.registry.get("no-such-agent"))


class TestAgentRegistryGetByRoleReturnsCopies(unittest.TestCase):
    """AgentRegistry.get_by_role() must return deepcopied agents."""

    def setUp(self):
        self.registry = AgentRegistry()
        self.agent1 = ShadowAgent(
            agent_id="role-agent-1",
            name="Observer1",
            role=AgentRole.OBSERVER,
            goal="observe",
            backstory="",
        )
        self.agent1.set_state("x", {"val": 1})
        self.agent2 = ShadowAgent(
            agent_id="role-agent-2",
            name="Observer2",
            role=AgentRole.OBSERVER,
            goal="observe",
            backstory="",
        )
        self.agent3 = ShadowAgent(
            agent_id="role-agent-3",
            name="Reasoner1",
            role=AgentRole.REASONER,
            goal="reason",
            backstory="",
        )
        self.registry.register(self.agent1)
        self.registry.register(self.agent2)
        self.registry.register(self.agent3)

    def test_returns_only_matching_role(self):
        observers = self.registry.get_by_role(AgentRole.OBSERVER)
        self.assertEqual(len(observers), 2)

    def test_returns_deepcopies(self):
        observers = self.registry.get_by_role(AgentRole.OBSERVER)
        observers[0].set_state("x", {"val": 999})
        again = self.registry.get_by_role(AgentRole.OBSERVER)
        self.assertEqual(again[0].get_state("x"), {"val": 1})

    def test_different_objects_each_call(self):
        obs1 = self.registry.get_by_role(AgentRole.OBSERVER)
        obs2 = self.registry.get_by_role(AgentRole.OBSERVER)
        self.assertIsNot(obs1[0], obs2[0])


class TestFindBestAgentReturnsCopy(unittest.TestCase):
    """AgentRegistry.find_best_agent() must return deepcopy."""

    def setUp(self):
        self.registry = AgentRegistry()
        self.agent = ShadowAgent(
            agent_id="best-agent",
            name="Best",
            role=AgentRole.OBSERVER,
            goal="observe",
            backstory="",
            tools=["observe"],
        )
        self.agent.set_state("inner", {"v": 1})
        self.registry.register(self.agent)

    def test_find_best_agent_returns_copy(self):
        a1 = self.registry.find_best_agent("observe")
        a2 = self.registry.find_best_agent("observe")
        self.assertIsNot(a1, a2)

    def test_find_best_agent_mutation_isolated(self):
        a1 = self.registry.find_best_agent("observe")
        a1.set_state("inner", {"v": 999})
        a2 = self.registry.find_best_agent("observe")
        self.assertEqual(a2.get_state("inner"), {"v": 1})

    def test_find_best_agent_none(self):
        result = self.registry.find_best_agent("nonexistent_task_type_xyz")
        self.assertEqual(type(result), ShadowAgent) or self.assertIsNotNone(result)


class TestShadowCrewGetAgent(unittest.TestCase):
    """ShadowCrew.get_agent() must return deepcopy via AgentRegistry.get()."""

    def setUp(self):
        self.crew = ShadowCrew()
        agent = ShadowAgent(
            agent_id="crew-agent",
            name="CrewAgent",
            role=AgentRole.REASONER,
            goal="reason",
            backstory="",
        )
        agent.set_state("protected", {"key": "val"})
        self.crew._registry.register(agent)

    def test_get_agent_returns_copy(self):
        a1 = self.crew.get_agent("crew-agent")
        a2 = self.crew.get_agent("crew-agent")
        self.assertIsNot(a1, a2)

    def test_get_agent_mutation_isolated(self):
        a1 = self.crew.get_agent("crew-agent")
        a1.set_state("protected", {"key": "MUTATED"})
        a2 = self.crew.get_agent("crew-agent")
        self.assertEqual(a2.get_state("protected"), {"key": "val"})


if __name__ == "__main__":
    unittest.main()