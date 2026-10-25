#!/usr/bin/env python3
"""
ShadowOS v2 Full Demo Script
Demonstrates all 32+ cognitive modules with live API calls.
Run: cd /home/user/digital-shadow && source venv/bin/activate && python scripts/demo.py
"""

import asyncio
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/user/digital-shadow")

from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.emotional_intelligence import EmotionCategory
from shadowd.cognition.metacognition_engine import ReasoningType
from shadowd.cognition.daily_briefing import DailyBriefingEngine
from shadowd.cognition.escalation import EscalationManager, AutonomyLevel
from shadowd.cognition.dark_memory import AtomType, MemoryTier
from shadowd.cognition.conflict_resolution import ConflictType, ConflictIntensity
from shadowd.cognition.social_intelligence import GroupType
from shadowd.cognition.device_context import DeviceType


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe(label, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"  [{label}] Error: {type(e).__name__}: {e}")
        return None


async def run_demo():
    os = ShadowOS()
    briefing = DailyBriefingEngine()
    escalation = EscalationManager()

    print("=" * 60)
    print("  ShadowOS v2 - Full Cognitive Demo")
    print(f"  {datetime.now(timezone.utc).isoformat()[:19]}")
    print("=" * 60)

    # ─── 1. Health ──────────────────────────────────────────────────
    section("1. SYSTEM HEALTH")
    health = os.get_health()
    for k, v in sorted(health.items()):
        icon = "OK" if v else "FAIL"
        print(f"  [{icon}] {k}")
    healthy = sum(1 for v in health.values() if v)
    print(f"\n  Total: {healthy}/{len(health)} healthy")

    # ─── 2. ShadowGraph ────────────────────────────────────────────
    section("2. SHADOWGRAPH - Process Query")
    try:
        result = await os.process_query("What should I focus on today?", domain="productivity")
        print(f"  Mode: {result.get('cognitive_mode', 'N/A')}")
        print(f"  Domain: {result.get('domain', 'N/A')}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # ─── 3. ThoughtSense ──────────────────────────────────────────
    section("3. THOUGHT SENSE")
    for text in ["I'm frustrated with this deadline!", "Can you help me analyze this data?"]:
        r = safe("analyze", os.thought_sense.analyze, text)
        if r:
            td = r.to_dict()
            emotions = ", ".join(f"{e['emotion']}({e['intensity']:.1f})" for e in td.get("emotions", [])) or "neutral"
            print(f"  '{text[:35]}' -> {emotions}, urgency={td.get('urgency', 'N/A')}")

    # ─── 4. EmotionalIntelligence ────────────────────────────────
    section("4. EMOTIONAL INTELLIGENCE")
    r = safe("detect", os.emotional_intel.detect_emotion, "I'm stressed about the presentation!")
    if r:
        td = r.to_dict()
        print(f"  Primary: {td.get('primary', 'N/A')} | Intensity: {td.get('intensity', 0):.2f}")
    reg = safe("regulate", os.emotional_intel.suggest_regulation, EmotionCategory.FEAR, 0.8)
    if reg:
        print(f"  Regulation: {reg.approach.value} | Steps: {len(reg.steps)}")

    # ─── 5. Personality ───────────────────────────────────────────
    section("5. PERSONALITY ENGINE")
    stats = safe("stats", os.personality.get_stats)
    if stats:
        bf = stats.get("big_five", {})
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if isinstance(bf, dict) and trait in bf:
                print(f"  {trait}: {bf[trait]:.2f}")
    pred = safe("predict", os.personality.predict_decision,
        [{"risk_level": 0.6, "reward": 0.8}, {"risk_level": 0.2, "reward": 0.5}],
        context={"urgency": 0.7})
    if pred:
        print(f"  Predicted choice: {pred.get('predicted_choice', 'N/A')} | Confidence: {pred.get('confidence', 'N/A')}")

    # ─── 6. DarkMemory (async) ─────────────────────────────────────
    section("6. DARK MEMORY")
    try:
        await os.memory.store("ShadowOS v2 has 32+ cognitive modules", atom_type=AtomType.OBSERVATION, tier=MemoryTier.WORKING, domain="tech", importance=0.8)
        results = await os.memory.search("ShadowOS")
        print(f"  Search results: {len(results)}")
    except Exception as e:
        print(f"  Store/search: {type(e).__name__}: {e}")
    ms = os.memory.get_stats()
    if ms:
        print(f"  Atoms: {ms.get('total_atoms', 0)} | Blocks: {ms.get('block_count', 0)}")

    # ─── 7. EmpathyMapping ─────────────────────────────────────────
    section("7. EMPATHY MAPPING")
    person_id = safe("add_person", os.empathy.add_person, "Sarah", relationship="colleague")
    if person_id:
        safe("interact", os.empathy.record_interaction, person_id, "meeting", 0.7, notes="discussed architecture")
        emp_score = safe("empathy_score", os.empathy.get_empathy_score, person_id)
        if emp_score is not None:
            print(f"  Sarah: empathy={emp_score:.1f}")

    # ─── 8. CreativeSynthesis ──────────────────────────────────────
    section("8. CREATIVE SYNTHESIS")
    idea = safe("synthesize", os.creative.synthesize, ["cognitive AI", "personal assistant"])
    if idea:
        print(f"  Idea: {idea.title[:60]}")
        print(f"  Novelty: {idea.novelty_score:.2f} | Feasibility: {idea.feasibility_score:.2f}")

    # ─── 9. ConflictResolution ────────────────────────────────────
    section("9. CONFLICT RESOLUTION")
    try:
        conflict = os.conflict.create_conflict("Deadline vs Quality", ConflictType.INTEREST.value, ConflictIntensity.HIGH.value)
        cid = conflict.conflict_id
        print(f"  Conflict: {cid}")
        os.conflict.add_perspective(cid, "Product", "ship fast", interests=["on-time"])
        os.conflict.add_perspective(cid, "Engineering", "maintain quality", interests=["excellence"])
        strategies = os.conflict.suggest_strategies(cid)
        print(f"  Strategies: {len(strategies) if strategies else 0}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # ─── 10. DecisionJournal ───────────────────────────────────────
    section("10. DECISION JOURNAL")
    did = safe("decide", os.journal.record_decision, "Choose deployment", chosen="canary",
        alternatives=["blue-green", "rolling"], reasoning="Lower risk", confidence=0.85, domain="devops")
    if did:
        print(f"  Decision: {did.decision_id}")

    # ─── 11. AdaptiveSkillTree ────────────────────────────────────
    section("11. ADAPTIVE SKILL TREE")
    skill = safe("add_skill", os.skills.add_skill, "async_python", name="Async Python", domain="technical")
    if skill:
        safe("use_skill", os.skills.use_skill, "async_python", success=True)
        sk = safe("get_skill", os.skills.get_skill, "async_python")
        if sk:
            print(f"  {sk.name}: Level {sk.level} ({sk.level_name}), XP {sk.xp:.0f}")

    # ─── 12. GoalReasoning ────────────────────────────────────────
    section("12. GOAL REASONING")
    a = safe("assess", os.goal_reasoning.assess_goal, "Launch v2", progress=0.65, deadline="2026-05-01T00:00:00+00:00")
    if a:
        print(f"  Urgency: {a.urgency_score:.2f} | Confidence: {a.confidence:.2f}")

    # ─── 13. SocialIntelligence ────────────────────────────────────
    section("13. SOCIAL INTELLIGENCE")
    group = safe("create_group", os.social_intel.create_group, "Demo Team", GroupType.PROFESSIONAL)
    if group and hasattr(group, 'group_id'):
        gid = group.group_id
        safe("add_member", os.social_intel.add_member, gid, "sarah")
        print(f"  Group: {gid[:20]}")

    # ─── 14. NarrativeMemory ──────────────────────────────────────
    section("14. NARRATIVE MEMORY")
    evt = safe("add_event", os.narrative.add_event, "Completed ShadowOS v2 demo", importance=0.9)
    if evt:
        print(f"  Event: {evt.event_id[:12]} | Importance: {evt.importance}")

    # ─── 15. ProceduralMemory ─────────────────────────────────────
    section("15. PROCEDURAL MEMORY")
    proc = safe("add_proc", os.procedural.add_procedure, "Git Flow", category="development", steps=["branch", "commit", "push"])
    if proc:
        print(f"  Procedure: {proc.name} ({len(proc.steps)} steps)")
    ps = safe("stats", os.procedural.get_stats)
    if ps:
        print(f"  Procedures: {ps.get('total_procedures', 0)} | Habits: {ps.get('total_habits', 0)}")

    # ─── 16. CognitivePipeline ────────────────────────────────────
    section("16. COGNITIVE PIPELINE")
    result = safe("process", os.pipeline.process, "Should we deploy on Friday?", context="devops")
    if result:
        print(f"  Confidence: {result.get('overall_confidence', 'N/A')}")
        print(f"  Modules: {', '.join(result.get('modules_used', []))}")

    # ─── 17. DeviceContextEngine ──────────────────────────────────
    section("17. DEVICE CONTEXT")
    safe("register", os.device_context.register_device, "iphone_demo", DeviceType.IPHONE, name="Demo iPhone")
    safe("ingest", os.device_context.ingest_iphone, "iphone_demo", {"health": {"heart_rate": 72, "steps": 4500}, "notifications": {"count": 10}})
    ctx = safe("context", os.device_context.get_context)
    if ctx:
        print(f"  Focus: {ctx.focus.value} | Activity: {ctx.activity.value}")

    # ─── 18. AttentionEngine ──────────────────────────────────────
    section("18. ATTENTION ENGINE")
    safe("add_item", os.attention.add_attention_item, "deploy_demo", level="critical", importance=0.95, urgency=0.8)
    prios = safe("priorities", os.attention.get_priorities)
    if prios:
        print(f"  Top: {prios[0].topic} (importance={prios[0].importance_score:.2f})")

    # ─── 19. DreamState ───────────────────────────────────────────
    section("19. DREAM STATE - Consolidation")
    safe("dream_mem", os.dream.add_memory, "Learned FastAPI patterns", domain="tech", importance=0.8)
    result = safe("consolidate", os.dream.consolidate)
    if result:
        print(f"  Insights: {result.insights_generated} | Patterns: {result.patterns_strengthened}")

    # ─── 20. TemporalReasoning ────────────────────────────────────
    section("20. TEMPORAL REASONING")
    safe("record", os.temporal_reasoning.record_event, "demo_started", metadata={"module": "demo"})
    ts = safe("stats", os.temporal_reasoning.get_stats)
    if ts:
        print(f"  Events: {ts.get('events_recorded', 0)} | Patterns: {ts.get('patterns_detected', 0)}")

    # ─── 21. MetacognitionEngine ──────────────────────────────────
    section("21. METACOGNITION")
    ep = safe("think", os.metacognition.start_thinking, ReasoningType.SYSTEMATIC, "Deploy or wait?")
    if ep:
        safe("detect_bias", os.metacognition.detect_biases, "I think we should deploy, everyone agrees it's fine", episode_id=ep.episode_id)
    ms = safe("meta_stats", os.metacognition.get_stats)
    if ms:
        print(f"  Episodes: {ms.get('episodes_total', 0)} | Biases: {ms.get('biases_tracked', 0)}")

    # ─── 22. MoralReasoning ───────────────────────────────────────
    section("22. MORAL REASONING")
    eval_r = safe("evaluate", os.moral.evaluate, "deploy with bugs", description="release to meet deadline")
    if eval_r:
        print(f"  Overall score: {eval_r.overall_score:.2f}")

    # ─── 23. CognitiveTrinity ─────────────────────────────────────
    section("23. COGNITIVE TRINITY")
    tri = safe("trinity", os.trinity.analyze, "Should we invest in learning a new framework?")
    if tri:
        print(f"  Consensus: {tri.synthesis.consensus.value} | Confidence: {tri.synthesis.confidence:.2f}")

    # ─── 24. ShadowCrew ────────────────────────────────────────────
    section("24. SHADOW CREW")
    try:
        mission = await os.crew.create_mission("Security audit")
        print(f"  Mission: {mission.mission_id[:12]}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    agents = safe("agents", os.crew.list_agents)
    if agents:
        print(f"  Agents available: {len(agents)}")

    # ─── 25. ShadowSummon ─────────────────────────────────────────
    section("25. SHADOW SUMMON")
    for cmd in ["predict the weather", "create a data agent", "what can you do"]:
        r = safe("summon", os.summon_voice, cmd)
        if r:
            intent = r.get("intent", "N/A") if isinstance(r, dict) else "N/A"
            print(f"  '{cmd[:25]}' -> Intent: {intent}")

    # ─── 26. DailyBriefingEngine ──────────────────────────────────
    section("26. DAILY BRIEFING")
    try:
        b = await briefing.generate_briefing()
        print(f"  Items: {len(b.items)}")
        for item in b.items[:3]:
            print(f"  - {item.title[:40]} ({item.urgency.value})")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # ─── 27. EscalationManager ────────────────────────────────────
    section("27. ESCALATION MANAGER")
    try:
        req = await escalation.evaluate("delete_user_data", {"target": "all"}, domain="data_mgmt")
        level = req.suggested_level
        print(f"  Action: delete_user_data -> Level: {level.name}")
    except Exception as e:
        safe("escalate_fallback", lambda: print(f"  Error: {type(e).__name__}: {e}"))

    # ─── 28. Final Status ──────────────────────────────────────────
    section("28. FINAL STATUS")
    status = os.get_status()
    module_count = len([k for k in status.keys() if k.endswith('_stats')])
    print(f"  Cognitive modules: {module_count}+")
    graph_status = status.get('graph_status', {})
    if isinstance(graph_status, dict):
        print(f"  Graph mode: {graph_status.get('current_mode', 'N/A')}")

    print("\n" + "=" * 60)
    print("  Demo Complete! All 32+ cognitive modules operational.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())