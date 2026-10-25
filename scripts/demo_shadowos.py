#!/usr/bin/env python3
"""
ShadowOS v2 Interactive Demonstration

Walks through all 9 cognitive modules with live data,
showing how each module works and how they integrate together.

Usage:
  cd /home/user/digital-shadow
  source venv/bin/activate
  python scripts/demo_shadowos.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.dark_memory import AtomType
from shadowd.cognition.shadow_crew import DelegationMode
from shadowd.cognition.escalation import EscalationManager, AutonomyLevel
from shadowd.cognition.daily_briefing import DailyBriefingEngine
from shadowd.cognition.stealth_layer import DataCategory, AccessAction, CamouflageMode


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def subsection(title: str):
    print(f"\n--- {title} ---\n")


async def run_demo():
    print("""
  ███████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ██╗    ██╗
  ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗██║    ██║
  ███████╗███████║███████║██║  ██║██║   ██║██║ █╗ ██║
  ╚════██║██╔══██║██╔══██║██║  ██║██║   ██║██║███╗██║
  ███████║██║  ██║██║  ██║██████╔╝╚██████╔╝╚███╔███╔╝
  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══╝╚══╝
  
  ShadowOS v2.0 - Cognitive AI Platform
  Interactive Demonstration
    """)

    import tempfile
    os_inst = ShadowOS(data_dir=tempfile.mkdtemp(prefix="shadow_demo_"))

    # ==================== STARTUP ====================
    section("1. SYSTEM STARTUP")
    print("Initializing all 7 cognitive modules...")
    t0 = time.time()
    await os_inst.start()
    elapsed = (time.time() - t0) * 1000
    print(f"  All modules initialized in {elapsed:.1f}ms")
    health = os_inst.get_health()
    print(f"  System status: {health['status']}")
    print(f"  Modules: {', '.join(health['modules'].keys())}")

    # ==================== SHADOWGRAPH ====================
    section("2. COGNITIVE STATE MACHINE (ShadowGraph)")
    status = os_inst.graph.get_status()
    print(f"  Current cognitive mode: {status['current_mode']}")
    print(f"  Graph nodes: {', '.join(status['nodes'])}")
    print(f"  Edges: {status['edges_count']}")

    subsection("Processing a query through the cognitive graph")
    result = await os_inst.process_query("Will my project deadline be met?")
    print(f"  Query: 'Will my project deadline be met?'")
    print(f"  Resulting cognitive mode: {result['cognitive_mode']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Domain: {result['domain']}")

    print(f"\n  GraphViz (GraphViz dot format):")
    dot_lines = os_inst.get_graph_visualization().split('\n')
    for line in dot_lines[:5]:
        print(f"    {line}")
    print(f"    ... ({len(dot_lines)} lines total)")

    # ==================== DARKMEMORY ====================
    section("3. UNIFIED MEMORY SYSTEM (DarkMemory)")

    subsection("Storing observations")
    observations = [
        ("Working on Python API redesign", "career", 0.8),
        ("Lunch with Sarah at the new cafe", "social", 0.5),
        ("Investment portfolio up 3% this month", "finance", 0.7),
        ("Ran 5km in the morning", "health", 0.6),
        ("Read chapter on Bayesian inference", "learning", 0.7),
    ]
    for content, domain, importance in observations:
        atom = await os_inst.memory.store(content, atom_type=AtomType.OBSERVATION, domain=domain, importance=importance)
        print(f"  Stored: [{domain}] {content[:60]}... (importance: {importance})")

    subsection("Searching memory")
    results = await os_inst.search_memory("Python programming")
    print(f"  Search 'Python programming': {len(results)} results")
    for r in results[:3]:
        print(f"    - [{r.get('domain', '?')}] {r.get('content', '')[:80]}")

    subsection("Knowledge Graph")
    await os_inst.memory.store_entity("sarah", "person", {"name": "Sarah", "role": "colleague"})
    await os_inst.memory.store_entity("api_redesign", "project", {"name": "API Redesign", "status": "in_progress"})
    await os_inst.memory.store_relation("sarah", "api_redesign", "collaborates_on")
    kg = await os_inst.memory.query_kg(entity_id="sarah")
    print(f"  Entity: {kg['entity']}")
    print(f"  Relations: {len(kg['neighbors'].get('outgoing', []))} outgoing")

    subsection("Memory Blocks (Letta-style)")
    blocks = os_inst.get_memory_blocks()
    print(f"  {len(blocks)} memory blocks:")
    for b in blocks[:5]:
        print(f"    - {b['label']}: {b['length']}/{b['limit']} chars {'(protected)' if b['protected'] else ''}")

    # ==================== SHADOWCREW ====================
    section("4. MULTI-AGENT TEAM (ShadowCrew)")
    agents = os_inst.get_crew_agents()
    print(f"  {len(agents)} specialized agents:")
    for a in agents:
        print(f"    {a['name']} ({a['role']}) - {a['goal'][:60]}...")

    subsection("Creating and executing a mission")
    mission = await os_inst.create_mission("Predict user's project completion timeline")
    print(f"  Mission created: {mission['mission_id']}")
    print(f"  Tasks: {mission['tasks_count']}")
    result = await os_inst.execute_mission(mission['mission_id'])
    print(f"  Mission status: {result['status']}")

    # ==================== PERSONALITY ====================
    section("5. PERSONALITY ENGINE")
    subsection("Recording behavioral signals")
    behaviors = [
        ("question_asked", {"complexity": "complex", "domain": "career"}),
        ("decision_made", {"speed": "deliberate", "risk_level": "medium"}),
        ("task_completed", {"on_time": True, "quality": "high"}),
        ("social_interaction", {"mode": "one_on_one", "energy": "thoughtful"}),
        ("communication", {"length": "medium", "formality": "casual"}),
    ]
    for event_type, details in behaviors:
        os_inst.personality.record_behavior(event_type, details)
        print(f"  Recorded: {event_type}")

    subsection("Running personality distillation")
    distill = await os_inst.personality.distill()
    print(f"  Trait updates: {distill['trait_updates']}")
    print(f"  Style updates: {distill.get('style_updates', 0)}")
    print(f"  Profile version: {distill['profile_version']}")

    profile = os_inst.get_personality()
    bf = profile['big_five']
    print(f"\n  Big Five Personality:")
    dims = [('Openness', bf['openness']), ('Conscientiousness', bf['conscientiousness']),
            ('Extraversion', bf['extraversion']), ('Agreeableness', bf['agreeableness']),
            ('Neuroticism', bf['neuroticism'])]
    for name, val in dims:
        bar = '█' * int(val * 20) + '░' * (20 - int(val * 20))
        print(f"    {name:20s} [{bar}] {val:.2f}")
    print(f"\n  Decision style: {profile['decision_style']}")
    print(f"  Communication style: {profile['communication_style']}")

    subsection("Decision prediction")
    pred = os_inst.personality.predict_decision([
        {"name": "Safe Option", "reward": 0.6, "risk_level": 0.1, "novelty": 0.2},
        {"name": "Bold Option", "reward": 0.9, "risk_level": 0.7, "novelty": 0.8},
    ])
    print(f"  Predicted choice: Option {pred['predicted_choice']} ({pred['scores'][pred['predicted_choice']]['score']:.2f})")
    print(f"  Confidence: {pred.get('confidence', 0):.2f}")

    # ==================== STEALTH ====================
    section("6. STEALTH LAYER (Privacy & Camouflage)")
    subsection("Access control")
    granted, reason = os_inst.stealth.check_access("shadow_ai", AccessAction.READ, "obs1", DataCategory.BEHAVIORAL)
    print(f"  ShadowAI reads behavioral data: {'GRANTED' if granted else 'DENIED'} ({reason})")
    denied, reason = os_inst.stealth.check_access("external", AccessAction.READ, "obs2", DataCategory.CREDENTIALS)
    print(f"  External reads credentials: {'GRANTED' if denied else 'DENIED'} ({reason})")

    subsection("Data anonymization")
    data = {"name": "John Smith", "email": "john@example.com", "age": 30, "location": "San Francisco"}
    anon = os_inst.stealth.transform_for_sharing(data, CamouflageMode.ANONYMIZE)
    print(f"  Original:  {data}")
    print(f"  Anonymized: {anon}")

    subsection("Differential privacy")
    from shadowd.cognition.stealth_layer import CamouflageEngine
    engine = CamouflageEngine(epsilon=1.0)
    values = [100.0, 85.3, 92.7, 78.1, 95.6]
    noisy = [engine.apply_differential_privacy(v) for v in values]
    print(f"  Original values:  {values}")
    print(f"  With DP noise:   {[round(v, 1) for v in noisy]}")

    # ==================== ESCALATION ====================
    section("7. ESCALATION SPECTRUM (Graduated Autonomy)")
    esc = EscalationManager()
    actions = [
        ("observe", "Routine observation"),
        ("analyze", "Data analysis"),
        ("notify", "Send notification"),
        ("write_file", "Modify a file"),
        ("send_message", "Send a message"),
        ("execute_command", "Run a command"),
        ("delete", "Delete data"),
        ("financial_transaction", "Move money"),
    ]
    print(f"  Action autonomy levels:")
    for action, desc in actions:
        can, req = await esc.can_execute(action, {"description": desc})
        level = req.actual_level.name
        indicator = {"AUTO": "✅", "NOTIFY": "🔔", "ASK": "❓", "FORBID": "🚫"}.get(level, "?")
        print(f"    {indicator} {action:25s} → {level} ({'can execute' if can else 'blocked'})")

    # ==================== NEURAL MESH ====================
    section("8. NEURAL MESH (P2P Distributed Shadow)")
    from shadowd.cognition.neural_mesh import MeshNode
    phone = MeshNode(node_id="iphone_15", name="iPhone 15", device_type="phone", address="100.x.x.1", port=8765)
    laptop = MeshNode(node_id="macbook_pro", name="MacBook Pro", device_type="laptop", address="100.x.x.2", port=8765)
    os_inst.mesh.add_peer(phone)
    os_inst.mesh.add_peer(laptop)
    peers = os_inst.get_mesh_peers()
    print(f"  Local node: {os_inst.mesh.local_node.name} ({os_inst.mesh.local_node.device_type})")
    print(f"  Peers ({len(peers)}):")
    for p in peers:
        print(f"    - {p['name']} ({p['device_type']}) at {p['address']}:{p['port']}")

    mesh_stats = os_inst.mesh.get_stats()
    print(f"  Gossip seen messages: {mesh_stats['gossip_stats']['seen_messages']}")
    print(f"  Sync state keys: {mesh_stats['sync_stats']['keys']}")

    # ==================== SHADOW VOICE ====================
    section("9. SHADOW VOICE (Voice Agent)")
    voice_status = os_inst.voice.get_state()
    print(f"  Voice state: {voice_status['state']}")
    print(f"  Available voices: en_US-ryan-high (default), en_US-lessac-high, xtts-v2")

    from shadowd.cognition.shadow_voice import IntentParser
    parser = IntentParser()
    test_utterances = [
        "What will happen if I take this job offer?",
        "Schedule a meeting with Sarah",
        "Emergency! The server is down!",
        "Cancel that last request",
        "Hello shadow, good morning",
    ]
    print(f"\n  Intent parsing examples:")
    for utterance in test_utterances:
        intent, confidence = parser.parse(utterance)
        entities = parser.extract_entities(utterance)
        ent_str = f" | Entities: {entities}" if entities else ""
        print(f"    '{utterance[:50]}' → Intent: {intent.value} ({confidence:.1f}){ent_str}")

    # ==================== DAILY BRIEFING ====================
    section("10. DAILY BRIEFING")
    briefing_engine = DailyBriefingEngine(
        memory=os_inst.memory,
        personality=os_inst.personality,
        mesh=os_inst.mesh,
    )
    briefing = await briefing_engine.generate_briefing()
    print(f"  Date: {briefing.date}")
    print(f"  Overnight events: {briefing.overnight_count}")
    print(f"  Wellness score: {briefing.wellness_score:.0%}")
    print(f"  Items: {len(briefing.items)}")
    print(f"  Summary: {briefing.summary}")
    print(f"\n  Briefing Markdown:\n")
    md = briefing.to_markdown()
    for line in md.split('\n')[:25]:
        print(f"    {line}")

    # ==================== SHUTDOWN ====================
    section("SHUTDOWN")
    await os_inst.stop()
    print("  ShadowOS stopped gracefully")

    print(f"""
  ███████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ██╗    ██╗
  ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗██║    ██║
  ███████╗███████║███████║██║  ██║██║   ██║██║ █╗ ██║
  ╚════██║██╔══██║██╔══██║██║  ██║██║   ██║██║███╗██║
  ███████║██║  ██║██║  ██║██████╔╝╚██████╔╝╚███╔███╔╝
  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══╝╚══╝

  Demo complete! All 9 cognitive modules demonstrated.
  
  Key takeaways:
  - ShadowGraph: Cognitive state machine with 10 modes and conditional routing
  - DarkMemory: Unified memory with SQLite + vector + KG + blocks
  - ShadowCrew: 6 specialized agents in coordinated missions
  - NeuralMesh: P2P gossip + sync + consensus across devices
  - StealthLayer: Privacy guard + camouflage + differential privacy
  - PersonalityEngine: Big Five + decision prediction from behavior
  - ShadowVoice: Intent parsing + Piper TTS + Whisper STT
  - DailyBriefing: Morning intelligence report from all modules
  - EscalationManager: 4-level autonomy + 4 guardrails
    """)


if __name__ == "__main__":
    asyncio.run(run_demo())