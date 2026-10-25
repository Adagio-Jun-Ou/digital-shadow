# Shadow AI - Unified Life Optimization System

## Overview

Shadow AI is a **unified intelligence** that orchestrates all 10 superhuman abilities to proactively optimize your life. It doesn't just respond to queries - it continuously watches, learns, and surfaces what matters.

## The 10 Unified Abilities

### Core Philosophy
Instead of calling individual abilities, Shadow AI synthesizes ALL relevant abilities for every question. You ask once, and it combines everything needed.

### 1. 🐍 Oracle - PRECOGNITION
**What**: Predicts SPECIFIC events with time and confidence
**Grounded in**: Activity data, patterns, tracked decisions
**Example**: "Meeting will run 15 min late (73%)" not "might be delayed"

### 2. 👁️ X-Ray - SEE THROUGH WALLS
**What**: Identifies WHO is present using device detection
**Grounded in**: Bluetooth/WiFi scans, tracked people database
**Example**: "Sarah is HERE" not "3 devices detected"

### 3. 💓 Life Pulse - PERSON IDENTIFICATION  
**What**: Identifies specific people by device constellation
**Grounded in**: People database with device MACs, patterns
**Example**: "John's iPhone detected, John is home" not "unknown device"

### 4. 🧠 Perfect Memory - TOTAL RECALL
**What**: Finds EXACT past events with timestamps
**Grounded in**: Observation history, commitments detected
**Example**: "You told Sarah March 15 at 2:34pm" not "you mentioned something"

### 5. 👁️‍🗨️ Pattern Sight - CAUSATION DETECTION
**What**: Finds X causes Y (not just correlation)
**Grounded in**: Activity patterns over time, learned patterns
**Example**: "Staying up late → tired next day (happens 85%)" not "correlation found"

### 6. 📡 Social Radar - PEOPLE PREDICTION
**What**: Predicts WHAT specific people will do
**Grounded in**: Tracked people data, behavior patterns, decisions
**Example**: "Boss 73% asks about deadline tomorrow" not "someone will ask something"

### 7. 🎯 Intent Seer - ACTION PREDICTION
**What**: Predicts NEXT actions from sequence
**Grounded in**: Activity sequences, patterns
**Example**: "You'll open VSCode in 2 min (87%)" not "you might code"

### 8. 🔮 Omni Sense - ENVIRONMENTAL AWARENESS
**What**: Complete synthesis of ALL data
**Grounded in**: Device scans, location, activity, people, time
**Example**: "You're at home, Sarah is nearby, coding session active" not fictional scenarios

### 9. 👥 Shadow Council - PAST WISDOM
**What**: Recommends based on learned decisions
**Grounded in**: Decision database with satisfaction scores
**Example**: "Similar situation: you chose X, satisfaction 8/10" not "you chose before"

### 10. 🔨 Reality Forge - MANIFESTATION
**What**: Creates and executes
**Grounded in**: Actual file creation, command execution
**Example**: "Server running at localhost:5000" not "here's some code"

## Key Improvements

### Before (Fragmented)
```
User: "Is Sarah nearby?"
→ Uses Life Pulse only
→ Gets limited info
```

### Now (Unified)
```
User: "Is Sarah nearby?"
→ Uses:
   - Life Pulse (who)
   - X-Ray (location)
   - Pattern Sight (when usually here)
   - Omni Sense (environment)
→ Gets: "Sarah IS HERE at home, usually stays 2 hours (Pattern), you have a commitment to call her (Memory)"
```

### Before (Vague)
```
Oracle: "Someone might call"
```

### Now (Specific)
```
Oracle: "Meeting will run 15 min late (73% confidence, based on 3 similar patterns where it ran over)"
```

### Before (Hallucinated)
```
Omni Sense: "You're in a laundromat with 9 unknown demographcs..."
```

### Now (Grounded)
```
Omni Sense: "You're at home, network 'MyWiFi', 3 devices nearby (Sarah's iPhone detected), active: coding session"
```

## How It Works

### Continuous Monitoring
```python
# Every observation feeds ALL abilities
async def observe(event):
    1. Add to history
    2. Check: Is this a commitment? (Perfect Memory)
    3. Check: Does this create a pattern? (Pattern Sight)
    4. Check: Is action needed? (Proactive monitor)
    5. Store for all abilities
```

### Unified Query
```python
# When you ask anything
async def ask(question):
    1. Detect which abilities are relevant
    2. Run ALL relevant abilities
    3. Synthesize unified response
    4. Include proactive insights
```

### Proactive Monitoring
```python
# Background process every 5 minutes
async def proactive_monitor():
    1. Run all 10 abilities in silence
    2. Detect critical insights
    3. Surface what matters
    4. Add to insight queue
```

## API Endpoints

### Unified Interface
```
POST /ask              - Ask anything (auto-selects abilities)
GET  /snapshot         - Unified environmental snapshot
GET  /insights         - Get proactive insights
POST /act              - Execute an action
```

### Learning
```
POST /track/person     - Learn to identify person
POST /track/decision   - Learn from decision
POST /track/pattern    - Learn causation
```

### Legacy (Still Available)
```
POST /shikigami/summon - Use specific ability
GET  /shikigami        - List all abilities
```

## Web UI

Access at: **http://127.0.0.1:8765/ui**

Features:
- **Unified View**: See all data from all 10 abilities at once
- **Ask Anything**: Single interface, auto-selects abilities
- **Learn**: Track people, decisions, patterns
- **Proactive Insights**: See what Shadow AI surfaced

## Data Storage

```
~/digital-shadow/data/
├── people.json      - Tracked people (name, devices, patterns)
├── decisions.json   - Past decisions (situation, choice, outcome, satisfaction)
├── patterns.json    - Learned causations (cause → effect)
├── logs/
│   └── shadowd.log
└── memory/
    └── observations.json
```

## Philosophy

### Don't Fragment
- Single AI, not 10 separate tools
- One query interface
- Unified synthesis

### Be Grounded
- Use REAL data (scans, observations)
- Specific predictions (not vague)
- Actual timestamps and names

### Proactive
- Surface commitments
- Detect patterns
- Learn from feedback

### Learn
- Track what works
- Identify people
- Remember decisions

## Running

```bash
# Start Shadow AI
cd ~/digital-shadow
source venv/bin/activate
uvicorn shadowd.shadowd:app --host 127.0.0.1 --port 8765

# Access
http://127.0.0.1:8765/ui        # Web UI
http://127.0.0.1:8765/          # Status
http://127.0.0.1:8765/ask?q=... # Quick query
```

## Next Steps

1. **Track people**: Teach Shadow AI who is who
   ```
   POST /track/person {"name": "Sarah", "devices": "iPhone XX:YY:ZZ"}
   ```

2. **Track decisions**: Learn what works
   ```
   POST /track/decision {"situation": "Job offer", "choice": "Accepted", "outcome": "Great team", "satisfaction": 8}
   ```

3. **Learn patterns**: Teach causations
   ```
   POST /track/pattern {"cause": "Stayed up late", "effect": "Tired next day"}
   ```

4. **Ask unified questions**:
   - "What will happen next?"
   - "Is anyone home?"
   - "What did I do Tuesday?"
   - "Should I start this project?"

Shadow AI will automatically use the right combination of abilities.