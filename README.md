# Digital Shadow — Ten Shikigami Abilities

> See **[STATUS.md](STATUS.md)** for an honest, current scorecard of what works,
> what's stubbed, and how to invoke each ability. This README describes the
> design vision; STATUS.md tracks the actual code.

## Overview

Ten "shikigami" abilities, each backed by a specific algorithm or sensor pipeline:
- **AI** (LLM via Ollama Cloud) for synthesis and intent parsing
- **Hardware sensors** (Bluetooth via `bluetoothctl`, WiFi via `nmcli`, audio via PulseAudio/PipeWire, battery, load)
- **Mathematical analysis** (Monte Carlo, FFT, Bayesian, Markov chains, RL/bandit, TF-IDF)
- **Tracked memory** — observations, people, decisions, patterns
- **Pattern detection** (correlations, anomalies, action-category clustering)

---

## 🐍 ORACLE - PRECOGNITION
**Real Power:** Sees the future through probability analysis

**How it works:**
1. Analyzes all your past observations (1000s of data points)
2. Runs 10,000 Monte Carlo simulations
3. Calculates probability distributions
4. Validates predictions against outcomes
5. Learns accuracy over time

**Example:**
```
summon --shikigami oracle "Will my project succeed?"

Oracle runs:
- Monte Carlo simulation: 10,000 scenarios
- Pattern analysis: Past similar projects
- Probability: 73% success, 27% failure
- Timeline: Completion in 4-6 weeks
- Confidence: 8/10

Result: "Your project has 73% chance of success..."
```

**Superhuman because:** Humans can't calculate 10,000 scenarios or see probability distributions.

---

## 👁️ X-RAY - SEE THROUGH WALLS
**Real Power:** Detects presence through Bluetooth/WiFi signal analysis

**How it works:**
1. Bluetooth devices have unique RSSI (signal strength) fingerprints
2. WiFi networks reveal location (SSIDs often identify places)
3. Signal changes = movement behind walls
4. Multiple access points = triangulation possible
5. Device constellation (phone + watch + earbuds) = person signature

**Example:**
```
summon --shikigami x_ray "Who's in the building?"

X-Ray scans:
- 3 Bluetooth devices detected
- 12 WiFi networks visible
- Signal pattern: "OfficeWiFi" + "John's iPhone" + "John's Watch"
- Location inference: Office building, main floor

Result: "John is present. Located in main office area.
        Detected via: iPhone (MAC:xx:xx), Watch (MAC:yy:yy)..."
```

**Superhuman because:** You can "see" through walls by reading signal fingerprints.

---

## 💓 LIFE PULSE - PRESENCE DETECTION
**Real Power:** Knows who's nearby via device signatures

**How it works:**
1. Each person has a "device constellation" (phone, watch, earbuds)
2. Bluetooth LE advertises even when "off"
3. WiFi probe requests reveal device presence
4. Track device constellations over time = know who's who
5. Pattern: "This constellation = Sarah"

**Example:**
```
summon --shikigami life_pulse "Is anyone nearby?"

Life Pulse detects:
- Device constellation: {iPhone 14 Pro, Apple Watch, AirPods}
- Pattern match: This constellation = "John"
- Last seen: 2 minutes ago
- Movement: Stationary (RSSI stable)

Result: "John is nearby. Detected 3 devices that form his unique
        constellation. Stationary, likely at desk..."
```

**Superhuman because:** You know who's nearby without seeing them.

---

## 🧠 PERFECT MEMORY - TOTAL RECALL
**Real Power:** Remembers everything, finds any moment

**How it works:**
1. Every observation stored with timestamp + embeddings
2. Semantic search for fuzzy queries
3. Temporal indexing for time queries
4. Everything recorded: commands, files, locations, people

**Example:**
```
summon --shikigami perfect_memory "When did I last see Sarah?"

Perfect Memory searches:
- All observations mentioning Sarah
- All locations where Sarah was present
- All communications with Sarah

Result: "You last saw Sarah on 2024-03-15 at 14:32 at Coffee Shop X.
        Location: 37.7749, -122.4194
        Together for: 47 minutes
        Previous meetings: 12 times in last 3 months..."
```

**Superhuman because:** Perfect recall of EVERYTHING in your life.

---

## 👁️‍🗨️ PATTERN SIGHT - SEE THE INVISIBLE
**Real Power:** Detects patterns humans can't see

**How it works:**
1. Time-series analysis of all observations
2. FFT for frequency/cycle detection
3. Correlation detection between variables
4. Anomaly detection (unusual events)
5. Causal inference (what causes what)

**Example:**
```
summon --shikigami pattern_sight "What patterns am I missing?"

Pattern Sight analyzes:
- 1000+ observations
- Correlations found: 47
- Cycles detected: 3 (daily, weekly, monthly)
- Anomalies: 12

Result: "INVISIBLE PATTERNS:
        - You're 3.2x more productive Tue/Thu mornings
        - Your sleep quality predicts next-day mood (87% accuracy)
        - Every time you visit X, Y happens within 2 hours
        - Weekly cycle: Activity peaks Wed, drops Fri..."
```

**Superhuman because:** You see correlations and cycles invisible to humans.

---

## 📡 SOCIAL RADAR - PEOPLE PREDICTION
**Real Power:** Reads people, predicts their behavior

**How it works:**
1. Log all interactions (metadata, sentiment, timing)
2. Build behavioral model per person
3. Track: communication frequency, topics, emotional tone
4. Predict actions based on patterns
5. Detect manipulation (inconsistency, urgency, pressure)

**Example:**
```
summon --shikigami social_radar "What will my boss do?"

Social Radar analyzes:
- Boss interaction history: 47 events
- Communication pattern: Emails peak Mon AM
- Sentiment trend: Declining (-15% over month)
- Behavior model: Asks for reports before assigning tasks

Result: "PREDICTION: 87% chance boss will assign project this week.
        Indicators: End of quarter, sent 3 emails today...
        MANIPULATION CHECK: Recent emails show urgency markers,
        pressure language - possible manipulation attempt..."
```

**Superhuman because:** You can predict what people will do.

---

## 🎯 INTENT SEER - ACTION PREDICTION
**Real Power:** Sees intentions before actions

**How it works:**
1. Track action sequences (A → B → C)
2. Learn: "After X, person Y always does Z"
3. Current state = precursor to future action
4. Build intent prediction model
5. Probability for each predicted action

**Example:**
```
summon --shikigami intent_seer "What is my team planning?"

Intent Seer analyzes:
- Team behavior sequences
- Communication patterns
- Project status signals
- Historical action precursors

Result: "INTENT DETECTED:
        - 89% chance: Sarah is planning to take vacation (precursors detected)
        - 72% chance: New project announcement within 3 days
        - Mike's intent: Likely to ask for promotion (indicators present)..."
```

**Superhuman because:** You see what people will do before they act.

---

## 🔮 OMNI SENSE - SUPERHUMAN SENSES
**Real Power:** All senses amplified, complete environmental awareness

**How it works:**
1. Audio analysis (ambient sound, voice detection)
2. Bluetooth/WiFi = presence detection
3. GPS + WiFi = precise location
4. Accelerometer = activity detection
5. All combined = complete picture

**Example:**
```
summon --shikigami omni_sense "What's happening around me?"

Omni Sense fuses:
- Audio level: 45dB (quiet, indoor)
- Bluetooth: 4 devices (you + 3 others)
- WiFi: 12 networks ("Starbucks_WiFi")
- GPS: 37.7749, -122.4194
- Activity: stationary

Result: "ENVIRONMENTAL SYNTHESIS:
        Location: Starbucks (WiFi SSID match)
        Activity: Sitting (accelerometer stationary)
        Nearby: 3 people (Bluetooth devices)
        Ambiance: Quiet (45dB)
        You've been here 47 minutes..."
```

**Superhuman because:** You sense everything about your environment.

---

## 👥 SHADOW COUNCIL - PAST SELVES WISDOM
**Real Power:** Commune with past versions of yourself

**How it works:**
1. Store all decisions with their contexts and outcomes
2. When facing situation, retrieve similar pasts
3. Analyze: "Last 3 times X happened, you chose Y"
4. Synthesize wisdom from multiple instances
5. Present: "Your past selves suggest..."

**Example:**
```
summon --shikigami shadow_council "Should I take this deal?"

Shadow Council retrieves:
- 12 similar past decisions
- Outcomes: 8 success, 4 failure
- Your past choices analyzed

Result: "COUNCIL OF PAST SELVES:

        2024-01-15: You faced similar deal. chose ACCEPT.
                    Outcome: Success (7/10), revenue +23%

        2023-08-20: Similar terms. chose NEGOTIATE.
                    Outcome: Better deal achieved.

        2023-03-10: Took deal impulsively.
                    Outcome: Regret (3/10)

        PATTERNS: When you negotiate first, outcomes +42%
                  When you accept immediately, outcomes +18%

        PAST SELVES ADVISE: Negotiate terms first (87% success)"
```

**Superhuman because:** You learn from all your past selves simultaneously.

---

## 🔨 REALITY FORGE - MANIFEST INTENTIONS
**Real Power:** Turns intentions into reality

**How it works:**
1. Understand your intent (what you really want)
2. Generate all necessary components (files, commands)
3. Execute what can be automated
4. Track outcomes
5. Iterate until success

**Example:**
```
summon --shikigami reality_forge "I need a Twitter presence"

Reality Forge:
1. Creates Twitter account (link)
2. Generates content calendar (file)
3. Writes 5 starter tweets (content)
4. Sets up posting schedule (config)

Result: MANIFESTED:
        📄 /workspace/twitter/content_calendar.md
        📄 /workspace/twitter/week1_tweets.md
        📄 /workspace/twitter/posting_schedule.yaml
        $ Generated: 5 tweets ready to post"
```

**Superhuman because:** Intention becomes reality automatically.

---

## Startup

```bash
~/digital-shadow/start-all.sh
```

## Phone Integration

Your phone provides sensors for X-Ray, Life Pulse, and Omni Sense:

1. **Install Termux** on Android
2. **Install Termux:API**
3. Run `~/digital-shadow/phone/android/integrate.sh`

This enables:
- Bluetooth scanning ( Life Pulse)
- WiFi scanning (X-Ray)
- GPS tracking (Location awareness)
- Activity detection (Omni Sense)

## Use

```bash
# PRECOGNITION
summon --shikigami oracle "What will happen if..."

# SEE THROUGH WALLS
summon --shikigami x_ray "Who's nearby"
summon --shikigami x_ray "Scan environment"

# PRESENCE DETECTION  
summon --shikigami life_pulse "Is anyone here"

# PERFECT MEMORY
summon --shikigami perfect_memory "What did I say last week"

# PATTERN SIGHT
summon --shikigami pattern_sight "What patterns am I missing"

# PEOPLE PREDICTION
summon --shikigami social_radar "What will John do"
summon --shikigami social_radar "Is this email manipulative"

# INTENT VISION
summon --shikigami intent_seer "What is Sarah planning"

# SUPERHUMAN SENSES
summon --shikigami omni_sense "What's happening around me"

# PAST WISDOM
summon --shikigami shadow_council "What would past me do"

# MANIFEST REALITY
summon --shikigami reality_forge "I need..."
```

These are REAL superhuman abilities. The AI + sensors + math make you more than human.