# Decision Learning Systems Research

**Research Date:** April 3, 2026  
**Purpose:** Analysis for Shadow AI Decision Learning Module  
**Time Allocation:** 1/6th of research effort (60 minutes)

## Executive Summary

This document analyzes decision learning systems across six categories to inform Shadow AI's decision tracking and learning capabilities. Key findings include reinforcement learning frameworks for LLM agents, multi-criteria decision analysis libraries, knowledge graph decision tracking, and personal knowledge management patterns.

---

## 1. Reinforcement Learning for Decision Making

### 1.1 AgentGym-RL (663 stars)

**Repository:** https://github.com/WooooDyy/AgentGym-RL  
**Paper:** "AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning"  
**Accepted:** ICLR 2026 (Oral)

#### Key Architectural Patterns

```python
# Modular System Design
class AgentGymFramework:
    """
    Three core modules:
    1. Environment - Diverse scenarios via HTTP server-client architecture
    2. Agent - Reasoning/decision-making with long-horizon planning
    3. Training - RL pipelines (PPO, GRPO, RLOO, REINFORCE++)
    """
    
    class EnvironmentModule:
        # WebArena - 4 domains: shopping, forums, dev, CMS
        # Deep Search - RAG-based retrieval and reasoning
        # TextCraft - Text-based crafting game
        # BabyAI - Grid world with text instructions
        # SciWorld - Scientific exploration simulator
        
    class AgentModule:
        # Encapsulates reasoning + multi-turn interaction
        # Supports long-horizon planning + self-reflection
        
    class TrainingModule:
        # RL algorithms: PPO, GRPO, RLOO, REINFORCE++
        # Additional: SFT, DPO, AgentEvol
```

#### ScalingInter-RL Method

```python
class ScalingInterRL:
    """
    Progressive horizon-scaling strategy for stable RL optimization.
    
    Key insight: Balance exploration/exploitation by starting
    with smaller horizons, then gradually extend.
    """
    
    def __init__(self):
        self.round_scheduler = None
        
    def progressive_horizon_training(self):
        # Phase 1: Small horizon - exploit policy, early proficiency
        # Phase 2: Extend horizon - explore longer decision paths
        # Phase 3: Final horizon - higher-order cognitive behaviors
        
        # Implementation uses:
        # - FixedRoundsScheduler: Fixed max interactions
        # - StepRoundsScheduler: Gradual horizon increase
```

#### Implementation Details for Shadow AI

```python
# Adaptable patterns:

# 1. Multi-turn decision tracking
class DecisionTracker:
    def track_decision_trajectory(self, decision_chain):
        """
        Track full decision context:
        - Environment observation
        - Available actions
        - Chosen action
        - Outcome reward
        - Temporal context (previous decisions)
        """
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'timestamps': [],
            'contexts': []
        }
        return trajectory

# 2. Scaling interaction during training
class ProgressiveSkillBuilder:
    def build_decision_skills(self, user_id):
        # Start with simple, single-turn decisions
        # Gradually introduce multi-turn planning
        # Eventually handle complex, long-horizon scenarios
        
# 3. Environment abstraction
class DecisionEnvironment:
    def get_observation(self):
        """Current state of decision context"""
        
    def get_available_actions(self):
        """Possible decision paths"""
        
    def step(self, action):
        """Execute decision, return outcome"""
        
    def reset(self):
        """Start new decision episode"""
```

#### Performance Insights

- ScalingInter-7B matches/surpasses GPT-4o on WebArena
- Progressive horizon training prevents collapse seen in longer-turn settings
- Modular design allows swapping environments/training methods

---

### 1.2 RL4VLM - Vision-Language Models for Decision Making

**Repository:** https://github.com/RL4VLM/RL4VLM  
**Paper:** "Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning"  
**Conference:** NeurIPS 2024

#### Key Contributions

```python
class RL4VLMArchitecture:
    """
    Fine-tunes VLMs (LLaVA) as decision-making agents.
    
    Key innovations:
    1. Vision-language understanding for environmental observations
    2. PPO fine-tuning for decision optimization
    3. GymCards environment for card games (testbed)
    4. ALFWorld environment for household tasks
    """
    
    def __init__(self):
        self.backbone = "llava-1.6-mistral-7b"
        self.training_steps = [
            "SFT_checkpoint",  # Supervised fine-tuning first
            "RL_finetuning"   # Then RL optimization
        ]
```

#### Training Pipeline

```python
# From their codebase structure:

# 1. SFT Phase - Learn to follow instructions
sft_data = "instruction-following-data"
sft_checkpoint = finetune_llava(
    base_model="llava-v1.6-mistral-7b",
    data=sft_data
)

# 2. RL Phase - Optimize decision making
rl_agent = VLM_PPO(
    model=sft_checkpoint,
    environment=gym_cards_env,
    reward_function=card_game_reward
)
```

#### Shadow AI Adaptation

```python
# For decision learning with context understanding:

class VisionContextDecisionTracker:
    def process_environment_observation(self, observation):
        """
        Use VLM to understand:
        - Visual context (diagrams, screenshots, documents)
        - Text context (emails, messages, documents)
        - Combined multimodal context
        """
        vlm_understanding = vlm_model.understand(observation)
        return vlm_understanding
    
    def make_decision_with_vision(self, context, options):
        # Combine visual understanding with decision logic
        observation_embedding = self.vision_encoder(context)
        decision_logits = self.decision_head(observation_embedding)
        return decision_logits
```

---

### 1.3 Applied Reinforcement Learning (331 stars)

**Repository:** https://github.com/mimoralea/applied-reinforcement-learning

Comprehensive tutorial covering decision-making foundations:

#### Curriculum Structure

```markdown
1. Introduction to Decision-Making
2. Sequential Decisions
   - MDPs, policies, value functions
3. Deterministic and Stochastic Actions
   - Handling uncertainty
4. Known and Unknown Environments
   - Model-based vs model-free RL
5. Discrete and Continuous States
   - Function approximation
6. Discrete and Continuous Actions
   - Policy search methods
7. Observable and Partially-Observable States
   - POMDPs, state estimation
8. Single and Multiple Agents
   - Game theory, multi-agent RL
9. Cooperative and Adversarial Agents
   - Team vs competitive settings
10. Decision-Making and Humans
    - Cognitive science connections
```

#### Key Patterns for Shadow AI

```python
# Sequential decision framework
class SequentialDecisionMaker:
    def __init__(self):
        self.value_function = {}
        self.policy = {}
        
    def evaluate_policy(self, state):
        """Calculate expected return from state"""
        return self.value_function.get(state, 0)
    
    def improve_policy(self, state):
        """Select best action given value estimates"""
        return max(self.actions, 
                   key=lambda a: self.q_value(state, a))
    
    def policy_iteration(self):
        """Alternate evaluation and improvement"""
        while not self.policy_stable():
            self.evaluate()
            self.improve()

# For Shadow AI: Track user decisions across time
class UserDecisionTracker:
    def __init__(self, user_id):
        self.decision_history = []
        self.outcome_history = []
        self.context_history = []
        
    def record_decision(self, decision, context, outcome):
        self.decision_history.append(decision)
        self.outcome_history.append(outcome)
        self.context_history.append(context)
        
    def learn_policy(self):
        """Learn decision patterns from history"""
        # Use value iteration or policy iteration
        # on user's historical decisions
```

---

## 2. Multi-Criteria Decision Analysis (MCDA)

### 2.1 pyDecision (343 stars)

**Repository:** https://github.com/Valdecy/pyDecision  
**Paper:** "Enhancing Decision Analysis with a Large Language Model" (Journal of Modelling in Management, 2026)

Comprehensive MCDA library with 50+ methods:

#### Available Methods

```python
# Weight derivation methods
ahp_method          # Analytic Hierarchy Process
anp_method          # Analytic Network Process  
fuzzy_ahp          # Fuzzy AHP
bwm_method          # Best-Worst Method
fucam_method        # Full Consistency Method

# Outranking methods
electre_i           # ELimination Et Choix Traduisant la REalite
electre_ii, iii, iv
electre_tri_b       # ELECTRE for sorting
promethee_i, ii, iii, iv, v, vi, gaia

# Scoring methods
topsis_method       # Technique for Order Preference
fuzzy_topsis
vikor_method        # VIKOR for compromise solution
fuzzy_vikor
saw_method          # Simple Additive Weighting
waspas_method       # Weighted Aggregates Sum Product

# Distance/dominance methods
codas_method        # Combinative Distance-based Assessment
edas_method         # Distance from Average Solution
mabac_method        # Border Approximation area Comparison
mairca_method       # Ideal-Real Comparative Analysis
marcos_method       # Ranking Alternatives via COmpromise

# Utility-based
maut_method         # Multi-Attribute Utility Theory
utadis_i, ii, iii   # UTilites Additives DIScriminantes

# And 30+ more methods...
```

#### Implementation Pattern

```python
from pyDecision.algorithm import topsis_method, ahp_method

# Example: Choosing between tools with multiple criteria
dataset = np.array([
    # Cost  Performance  Ease  Support
    [100,  0.85,        0.7,  0.8],   # Tool A
    [150,  0.95,        0.6,  0.9],   # Tool B
    [80,   0.75,        0.9,  0.7],   # Tool C
    [120,  0.88,        0.75, 0.85]   # Tool D
])

# Weight criteria using AHP
weights, consistency = ahp_method(criterion_comparison_matrix, 
                                   wd='geometric')

# Rank alternatives using TOPSIS
ranked = topsis_method(dataset, weights, 
                       criteria_types=['min', 'max', 'max', 'max'])
```

#### Shadow AI Integration

```python
class MultiCriteriaDecisionEngine:
    """
    Help users make decisions with multiple competing objectives.
    
    Example user query: "Should I take job A or job B?"
    Criteria: salary, location, work-life balance, growth potential
    """
    
    def __init__(self):
        self.mcda_methods = {
            'topsis': topsis_method,
            'electre': electre_ii_method,
            'promethee': promethee_ii_method,
            'vikor': vikor_method,
            'maut': maut_method
        }
    
    def analyze_decision(self, alternatives, criteria, weights=None):
        """
        alternatives: List of options with criteria values
        criteria: Names and types (minimize/maximize)
        weights: User preferences (derived if not provided)
        """
        
        # If weights not provided, use AHP to derive
        if weights is None:
            weights, rc = self.derive_weights_from_user_preferences()
            
        # Apply multiple MCDA methods
        results = {}
        for method_name, method_func in self.mcda_methods.items():
            ranking = method_func(alternatives, weights, criteria_types)
            results[method_name] = ranking
            
        # Aggregate rankings for robust decision
        final_ranking = self.aggregate_rankings(results)
        
        return {
            'ranking': final_ranking,
            'method_results': results,
            'sensitivity': self.sensitivity_analysis(alternatives, weights)
        }
    
    def derive_weights_from_user_preferences(self):
        """
        Interactive AHP to get user preferences:
        - "How important is salary vs work-life balance?"
        - Build pairwise comparison matrix
        - Derive weights with consistency check
        """
        pass
    
    def track_decision_outcome(self, decision_id, actual_outcome):
        """
        Compare predicted vs actual outcomes to improve
        weight estimation and alternative evaluation.
        """
        pass

# Usage in Shadow AI
class DecisionSupportModule:
    def help_user_decide(self, user_context):
        """
        Example flow:
        User: "I need to decide between these 3 apartments"
        
        AI: "Let me help you analyze this systematically.
             What criteria matter most to you?"
        
        User: "Price, location, size, amenities"
        
        AI: "I'll help you prioritize these. How important is
             price compared to location on a scale of 1-9?"
        
        [After gathering preferences]
        
        AI: "Based on your priorities, Apartment B ranks first.
             However, sensitivity analysis shows Apartment A
             would be better if you valued location more highly.
             Would you like to explore different scenarios?"
        """
        mcda = MultiCriteriaDecisionEngine()
        return mcda.analyze_decision(user_context['alternatives'],
                                     user_context['criteria'])
```

---

## 3. Decision Journaling & Knowledge Graphs

### 3.1 graphthulhu (125 stars)

**Repository:** https://github.com/skridlevsky/graphthulhu  
**Purpose:** MCP server for AI access to Logseq/Obsidian knowledge graphs

#### Decision Tracking Implementation

From `/tools/decision.go`:

```go
// Core decision tracking structure
type decisionBlock struct {
    UUID       string `json:"uuid"`
    Content    string `json:"content"`
    Page       string `json:"page"`
    Marker     string `json:"marker"`      // DECIDE, DONE, TODO, DOING, WAIT, LATER
    Deadline   string `json:"deadline,omitempty"`
    Resolved   string `json:"resolved,omitempty"`
    Outcome    string `json:"outcome,omitempty"`
    DaysLeft   *int   `json:"daysLeft,omitempty"`
    Overdue    bool   `json:"overdue"`
    Deferred   int    `json:"deferred,omitempty"`
    DeferredOn string `json:"deferredOn,omitempty"`
}

// Decision lifecycle tracking
func (d *Decision) DecisionCreate(ctx context.Context, input DecisionCreateInput) {
    // Create DECIDE block with:
    // - Question/prompt
    // - #decision tag
    // - deadline property
    // - options list
    // - context notes
    
    content := fmt.Sprintf("DECIDE %s #decision", input.Question)
    content += fmt.Sprintf("\ndeadline:: %s", input.Deadline)
    if len(input.Options) > 0 {
        content += fmt.Sprintf("\noptions:: %s", strings.Join(input.Options, ", "))
    }
    if input.Context != "" {
        content += fmt.Sprintf("\ncontext:: %s", input.Context)
    }
}

func (d *Decision) DecisionResolve(ctx context.Context, input DecisionResolveInput) {
    // Mark as DONE with:
    // - Resolution date
    // - Chosen option
    // - Outcome/reasoning
    
    content = strings.Replace(content, "DECIDE ", "DONE ", 1)
    content = upsertProperty(content, "resolved", time.Now().Format("2006-01-02"))
    content = upsertProperty(content, "outcome", input.Outcome)
}

func (d *Decision) DecisionDefer(ctx context.Context, input DecisionDeferInput) {
    // Push deadline with:
    // - New deadline
    // - Deferral count (warn after 3+)
    // - Reason for deferral
    
    deferCount := getDeferCount(content) + 1
    content = upsertProperty(content, "deadline", input.NewDeadline)
    content = upsertProperty(content, "deferred", fmt.Sprintf("%d", deferCount))
    
    if deferCount >= 3 {
        // Warning: Consider resolving or abandoning
    }
}
```

#### Decision Health Analysis

```go
// Analysis health audit for decision pages
func (d *Decision) AnalysisHealth(ctx context.Context) {
    // Find all analysis/strategy/assessment pages
    // Check:
    // 1. Outgoing links (should have 3+)
    // 2. Has decision marker
    // 3. Mark as unhealthy if isolated
    
    for _, page := range analysisPages {
        outgoing := countOutgoingLinks(page)
        hasDecision := checkForDecisionMarker(page)
        
        isHealthy := outgoing >= 3 || hasDecision
        
        if !isHealthy {
            issue := "isolated analysis — fewer than 3 outgoing links and no decision"
        }
    }
}
```

#### Shadow AI Adaptation

```python
class DecisionJournal:
    """
    Track decisions over time with:
    - Question/prompt
    - Options considered
    - Deadline
    - Resolution
    - Outcome
    - Context
    - Deferal history
    """
    
    def __init__(self):
        self.decisions = {}
        self.deferred_threshold = 3  # Warn if deferred 3+ times
        
    def create_decision(self, question, options, deadline, context=""):
        decision_id = generate_uuid()
        self.decisions[decision_id] = {
            'id': decision_id,
            'status': 'DECIDE',
            'question': question,
            'options': options,
            'deadline': deadline,
            'context': context,
            'created': datetime.now(),
            'defer_count': 0,
            'created_at': datetime.now()
        }
        return decision_id
    
    def resolve_decision(self, decision_id, chosen_option, outcome, reasoning):
        decision = self.decisions[decision_id]
        decision['status'] = 'DONE'
        decision['resolved_at'] = datetime.now()
        decision['chosen_option'] = chosen_option
        decision['outcome'] = outcome
        decision['reasoning'] = reasoning
        
        # Track for learning
        self._record_outcome(decision)
        
    def defer_decision(self, decision_id, reason, new_deadline):
        decision = self.decisions[decision_id]
        decision['defer_count'] += 1
        decision['last_deferred'] = datetime.now()
        decision['defer_reason'] = reason
        decision['deadline'] = new_deadline
        
        if decision['defer_count'] >= self.deferred_threshold:
            self._warn_user(decision)
            
    def get_overdue_decisions(self):
        now = datetime.now()
        overdue = []
        for decision in self.decisions.values():
            if decision['status'] == 'DECIDE':
                if decision['deadline'] < now:
                    decision['overdue'] = True
                    overdue.append(decision)
        return overdue
    
    def analyze_decision_patterns(self):
        """
        Learn from past decisions:
        - Which types of decisions get deferred?
        - Average time to resolution
        - Success rate of outcomes
        - Pattern analysis
        """
        patterns = {
            'deferral_rate': 0,
            'avg_resolution_time': 0,
            'outcome_success_rate': 0,
            'common_decision_types': {}
        }
        
        # Analyze historical decisions
        for decision in self.decisions.values():
            if decision['status'] == 'DONE':
                # Track resolution time
                # Track outcome success
                # Track decision categories
                pass
                
        return patterns
```

---

### 3.2 commit-story (18 stars)

**Repository:** https://github.com/wiggitywhitney/commit-story

#### Journal Entry Generation

```javascript
// From commit-story architecture

class CommitJournalGenerator {
    async generateJournalEntry(commitSha, commitMessage, diffs, claudeContext) {
        /*
        Combines:
        1. Git commit information
        2. Code diffs
        3. AI assistant conversation
        4. Technical decisions
        
        Into rich narrative journal entry.
        */
        
        const entry = {
            timestamp: new Date(),
            commitSha: commitSha,
            summary: await this.generateSummary(diffs, claudeContext),
            dialogue: this.extractKeyDialogue(claudeContext),
            decisions: this.extractDecisions(claudeContext),
            codeChanges: this.summarizeCodeChanges(diffs)
        };
        
        return this.formatMarkdown(entry);
    }
    
    extractDecisions(claudeContext) {
        // Find decision points in conversation
        // Extract why decisions were made
        // Note alternatives considered
        // Capture trade-offs
    }
    
    extractKeyDialogue(claudeContext) {
        // Select conversational highlights
        // Include user prompts + AI reasoning
        // Focus on learning moments
    }
}

// Sample journal entry
/*
## 9:46:42 AM CDT - Commit: 1502704e

### Summary - 1502704e
The developer created a new PRD for restructuring the prompts 
used in the system. They analyzed successful prompt patterns from 
existing commands and proposed applying the same step-based 
principles to avoid format-first antipatterns.

### Development Dialogue - 1502704e
**Human:** "Look at prompts that consistently work well for me 
like /prd-create and /prd-next. I'm considering whether the 
Technical Decisions section prompt needs to be updated to follow 
steps to prevent AI from skipping ahead."

**Human:** "I like this except I want two additional things: 
an analysis step to make sure no critical bit gets lost, and 
before/after tests on multiple commits with human approval."

### Technical Decisions - 1502704e
- **DECISION: Step-Based Prompt Architecture PRD Creation**
  - Restructuring prompts to follow successful patterns
  - Emphasis on preventing AI from skipping critical analysis steps
  - Inclusion of mandatory human approval testing

### Commit Details - 1502704e
**Files Changed**: prds/1-automated-git-journal-system.md, 
                    prds/4-step-based-prompt-architecture.md  
**Lines Changed**: ~213 lines  
**Message**: "feat(prd-4): create step-based prompt architecture PRD"
*/
```

#### Shadow AI Integration

```python
class AutomatedDecisionJournal:
    """
    Automatically capture Shadow AI's decision-making context.
    
    Similar to commit-story but for AI decisions:
    - What information was available
    - What alternatives were considered
    - Why a particular choice was made
    - What the outcome was
    """
    
    def record_shadow_decision(self, decision_id):
        decision = self.decisions[decision_id]
        
        entry = {
            'timestamp': datetime.now(),
            'decision_id': decision_id,
            'summary': self._generate_summary(decision),
            'available_information': decision['context'],
            'alternatives_considered': decision['options'],
            'chosen_alternative': decision['chosen_option'],
            'reasoning': decision['reasoning'],
            'confidence': decision['confidence'],
            'expected_outcomes': decision['expected_outcomes'],
            'actual_outcomes': None  # Filled in later
        }
        
        self._write_journal_entry(entry)
        
    def update_with_outcome(self, decision_id, actual_outcome, satisfaction):
        """
        Update decision journal with actual results:
        - Did the expected outcome occur?
        - User satisfaction level
        - Lessons learned
        - What would we do differently?
        """
        decision = self.decisions[decision_id]
        correlation = self._compare_expected_vs_actual(
            decision['expected_outcomes'],
            actual_outcome
        )
        
        # Learning for future similar decisions
        self._update_decision_model(correlation, satisfaction)
```

---

## 4. Personal Knowledge Management for Decisions

### 4.1 WhatsAppRAG / Personal RAG System (1 star)

**Repository:** https://github.com/MQasimJalil/WhatsAppRAG

Comprehensive second brain architecture:

#### Architecture Overview

```python
class PersonalRAGSystem:
    """
    Dual storage architecture:
    - SQLite: Metadata + FTS5 full-text search
    - ChromaDB: Vector embeddings for semantic search
    """
    
    def __init__(self):
        self.metadata_db = SQLiteWithFTS5("data/database/memories.db")
        self.vector_db = ChromaDBWrapper("data/vector_db/chroma")
        self.ai_client = GeminiClient()
        self.ingestion_pipeline = IngestionPipeline()
        self.file_watcher = FileWatcherService()
        
    # Storage schema
    class Memory:
        id: str
        content: str
        source_type: SourceType  # NOTE, COMMIT, SESSION, etc.
        metadata: dict
        tags: list[str]
        embedding: list[float]
        timestamp: datetime
        chunk_id: str
        parent_id: Optional[str]
        
    class SourceType(Enum):
        NOTE = "note"
        COMMIT = "commit"
        SESSION = "session"
        CONVERSATION = "conversation"
        DECISION = "decision"  # KEY: Decision tracking
```

#### Decision-Specific Features

```python
class DecisionMemory(IngestionSource):
    """
    Capture and track decisions with:
    - Question/prompt
    - Options considered
    - Chosen option
    - Reasoning
    - Outcome (when known)
    - References to related decisions
    """
    
    def extract_content(self, raw_data):
        return {
            'content': raw_data['decision_text'],
            'metadata': {
                'decision_type': raw_data['type'],
                'decision_maker': raw_data['user'],
                'context': raw_data['context'],
                'alternatives': raw_data['alternatives'],
            },
            'tags': self._extract_decision_tags(raw_data),
            'timestamp': raw_data['timestamp']
        }
        
    def should_ingest(self, raw_data):
        # Only ingest if decision is finalized
        return raw_data.get('status') == 'resolved'
        
    def _extract_decision_tags(self, raw_data):
        tags = []
        if 'domain' in raw_data:
            tags.append(f"domain:{raw_data['domain']}")
        if 'priority' in raw_data:
            tags.append(f"priority:{raw_data['priority']}")
        if 'outcome' in raw_data:
            tags.append(f"outcome:{raw_data['outcome']}")
        return tags
```

#### Ingestion Pipeline with Smart Chunking

```python
class IngestionPipeline:
    def __init__(self):
        self.chunker = ContentAwareChunker()
        self.extractor = ContentExtractor()
        
    def ingest_decision(self, decision_content):
        # Extract metadata
        metadata = self.extractor.extract_frontmatter(decision_content)
        
        # Content-aware chunking
        chunks = self.chunker.chunk(
            decision_content,
            chunk_type='semantic',  # For decisions
            chunk_size=1000,
            overlap=200
        )
        
        # Generate embeddings
        for chunk in chunks:
            embedding = self.ai_client.embed(chunk['text'])
            
            # Store in both databases
            self.metadata_db.insert({
                'id': generate_uuid(),
                'content': chunk['text'],
                'metadata': metadata,
                'tags': metadata.get('tags', []),
                'chunk_id': chunk['id']
            })
            
            self.vector_db.insert({
                'id': chunk['id'],
                'embedding': embedding,
                'metadata': metadata
            })
```

#### Retrieval for Decision Support

```python
class DecisionRetrieval:
    def retrieve_similar_decisions(self, query, top_k=10):
        """
        Find decisions similar to current situation.
        
        Hybrid search:
        1. Semantic similarity (vector search)
        2. Keyword matching (FTS5)
        3. Metadata filtering (date, tags, source)
        """
        
        # Semantic search
        query_embedding = self.ai_client.embed(query)
        vector_results = self.vector_db.search(
            query_embedding, 
            top_k=top_k,
            filter={'source_type': 'decision'}
        )
        
        # Keyword search
        fts_results = self.metadata_db.search(
            query,
            fts5_query=self._extract_keywords(query),
            filter={'source_type': 'decision'}
        )
        
        # Merge and rank
        merged = self._merge_results(vector_results, fts_results)
        
        return self._rank_by_relevance(merged, query)
    
    def get_decision_history(self, decision_type=None, outcome=None):
        """Retrieve all past decisions with optional filtering."""
        return self.metadata_db.query(
            source_type='decision',
            decision_type=decision_type,
            outcome=outcome
        )
```

---

## 5. Awesome Decision Making / Reinforcement Learning Resources

**Repository:** https://github.com/jiachenli94/Awesome-Decision-Making-Reinforcement-Learning  
**Stars:** 240  
**Maintainer:** Jiachen Li (UC Berkeley)

Curated list of state-of-the-art resources:

### Categories

```markdown
1. RL & IRL & GAIL
   - Maximum Entropy Deep IRL
   - Guided Cost Learning
   - Generative Adversarial Imitation Learning
   - InfoGAIL: Interpretable Imitation Learning
   - Self-Imitation Learning
   - AIRL: Adversarial IRL
   - Multi-Agent GAIL
   
2. Autonomous Driving
   - Imitation Learning for Driving
   - Multi-Agent RL
   - End-to-End Driving
   - Interpretable Neural Motion Planner
   - Multi-agent Negotiation
   
3. Simulators & Datasets
   - CARLA: Open Urban Driving Simulator
   - TORCS: Racing Car Simulator
   - NGSIM: Highway Trajectory Dataset
```

### Key Papers for Decision Learning

```python
# Foundational decision-making references:

foundational_papers = {
    "Maximum Entropy Deep IRL (2015)": {
        "insight": "Learn reward functions from demonstrations",
        "application": "Learn user preferences from past decisions"
    },
    
    "GAIL - Generative Adversarial Imitation Learning (2016)": {
        "insight": "Generate new behaviors from demonstrations",
        "application": "Suggest decisions similar to user's style"
    },
    
    "Self-Imitation Learning (2018)": {
        "insight": "Learn from past good experiences",
        "application": "Replay past successful decisions"
    },
    
    "Multi-Agent GAIL (2018)": {
        "insight": "Multi-agent coordination",
        "application": "Track decisions involving multiple stakeholders"
    }
}
```

---

## 6. Implementation Strategy for Shadow AI

### 6.1 Core Decision Learning Module

```python
# shadow_ai/decision_learning/core.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json

class DecisionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DEFERRED = "deferred"
    ABANDONED = "abandoned"

class DecisionType(Enum):
    BINARY = "binary"                    # Yes/No
    MULTI_CHOICE = "multi_choice"        # Choose one from N
    RANKING = "ranking"                  # Rank alternatives
    MULTI_CRITERIA = "multi_criteria"    # MCDA
    SEQUENTIAL = "sequential"            # Multi-step decision
    COLLABORATIVE = "collaborative"      # Involving others
    REVERSIBLE = "reversible"           # Can be undone
    IRREVERSIBLE = "irreversible"        # Cannot be undone

@dataclass
class DecisionOption:
    id: str
    name: str
    description: str
    criteria_values: Dict[str, Any] = field(default_factory=dict)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    estimated_outcomes: List[Dict] = field(default_factory=list)
    
@dataclass
class Decision:
    id: str
    question: str
    decision_type: DecisionType
    status: DecisionStatus
    context: str
    options: List[DecisionOption]
    criteria: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Resolution
    chosen_option: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    
    # Outcome tracking
    actual_outcome: Optional[str] = None
    outcome_satisfaction: Optional[int] = None  # 1-5 scale
    outcome_learned_from: bool = False
    
    # Deferal tracking
    defer_count: int = 0
    defer_reasons: List[str] = field(default_factory=list)
    
    # Learning metadata
    embedding: Optional[List[float]] = None
    similar_past_decisions: List[str] = field(default_factory=list)
    pattern_id: Optional[str] = None

@dataclass
class DecisionPattern:
    """Learned pattern from historical decisions."""
    id: str
    pattern_type: str
    context_features: Dict[str, Any]
    typical_criteria: List[str]
    typical_weights: Dict[str, float]
    success_rate: float
    avg_resolution_time: float  # hours
    sample_decision_ids: List[str]
```

### 6.2 Decision Storage with Dual Backend

```python
# shadow_ai/decision_learning/storage.py

import sqlite3
from typing import List, Optional, Dict
import chromadb
from datetime import datetime
import json

class DecisionStorage:
    """
    Dual-storage for decisions:
    - SQLite: Metadata, FTS5 for keyword search
    - ChromaDB: Vector embeddings for semantic similarity
    """
    
    def __init__(self, db_path: str, vector_db_path: str):
        # SQLite for metadata and FTS5
        self.conn = sqlite3.connect(db_path)
        self._init_metadata_db()
        
        # ChromaDB for vectors
        self.vector_client = chromadb.PersistentClient(path=vector_db_path)
        self.decision_collection = self.vector_client.get_or_create_collection(
            name="decisions",
            metadata={"description": "Decision embeddings"}
        )
        
    def _init_metadata_db(self):
        """Create SQLite tables with FTS5."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                status TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP,
                deadline TIMESTAMP,
                resolved_at TIMESTAMP,
                chosen_option TEXT,
                reasoning TEXT,
                confidence REAL,
                actual_outcome TEXT,
                outcome_satisfaction INTEGER,
                defer_count INTEGER,
                defer_reasons TEXT,
                pattern_id TEXT
            );
            
            CREATE TABLE IF NOT EXISTS decision_options (
                id TEXT PRIMARY KEY,
                decision_id TEXT,
                name TEXT,
                description TEXT,
                criteria_values TEXT,
                pros TEXT,
                cons TEXT,
                estimated_outcomes TEXT,
                FOREIGN KEY (decision_id) REFERENCES decisions(id)
            );
            
            CREATE TABLE IF NOT EXISTS decision_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                context_features TEXT,
                typical_criteria TEXT,
                typical_weights TEXT,
                success_rate REAL,
                avg_resolution_time REAL,
                sample_decision_ids TEXT
            );
            
            -- Full-text search for decisions
            CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
                question,
                context,
                reasoning,
                actual_outcome,
                content='decisions',
                content_rowid='rowid'
            );
            
            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions
            BEGIN
                INSERT INTO decisions_fts(rowid, question, context, reasoning, actual_outcome)
                VALUES (new.rowid, new.question, new.context, new.reasoning, new.actual_outcome);
            END;
            
            CREATE TRIGGER IF NOT EXISTS decisions_ad AFTER DELETE ON decisions
            BEGIN
                INSERT INTO decisions_fts(decisions_fts, rowid, question, context, reasoning, actual_outcome)
                VALUES('delete', old.rowid, old.question, old.context, old.reasoning, old.actual_outcome);
            END;
            
            CREATE TRIGGER IF NOT EXISTS decisions_au AFTER UPDATE ON decisions
            BEGIN
                INSERT INTO decisions_fts(decisions_fts, rowid, question, context, reasoning, actual_outcome)
                VALUES('delete', old.rowid, old.question, old.context, old.reasoning, old.actual_outcome);
                INSERT INTO decisions_fts(rowid, question, context, reasoning, actual_outcome)
                VALUES (new.rowid, new.question, new.context, new.reasoning, new.actual_outcome);
            END;
        """)
        self.conn.commit()
    
    def store_decision(self, decision: Decision):
        """Store decision in both databases."""
        # SQLite metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO decisions 
            (id, question, decision_type, status, context, created_at, 
             deadline, resolved_at, chosen_option, reasoning, confidence,
             actual_outcome, outcome_satisfaction, defer_count, defer_reasons, pattern_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.id, decision.question, decision.decision_type.value,
            decision.status.value, decision.context, decision.created_at,
            decision.deadline, decision.resolved_at, decision.chosen_option,
            decision.reasoning, decision.confidence, decision.actual_outcome,
            decision.outcome_satisfaction, decision.defer_count,
            json.dumps(decision.defer_reasons), decision.pattern_id
        ))
        
        # Store options
        for option in decision.options:
            self.conn.execute("""
                INSERT OR REPLACE INTO decision_options
                (id, decision_id, name, description, criteria_values, pros, cons, estimated_outcomes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                option.id, decision.id, option.name, option.description,
                json.dumps(option.criteria_values), json.dumps(option.pros),
                json.dumps(option.cons), json.dumps(option.estimated_outcomes)
            ))
        
        self.conn.commit()
        
        # ChromaDB embedding
        if decision.embedding:
            self.vector_collection.upsert(
                ids=[decision.id],
                embeddings=[decision.embedding],
                metadatas=[{
                    'question': decision.question,
                    'status': decision.status.value,
                    'decision_type': decision.decision_type.value,
                    'created_at': decision.created_at.isoformat()
                }],
                documents=[f"{decision.question}\n{decision.context}"]
            )
    
    def search_similar_decisions(self, query_embedding: List[float], 
                                  top_k: int = 10,
                                  filter_status: Optional[DecisionStatus] = None) -> List[Decision]:
        """Semantic search for similar decisions."""
        filters = {}
        if filter_status:
            filters['status'] = filter_status.value
            
        results = self.vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters if filters else None
        )
        
        decisions = []
        for decision_id in results['ids'][0]:
            decision = self.get_decision(decision_id)
            if decision:
                decisions.append(decision)
        
        return decisions
    
    def keyword_search_decisions(self, query: str, 
                                   top_k: int = 10,
                                   filter_status: Optional[DecisionStatus] = None) -> List[Decision]:
        """FTS5 keyword search for decisions."""
        sql = """
            SELECT d.id FROM decisions d
            JOIN decisions_fts fts ON d.rowid = fts.rowid
            WHERE decisions_fts MATCH ?
        """
        params = [query]
        
        if filter_status:
            sql += " AND d.status = ?"
            params.append(filter_status.value)
        
        sql += f" ORDER BY bm25(decisions_fts) LIMIT {top_k}"
        
        cursor = self.conn.execute(sql, params)
        decision_ids = [row[0] for row in cursor.fetchall()]
        
        return [self.get_decision(did) for did in decision_ids]
    
    def get_overdue_decisions(self) -> List[Decision]:
        """Get all overdue decisions."""
        cursor = self.conn.execute("""
            SELECT id FROM decisions
            WHERE status = 'pending' 
            AND deadline < ?
            AND deadline IS NOT NULL
        """, (datetime.now(),))
        
        return [self.get_decision(row[0]) for row in cursor.fetchall()]
```

### 6.3 Multi-Criteria Decision Analysis Integration

```python
# shadow_ai/decision_learning/mcda.py

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class MCDAConfig:
    method: str = "topsis"  # topsis, electre, promethee, vikor, etc.
    sensitivity_analysis: bool = True
    confidence_threshold: float = 0.7
    require_weights: bool = False

class MCDADecisionEngine:
    """
    Integration of pyDecision methods into Shadow AI.
    
    Helps users make complex decisions with multiple criteria:
    - Weight derivation via AHP
    - Alternative ranking via TOPSIS, ELECTRE, etc.
    - Sensitivity analysis
    - Multiple method consensus
    """
    
    def __init__(self, config: MCDAConfig = None):
        self.config = config or MCDAConfig()
        # Import pyDecision methods
        from pyDecision.algorithm import (
            topsis_method, ahp_method, electre_ii_method,
            promethee_ii_method, vikor_method, maut_method
        )
        self.methods = {
            'topsis': topsis_method,
            'ahp': ahp_method,
            'electre': electre_ii_method,
            'promethee': promethee_ii_method,
            'vikor': vikor_method,
            'maut': maut_method
        }
        
    def derive_weights_ahp(self, 
                           criteria: List[str],
                           pairwise_comparisons: Dict[Tuple[str, str], int]) -> Tuple[Dict[str, float], float]:
        """
        Derive weights using Analytic Hierarchy Process.
        
        Args:
            criteria: List of criteria names
            pairwise_comparisons: Dict of (criterion1, criterion2) -> comparison_value
                                  where value is 1-9 scale importance
        
        Returns:
            weights: Dict of criterion -> weight
            consistency_ratio: AHP consistency (< 0.1 is good)
        """
        n = len(criteria)
        comparison_matrix = np.ones((n, n))
        
        for i, c1 in enumerate(criteria):
            for j, c2 in enumerate(criteria):
                if i < j:
                    comparison_matrix[i, j] = pairwise_comparisons.get((c1, c2), 1)
                    comparison_matrix[j, i] = 1 / comparison_matrix[i, j]
        
        weights, consistency_ratio = ahp_method(comparison_matrix, wd='geometric')
        
        return {
            criteria[i]: weights[i] 
            for i in range(n)
        }, consistency_ratio
    
    def interactive_weight_derivation(self, criteria: List[str]) -> Dict[str, float]:
        """
        Guide user through AHP process interactively.
        
        Returns weights derived from user preferences.
        """
        print(f"We'll derive weights for {len(criteria)} criteria: {criteria}")
        print("For each pair, rate on scale 1-9:")
        print("1 = equal importance, 9 = extreme importance")
        print()
        
        comparisons = {}
        for i, c1 in enumerate(criteria):
            for j, c2 in enumerate(criteria):
                if i < j:
                    while True:
                        try:
                            val = int(input(f"How important is '{c1}' compared to '{c2}'? (1-9): "))
                            if 1 <= val <= 9:
                                comparisons[(c1, c2)] = val
                                break
                        except ValueError:
                            pass
        
        weights, cr = self.derive_weights_ahp(criteria, comparisons)
        
        if cr > 0.1:
            print(f"\nWarning: Consistency ratio {cr:.3f} > 0.1")
            print("Consider reviewing your comparisons for consistency.")
        
        return weights
    
    def analyze_decision(self,
                         alternatives: List[str],
                         criteria_names: List[str],
                         performance_matrix: np.ndarray,
                         weights: Optional[Dict[str, float]] = None,
                         criteria_types: Optional[List[str]] = None) -> Dict:
        """
        Run MCDA analysis on a decision.
        
        Args:
            alternatives: List of alternative names
            criteria_names: List of criterion names
            performance_matrix: n_alternatives x n_criteria matrix
            weights: Optional weights (derived via AHP if not provided)
            criteria_types: 'max' or 'min' for each criterion
        
        Returns:
            Dict with rankings, scores, and sensitivity analysis
        """
        # Derive weights if not provided
        if weights is None and self.config.require_weights:
            raise ValueError("Weights required but not provided")
        elif weights is None:
            # Equal weights as default
            weights = {c: 1.0/len(criteria_names) for c in criteria_names}
        
        weight_array = np.array([weights[c] for c in criteria_names])
        
        # Default: maximize all criteria
        if criteria_types is None:
            criteria_types = ['max'] * len(criteria_names)
        
        # Apply selected MCDA method
        method_func = self.methods[self.config.method]
        result = method_func(performance_matrix, weight_array, 
                            criteria_types=criteria_types)
        
        # Sensitivity analysis
        if self.config.sensitivity_analysis:
            sensitivity = self._sensitivity_analysis(
                alternatives, criteria_names, performance_matrix,
                weight_array, criteria_types
            )
        else:
            sensitivity = None
        
        # Build result
        rankings = {
            alternatives[i]: result['ranking'][i]
            for i in range(len(alternatives))
        }
        
        return {
            'rankings': rankings,
            'scores': result.get('scores', None),
            'method': self.config.method,
            'weights': weights,
            'sensitivity': sensitivity,
            'confidence': self._calculate_confidence(result, alternatives)
        }
    
    def _sensitivity_analysis(self, 
                             alternatives, 
                             criteria, 
                             perf_matrix,
                             weights,
                             criteria_types):
        """
        Test how robust the ranking is to weight changes.
        
        - Small weight perturbations
        - Rank reversal check
        - Critical criterion identification
        """
        base_result = self.analyze_decision(
            alternatives, criteria, perf_matrix, 
            {c: w for c, w in zip(criteria, weights)},
            criteria_types
        )
        
        perturbed_results = []
        for perturbation in [-0.1, 0.1]:  # ±10% weight changes
            for i, criterion in enumerate(criteria):
                perturbed_weights = weights.copy()
                perturbed_weights[i] *= (1 + perturbation)
                perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Normalize
                
                result = self.analyze_decision(
                    alternatives, criteria, perf_matrix,
                    {c: w for c, w in zip(criteria, perturbed_weights)},
                    criteria_types
                )
                perturbed_results.append({
                    'perturbation': perturbation,
                    'criterion': criterion,
                    'ranking_change': self._ranking_difference(
                        base_result['rankings'], 
                        result['rankings']
                    )
                })
        
        return perturbed_results
    
    def _ranking_difference(self, base_rankings, new_rankings):
        """Calculate ranking difference metric."""
        # Kendall tau distance
        base_order = sorted(base_rankings.keys(), key=lambda x: base_rankings[x])
        new_order = sorted(new_rankings.keys(), key=lambda x: new_rankings[x])
        
        # Simple difference metric
        diff = sum(base_order[i] != new_order[i] for i in range(len(base_order)))
        return diff
    
    def _calculate_confidence(self, result, alternatives):
        """
        Calculate confidence score for the decision.
        
        Higher confidence when:
        - Clear separation between top alternatives
        - Multiple methods agree
        - Low sensitivity to weight changes
        """
        # This is a heuristic confidence measure
        # Can be refined with more sophisticated approaches
        return self.config.confidence_threshold
```

### 6.4 Decision Learning and Analytics

```python
# shadow_ai/decision_learning/analytics.py

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

@dataclass
class DecisionPattern:
    pattern_id: str
    pattern_type: str  # 'recurrence', 'preference', 'context'
    context_features: Dict[str, any]
    typical_outcome: str
    success_rate: float
    sample_size: int
    recommendations: List[str]

class DecisionAnalytics:
    """
    Analyze historical decisions to:
    - Identify patterns
    - Learn user preferences
    - Predict outcomes
    - Improve future recommendations
    """
    
    def __init__(self, storage):
        self.storage = storage
    
    def identify_patterns(self, 
                         min_sample_size: int = 5) -> List[DecisionPattern]:
        """
        Find recurring decision patterns.
        
        Pattern types:
        1. Recurrence: Same type of decision made repeatedly
        2. Preference: Consistent choice under certain conditions
        3. Context: Decisions made in similar situations
        """
        patterns = []
        
        # Group decisions by type
        decisions_by_type = defaultdict(list)
        for decision in self.storage.get_all_decisions():
            decisions_by_type[decision.decision_type].append(decision)
        
        for decision_type, decisions in decisions_by_type.items():
            if len(decisions) < min_sample_size:
                continue
            
            # Analyze choices
            choice_distribution = defaultdict(int)
            context_features = defaultdict(lambda: defaultdict(int))
            outcomes = []
            
            for decision in decisions:
                if decision.chosen_option:
                    choice_distribution[decision.chosen_option] += 1
                
                # Extract context features
                for key, value in self._extract_features(decision).items():
                    context_features[key][value] += 1
                
                if decision.actual_outcome:
                    outcomes.append(decision.outcome_satisfaction or 0)
            
            # Check for dominant preference
            total_choices = sum(choice_distribution.values())
            max_choice = max(choice_distribution.items(), key=lambda x: x[1])
            
            if max_choice[1] / total_choices > 0.6:  # 60%+ consistency
                patterns.append(DecisionPattern(
                    pattern_id=f"pref_{decision_type.value}_{max_choice[0]}",
                    pattern_type='preference',
                    context_features=dict(context_features),
                    typical_outcome=max_choice[0],
                    success_rate=np.mean(outcomes) if outcomes else None,
                    sample_size=len(decisions),
                    recommendations=[
                        f"Consistently choose '{max_choice[0]}' for {decision_type.value} decisions",
                        f"This preference appears {max_choice[1]/total_choices:.0%} of the time"
                    ]
                ))
        
        return patterns
    
    def _extract_features(self, decision) -> Dict[str, str]:
        """Extract context features from a decision."""
        features = {}
        
        # Temporal features
        features['time_of_day'] = decision.created_at.hour // 6  # 0-3
        features['day_of_week'] = decision.created_at.weekday()
        
        # Context words
        context_lower = decision.context.lower()
        
        # Domain detection
        domains = ['work', 'personal', 'finance', 'health', 'relationship', 'project']
        for domain in domains:
            if domain in context_lower:
                features['domain'] = domain
                break
        
        # Urgency detection
        if decision.deadline:
            time_pressure = (decision.deadline - decision.created_at).days
            features['urgency'] = 'high' if time_pressure < 7 else 'medium' if time_pressure < 30 else 'low'
        
        # Number of options
        features['option_count'] = len(decision.options)
        
        return features
    
    def learn_from_outcome(self, decision: Decision) -> Dict[str, any]:
        """
        Compare expected vs actual outcome.
        
        Returns:
            - Outcome correlation
            - Lessons learned
            - Updated predictions
        """
        if not decision.expected_outcomes or not decision.actual_outcome:
            return {'status': 'insufficient_data'}
        
        # Compare expected vs actual
        correlation = self._compare_outcomes(
            decision.expected_outcomes,
            decision.actual_outcome
        )
        
        # Find similar decisions
        similar = self.storage.search_similar_decisions(
            decision.embedding, top_k=5
        )
        
        # Update pattern predictions
        lessons = self._extract_lessons(decision, similar)
        
        return {
            'expected_vs_actual_correlation': correlation,
            'lessons_learned': lessons,
            'confidence_adjustment': -0.1 if correlation < 0.5 else 0.1
        }
    
    def _compare_outcomes(self, expected: List[Dict], actual: str):
        """Calculate how well prediction matched reality."""
        # Simple text similarity approach
        # Could be enhanced with semantic similarity
        actual_lower = actual.lower()
        
        for expectation in expected:
            if expectation.get('description', '').lower() in actual_lower:
                return expectation.get('probability', 0.5)
        
        return 0.0
    
    def predict_outcome(self, decision: Decision) -> Dict[str, any]:
        """
        Predict outcome based on similar past decisions.
        
        Uses:
        - Semantic similarity
        - Pattern matching
        - Historical success rates
        """
        if not decision.embedding:
            return {'status': 'no_embedding'}
        
        # Find similar decisions
        similar = self.storage.search_similar_decisions(
            decision.embedding, top_k=10
        )
        
        if not similar:
            return {'status': 'insufficient_history'}
        
        # Aggregate outcomes
        outcome_distribution = defaultdict(int)
        satisfaction_scores = []
        
        for sim_decision in similar:
            if sim_decision.actual_outcome:
                outcome_distribution[sim_decision.actual_outcome] += 1
            if sim_decision.outcome_satisfaction:
                satisfaction_scores.append(sim_decision.outcome_satisfaction)
        
        # Most common outcome
        predicted_outcome = max(outcome_distribution.items(), 
                               key=lambda x: x[1], 
                               default=(None, 0))
        
        # Average satisfaction
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else None
        
        # Confidence based on similarity
        confidence = predicted_outcome[1] / len(similar) if similar else 0
        
        return {
            'predicted_outcome': predicted_outcome[0],
            'confidence': confidence,
            'expected_satisfaction': avg_satisfaction,
            'based_on_similar_decisions': len(similar),
            'outcome_distribution': dict(outcome_distribution)
        }
    
    def generate_recommendations(self, 
                                  current_decision: Decision,
                                  historical_patterns: List[DecisionPattern]) -> List[str]:
        """
        Generate recommendations for current decision.
        
        Based on:
        - Matched patterns
        - Past successful choices
        - Avoidance of past mistakes
        """
        recommendations = []
        
        # Match patterns
        matched_patterns = []
        for pattern in historical_patterns:
            if self._matches_pattern(current_decision, pattern):
                matched_patterns.append(pattern)
        
        for pattern in matched_patterns:
            if pattern.success_rate and pattern.success_rate > 0.7:
                recommendations.append(
                    f"Pattern detected: {pattern.recommendations[0]}"
                )
        
        # Check for defer warnings
        if current_decision.defer_count >= 3:
            recommendations.append(
                "This decision has been deferred multiple times. "
                "Consider taking action or explicitly abandoning."
            )
        
        # Check deadline pressure
        if current_decision.deadline:
            days_left = (current_decision.deadline - datetime.now()).days
            if days_left < 3:
                recommendations.append(
                    f"Urgent: Only {days_left} days until deadline. "
                    "Prioritize quick resolution."
                )
        
        return recommendations
    
    def _matches_pattern(self, decision: Decision, pattern: DecisionPattern) -> bool:
        """Check if decision matches a known pattern."""
        if decision.decision_type.value not in pattern.pattern_id:
            return False
        
        decision_features = self._extract_features(decision)
        
        # Check feature overlap
        matches = 0
        total = len(pattern.context_features)
        
        for key, dist in pattern.context_features.items():
            if key in decision_features:
                if decision_features[key] in dist:
                    matches += 1
        
        return matches / total > 0.5 if total > 0 else False
```

### 6.5 MCP Server Integration (Model Context Protocol)

```python
# shadow_ai/decision_learning/mcp_server.py

"""
MCP Server for Shadow AI Decision Learning.

Allows Claude Code to interact with Shadow AI's decision tracking
system through the Model Context Protocol.
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio
from datetime import datetime
import json

server = Server("shadow-ai-decisions")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="decision_create",
            description="Create a new decision to track",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "string"},
                    "deadline": {"type": "string", "format": "date"},
                    "criteria": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["question", "options"]
            }
        ),
        Tool(
            name="decision_resolve",
            description="Resolve a decision with chosen option",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "chosen_option": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["decision_id", "chosen_option"]
            }
        ),
        Tool(
            name="decision_defer",
            description="Defer a decision to later",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "new_deadline": {"type": "string", "format": "date"}
                },
                "required": ["decision_id", "reason"]
            }
        ),
        Tool(
            name="decision_record_outcome",
            description="Record actual outcome of a resolved decision",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "actual_outcome": {"type": "string"},
                    "satisfaction": {"type": "integer", "minimum": 1, "maximum": 5}
                },
                "required": ["decision_id", "actual_outcome"]
            }
        ),
        Tool(
            name="decision_search_similar",
            description="Find similar past decisions",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="decision_get_recommendations",
            description="Get recommendations for a decision",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"}
                },
                "required": ["decision_id"]
            }
        ),
        Tool(
            name="decision_mcda_analyze",
            description="Run multi-criteria decision analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "weights": {"type": "object"},
                    "method": {"type": "string", "enum": ["topsis", "electre", "promethee", "vikor"]}
                },
                "required": ["decision_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    from .core import DecisionTracker
    from .storage import DecisionStorage
    from .mcda import MCDADecisionEngine
    from .analytics import DecisionAnalytics
    
    tracker = DecisionTracker()
    storage = DecisionStorage("data/decisions.db", "data/vectors")
    mcda = MCDADecisionEngine()
    analytics = DecisionAnalytics(storage)
    
    if name == "decision_create":
        decision = tracker.create_decision(
            question=arguments["question"],
            options=arguments["options"],
            context=arguments.get("context", ""),
            deadline=arguments.get("deadline"),
            criteria=arguments.get("criteria", [])
        )
        storage.store_decision(decision)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "created",
                "decision_id": decision.id,
                "message": f"Decision '{decision.question}' created with ID {decision.id}",
                "deadline": decision.deadline.isoformat() if decision.deadline else None
            }, indent=2)
        )]
    
    elif name == "decision_resolve":
        decision = storage.get_decision(arguments["decision_id"])
        if not decision:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Decision not found"})
            )]
        
        tracker.resolve_decision(
            decision,
            chosen_option=arguments["chosen_option"],
            reasoning=arguments.get("reasoning", ""),
            confidence=arguments.get("confidence")
        )
        storage.store_decision(decision)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "resolved",
                "decision_id": decision.id,
                "chosen_option": decision.chosen_option,
                "resolved_at": decision.resolved_at.isoformat()
            }, indent=2)
        )]
    
    elif name == "decision_defer":
        decision = storage.get_decision(arguments["decision_id"])
        if not decision:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Decision not found"})
            )]
        
        tracker.defer_decision(
            decision,
            reason=arguments["reason"],
            new_deadline=arguments.get("new_deadline")
        )
        storage.store_decision(decision)
        
        warning = ""
        if decision.defer_count >= 3:
            warning = " Warning: This decision has been deferred 3+ times. Consider resolving or abandoning."
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "deferred",
                "decision_id": decision.id,
                "defer_count": decision.defer_count,
                "warning": warning
            }, indent=2)
        )]
    
    elif name == "decision_record_outcome":
        decision = storage.get_decision(arguments["decision_id"])
        if not decision:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Decision not found"})
            )]
        
        tracker.record_outcome(
            decision,
            actual_outcome=arguments["actual_outcome"],
            satisfaction=arguments.get("satisfaction")
        )
        storage.store_decision(decision)
        
        # Learn from outcome
        lessons = analytics.learn_from_outcome(decision)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "outcome_recorded",
                "decision_id": decision.id,
                "lessons_learned": lessons.get("lessons_learned", []),
                "confidence_adjustment": lessons.get("confidence_adjustment", 0)
            }, indent=2)
        )]
    
    elif name == "decision_search_similar":
        # Generate embedding for query
        from .embeddings import generate_embedding
        query_embedding = generate_embedding(arguments["query"])
        
        similar = storage.search_similar_decisions(
            query_embedding,
            top_k=arguments.get("top_k", 5)
        )
        
        results = []
        for dec in similar:
            results.append({
                "decision_id": dec.id,
                "question": dec.question,
                "chosen_option": dec.chosen_option,
                "outcome": dec.actual_outcome,
                "created_at": dec.created_at.isoformat()
            })
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "query": arguments["query"],
                "similar_decisions": results
            }, indent=2)
        )]
    
    elif name == "decision_get_recommendations":
        decision = storage.get_decision(arguments["decision_id"])
        if not decision:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Decision not found"})
            )]
        
        patterns = analytics.identify_patterns()
        recommendations = analytics.generate_recommendations(decision, patterns)
        predicted_outcome = analytics.predict_outcome(decision)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "decision_id": decision.id,
                "recommendations": recommendations,
                "predicted_outcome": predicted_outcome.get("predicted_outcome"),
                "confidence": predicted_outcome.get("confidence"),
                "based_on_patterns": len(patterns)
            }, indent=2)
        )]
    
    elif name == "decision_mcda_analyze":
        decision = storage.get_decision(arguments["decision_id"])
        if not decision:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Decision not found"})
            )]
        
        # Build performance matrix from options
        alternatives = [opt.name for opt in decision.options]
        criteria_names = decision.criteria
        
        # Extract criteria values from options
        perf_matrix = []
        for option in decision.options:
            row = [option.criteria_values.get(c, 0) for c in criteria_names]
            perf_matrix.append(row)
        
        # Run MCDA
        result = mcda.analyze_decision(
            alternatives=alternatives,
            criteria_names=criteria_names,
            performance_matrix=np.array(perf_matrix),
            weights=arguments.get("weights"),
            criteria_types=['max'] * len(criteria_names)  # Could be configured
        )
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "decision_id": decision.id,
                "method": result["method"],
                "rankings": result["rankings"],
                "confidence": result["confidence"],
                "sensitivity_analysis": result["sensitivity"]
            }, indent=2)
        )]

async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Key Insights for Shadow AI Implementation

### 7.1 Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      Shadow AI Decision Learning                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │   Decision   │      │    Storage   │      │   Analytics  │   │
│  │   Tracker    │─────▶│   - SQLite   │─────▶│   - Pattern  │   │
│  │              │      │   - ChromaDB │      │     Learning │   │
│  └──────────────┘      └──────────────┘      └──────────────┘   │
│         │                     │                      │            │
│         │                     │                      │            │
│         ▼                     ▼                      ▼            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    MCP Server Interface                   │  │
│  │  - decision_create                                        │  │
│  │  - decision_resolve                                       │  │
│  │  - decision_defer                                         │  │
│  │  - decision_record_outcome                                │  │
│  │  - decision_search_similar                                │  │
│  │  - decision_get_recommendations                           │  │
│  │  - decision_mcda_analyze                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Core Features to Implement

Based on this research:

1. **Decision Lifecycle Tracking** (from graphthulhu)
   - Create, resolve, defer, abandonment tracking
   - Status transitions: PENDING → IN_PROGRESS → RESOLVED/DEFERRED/ABANDONED
   - Deadline management with overdue alerts
   - Deferral warnings (count + reason tracking)

2. **Multi-Criteria Decision Support** (from pyDecision)
   - AHP for weight derivation through pairwise comparisons
   - Multiple MCDA methods (TOPSIS, ELECTRE, PROMETHEE)
   - Sensitivity analysis for robustness
   - Interactive weight elicitation

3. **Decision Learning** (from AgentGym-RL)
   - Track expected vs actual outcomes
   - Learn from outcome satisfaction
   - Pattern identification across decisions
   - Predictive recommendations for similar decisions

4. **Journal/Context Capture** (from commit-story, WhatsAppRAG)
   - Automatic context extraction
   - Decision dialogue preservation
   - Semantic search through decision history
   - Outcome correlation analysis

5. **Analytics Dashboard**
   - Pattern visualization
   - Success rate tracking
   - Resolution time metrics
   - Decision outcome correlation

### 7.3 Code Improvements Needed

1. **Embeddings Generation**
   ```python
   # Need to implement
   def generate_decision_embedding(decision: Decision) -> List[float]:
       """
       Create embedding combining:
       - Question text
       - Context description
       - Options descriptions
       - Criteria names
       """
       combined_text = f"{decision.question}\n{decision.context}\n"
       combined_text += "\n".join(opt.description for opt in decision.options)
       return embedding_model.encode(combined_text)
   ```

2. **Pattern Mining**
   ```python
   # Need deeper pattern analysis
   def mine_decision_patterns(decisions: List[Decision]) -> List[Pattern]:
       """
       Use clustering and ML to find:
       - Recurring decision types
       - Preference patterns
       - Context similarities
       - Outcome predictors
       """
       pass
   ```

3. **Outcome Correlation**
   ```python
   # Need to implement outcome comparison
   def compare_expected_vs_actual(expected: List[Dict], actual: str) -> float:
       """
       Semantic similarity between expected outcomes
       and actual outcome.
       
       Could use:
       - Embedding similarity
       - Keyword overlap
       - NLI-based comparison
       """
       pass
   ```

4. **Decision Quality Metrics**
   ```python
   # Need metrics for decision quality
   class DecisionQuality:
       time_to_resolution: float
       defer_count: int
       outcome_prediction_accuracy: float
       pattern_match_confidence: float
       user_satisfaction: int
       
       def overall_score(self) -> float:
           # Weighted combination
           pass
   ```

5. **MCDA Sensitivity**
   ```python
   # Need more sophisticated sensitivity analysis
   def monte_carlo_sensitivity(
       performance_matrix: np.ndarray,
       weights: np.ndarray,
       n_iterations: int = 1000
   ) -> Dict:
       """
       Monte Carlo simulation for ranking stability:
       - Perturb weights randomly
       - Re-rank alternatives
       - Calculate probability of rank reversal
       - Identify critical criteria
       """
       pass
   ```

---

## 8. Recommended Implementation Priority

### Phase 1: Core Decision Tracking (Week 1-2)
- Decision data model
- Storage layer (SQLite + ChromaDB)
- Basic CRUD operations
- MCP server integration

### Phase 2: Decision Lifecycle (Week 3-4)
- Status tracking
- Deadline management
- Deferral with warnings
- Resolution recording

### Phase 3: MCDA Integration (Week 5-6)
- AHP weight derivation
- TOPSIS ranking
- Sensitivity analysis
- Interactive weight elicitation

### Phase 4: Learning & Analytics (Week 7-8)
- Outcome tracking
- Pattern identification
- Similar decision retrieval
- Recommendation generation

### Phase 5: Advanced Features (Week 9-10)
- Predictive outcomes
- Pattern mining
- Quality metrics
- Visualization dashboard

---

## 9. References

1. AgentGym-RL: https://github.com/WooooDyy/AgentGym-RL (663 stars, ICLR 2026)
2. RL4VLM: https://github.com/RL4VLM/RL4VLM (409 stars, NeurIPS 2024)
3. pyDecision: https://github.com/Valdecy/pyDecision (343 stars)
4. graphthulhu: https://github.com/skridlevsky/graphthulhu (125 stars)
5. commit-story: https://github.com/wiggitywhitney/commit-story (18 stars)
6. WhatsAppRAG: https://github.com/MQasimJalil/WhatsAppRAG (1 star)
7. Applied RL: https://github.com/mimoralea/applied-reinforcement-learning (331 stars)
8. Awesome-Decision-Making-RL: https://github.com/jiachenli94/Awesome-Decision-Making-Reinforcement-Learning (240 stars)

---

**End of Research Document**