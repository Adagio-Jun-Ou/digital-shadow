# Digital Shadow — Project Status

*Honest scorecard. README.md is aspirational; this file is what currently works.*

Last audited: 2026-09-12

---

## TL;DR

- **Full test suite is 100% green**: 9918 passed, 31 skipped, 0 failures, ~8 minutes (single process, Windows).
- **Daemon**: starts cleanly, FastAPI + WebSocket, ~680 routes total. Auth works (JWT bearer) when `SHADOW_AUTH_ENABLED=true`. **Currently offline** as instructed.
- **21 shikigami abilities**: all wired, all work end-to-end via `/shikigami/summon` and `/shikigami/council`.
- **Native sensors**: BLE discovery, WiFi scan, audio activity, battery, load — all real.
- **Proactive insight engine**: background scanner emits manipulation / sentiment-shift / council / anomaly insights via WebSocket + `/insights/recent`.
- **Reality Forge**: now executes commands through an allowlist-based safe runner (workspace-cwd'd, output-capped, opt-in writes).
- **Web dashboard**: PWA mounted at `/dashboard` directly on the main daemon.
- **CLI**: `--list`, `--status`, `--memory`, `--insights`, `--observers`, `--login`, `--logout`, `--shikigami` flag, `--observe`.
- **Cognition v2** (`shadowd/cognition/`): 91 modules, 642 `/os/*` routes responding; multi-agent missions work. Not run as autonomous daemon by default.
- **FeedLoop dispatcher**: queued cross-module signals are now consumed and routed into real module calls exactly once (`_dispatch_feed_signals`), with dedup and dispatch stats in `get_status()`.
- **Cross-platform**: full suite runs on Windows as well as Linux (atomic writes, sandbox path parsing, POSIX-path observation handling, UTF-8 defaults).

---

## 2026-09-12 review pass

- Replaced POSIX-only directory fsync with `shadowd/atomic_io.py` (`fsync_dir` skips Windows) across 8 call sites.
- Fixed `oracle_bridge` HTTP fallback: `httpx` timeouts no longer escape (they do not subclass builtin `TimeoutError`).
- Fixed `knowledge_graph` / `escalation` interval boundaries (half-open `[from, until)`).
- Fixed fd leak when `os.fdopen` fails during atomic writes (Windows file locking).
- Fixed Windows path handling in `lantern` project-root detection and restricted-path validators.
- Hardened `SHADOW_FILE_SANDBOX` parsing for Windows drive letters; CLI now falls back gracefully without a console.
- Test harness: forced UTF-8 default I/O, tolerant temp-dir cleanup, repo-relative source paths, and de-fanged time-bomb tests (relative timestamps).


---

## How to run it

```bash
cd ~/digital-shadow
./start-shadowd.sh                # daemon on http://127.0.0.1:8765
./status-shadowd.sh               # JSON status
./stop-shadowd.sh

# Talk to it
venv/bin/python summon/summon_cli.py --list
venv/bin/python summon/summon_cli.py --shikigami oracle "Will tomorrow be productive?"
venv/bin/python summon/summon_cli.py --shikigami omni_sense "what's around me"
venv/bin/python summon/summon_cli.py "open question, auto-route to ability"
```

Use the venv. System Python is missing `hypothesis` (needed for ~160 tests).

---

## Ability status (the 10 shikigami)

| # | Ability         | Status      | Backing algorithms                                                  |
|---|-----------------|-------------|---------------------------------------------------------------------|
| 1 | Oracle          | ✅ Real      | Monte Carlo (10k sims) + Bayesian + Markov + temporal pattern engine |
| 2 | X-Ray           | ✅ Real      | bluetoothctl active discovery + RSSI per device + path-loss distance |
| 3 | Life Pulse      | ✅ Real      | Device-constellation matching, presence confidence scoring          |
| 4 | Perfect Memory  | ✅ Real      | Keyword + temporal + entity + knowledge-graph **+ TF-IDF semantic** |
| 5 | Pattern Sight   | ✅ Real      | FFT, correlation, decision-tree classification                      |
| 6 | Social Radar    | ✅ Real *    | RL/bandit + **lexicon sentiment + manipulation pattern detection**  |
| 7 | Intent Seer     | ✅ Real      | Markov state-transition prediction                                  |
| 8 | Omni Sense      | ✅ Real      | BT + WiFi + **audio/battery/load fusion**, location signature       |
| 9 | Shadow Council  | ✅ Real *    | **Action-category clustering, recency weight, trend, counterfactual** |
| 10| Reality Forge   | ✅ Real      | LLM-driven file generation + **allowlist-based safe command runner** |
| 11| Habit Tracker   | ✅ Real      | Weekday-hour bucketing + ISO-week streaks + broken-streak alerts    |
| 12| Energy Flow     | ✅ Real      | 3-hour smoothed circadian curve + 6-hour forecast + ASCII chart     |
| 13| Focus Sight     | ✅ Real      | Contiguous-activity blocks with purity score + current state        |
| 14| Connection      | ✅ Real *    | Recency + frequency + sentiment + reciprocity → strength per person |
| 15| Compass         | ✅ Real *    | Goal alignment vs baseline + trend + target-date urgency            |
| 16| Hearth          | ✅ Real      | Location classification + transitions + hour-mode + anomalies       |
| 17| Time Witness    | ✅ Real      | Per-category time budget over a sliding window + delta vs prior     |
| 18| Mirror          | ✅ Real      | First-person theme detection + self-talk sentiment + trend          |
| 19| Lantern         | ✅ Real      | Project state from file_change observations — active/idle/dormant + revisit candidates |
| 20| Tide            | ✅ Real      | Weekly sentiment trend with linear-slope detection + streak tracking |
| 21| Echo            | ✅ Real *    | Counterfactual analysis over DecisionTracker history with regret-pressure ranking |

\* Outputs depend on having tracked data. Social Radar needs at least one person tracked (`POST /track/person`); Shadow Council needs decisions tracked (`POST /track/decision`); Connection needs people tracked; Compass needs goals registered (`POST /track/goal`).

**Bold** = added in 2026-05 refinement pass. Previous gaps closed:
- BLE scanner was returning empty — now actively discovers + reads RSSI
- WiFi scanner was a stub — now nmcli/iwlist with real signal strength
- Perfect Memory had only keyword match — added cosine-similarity layer (`shikigami/semantic_memory.py`)
- Social Radar had no sentiment — added VADER-style scoring + manipulation detector (`shikigami/sentiment.py`)
- Shadow Council only listed past — added cross-decision synthesis (`shikigami/council_synthesis.py`)
- Omni Sense missed audio/battery/load — added native sensor fusion (`sensors/native_sensors.py`)
- `summon` CLI called nonexistent `/summon` and had no `--shikigami` flag — rewritten to hit `/shikigami/summon` with the flag README documents
- Tracked decisions / people didn't propagate to the live shikigami registry — added bridges in `shadow_ai.track_decision` and `track_person`
- Multi-armed bandit `KeyError` on never-seen action names (e.g. `engage_<new_person>`) — now lazily registers arms
- `EnvironmentModel._update_patterns` raised KeyError after a daemon restart because saved `defaultdict` deserialized to plain `dict` — now promotes back on access
- Bandit returned `+inf` UCB scores that broke JSON serialization in `/shikigami/summon` — clamped with `unexplored: true` flag
- Filesystem observer was logging its own log file as `file_change` events, creating a self-feedback loop — added ignore list (`/data/logs/`, caches, `.tmp`, `.swp`)
- Pre-existing test failures: `test_pair_device` (`params=` vs `json=`) and `test_observe_location` (missing `X-API-Key`) — fixed both

### Round 2 (2026-05-16) — feature additions

- **Embeddings-backed Perfect Memory**: `shikigami/embedding_index.py` — tries an Ollama-compatible embeddings endpoint (local server or cloud), caches vectors on disk by content hash, falls back to TF-IDF. Currently no embedding model is reachable from this account, so Perfect Memory uses TF-IDF — but the path is in place.
- **Proactive Insight Engine**: `shadowd/proactive_engine.py` — runs every 60s by default, scans recent observations for manipulation patterns, sentiment shifts per tracked person, council recommendations with high confidence, hourly-activity anomalies. Broadcasts on WebSocket `insights` topic and exposes a ring buffer via `/insights/recent`.
- **Reality Forge safe runner**: `shikigami/safe_runner.py` — Reality Forge now actually executes commands (instead of just logging them) via a hardened runner. Read-only allowlist, opt-in write allowlist (`SHADOW_FORGE_EXECUTE_WRITES=1`), shell metacharacter/path-traversal/abs-path checks, workspace-cwd sandbox, output cap, timeout. Banned: `sudo`, `curl`, `bash`, package managers, container tools, kill, etc.
- **Functional auth**: bcrypt pinned to `<4.1` to work around passlib 1.7.4's startup self-check bug. Verified end-to-end: `SHADOW_AUTH_ENABLED=true SHADOW_ADMIN_PASSWORD='example-password' ./start-shadowd.sh` → `POST /auth/login` → bearer token → CLI uses it via `~/.config/shadow/token` (written by `summon --login`).
- **Observability**: `/observers/status` reports per-task health for the 5 background observers (terminal/filesystem/clipboard/window/git). CLI `--observers`.
- **Web dashboard**: PWA HTML now served from `/dashboard` directly on the main daemon (previously required the separate `shadowosd` process).
- **Tests for the new modules**: `test_semantic_memory.py` (13), `test_sentiment.py` (25), `test_council_synthesis.py` (18), `test_native_sensors.py` (15), `test_embedding_index.py` (12), `test_safe_runner.py` (29). 112 new tests.
- **Bug**: pre-existing `test_update_user_blocks_user_id` had a malformed call (`update_user(user_id, user_id=...)` — same arg twice positionally and by name). Fixed by passing the kwarg via `**kwargs`.

### Round 4 (2026-05-16) — daemon offline, full suite green, 4 new abilities

- **Full pytest is now 100% green** — 9603+ tests passing, 0 failures, 16 intentional skips. Got here by fixing ~25 broken tests across the suite:
  - Markov RL non-reentrant lock deadlock in `MarkovBehaviorModel.get_value_estimate` (recursive call inside `with self._lock`) — switched to snapshot-then-recurse.
  - NotificationBus + ShadowPool deadlocks (RLock for cross-call lock holders).
  - `asyncio.get_event_loop()` failures in tests under Python 3.10+ — replaced with `asyncio.run` / `asyncio.new_event_loop()`.
  - Stale mocks in test fixtures (`empathy.get_overall_empathy` → `get_stats`, `journal` → `decision_journal`, `narrative.get_themes` → `get_dominant_themes`, `procedural.get_procedures_by_domain` → `list_procedures`, `oracle.get_active_domains` → `get_domain_states`).
  - Missing imports — `DeviceType`, `LocationType`, `VoiceSTTRequest` weren't imported into `cognition/api_router.py`.
  - Duplicate route `GET /os/journal/stats` defined twice — removed the dup.
  - `_check_redis` rate-limiter only caught a narrow set of exceptions; widened to fail-open on any exception (a broken cache must not take the API down).
  - HolographicMemory `reconstruct()` had an indentation bug where `result = ...` was nested under `else:` then referenced outside it → UnboundLocalError on full-fragment path.
  - Notification title sanitization was too lax (`_sanitize_content` instead of strict `_sanitize`) → `<script>` survived.
  - Mind engine's `think()` only caught httpx-specific errors; a plain Exception from mocked clients propagated and exposed stack traces. Added a catch-all that logs type only.
  - EnvironmentModel `_infer_location` required strict word boundaries → `HomeWiFi` matched nothing. Added substring fallback at lower confidence.
  - `_get_user_dir` sanitization test was checking the wrong invariant.
  - 8 tests with hard-coded "10 abilities" — updated to assert the actual count or `>= 10`.
- **Bounded hypothesis profile** (`max_examples=20`) — global CI profile loaded via `conftest.py`. Cuts full-suite runtime from ≥1 hour to ~5 minutes under `pytest-xdist -n 8`.
- **Tests now run in parallel**: `pytest-xdist` added; full suite finishes in 5 minutes.

### Round 9 (2026-05-17) — observability, dry-run, perf benchmark

- **`GET /abilities/health`** — Per-ability introspection. Combines the static `list_shikigami()` registry with live `ability_metrics` snapshot so each row has count / successes / failures / success_rate / p50 / p95 / avg_latency / last_invoked_ts. Uninvoked abilities show null stats, not zeros, so the caller can distinguish "never invoked" from "0% success".
- **`summon --health`** — CLI flag for the new endpoint. Prints a compact table: `ability  n  ok  rate  p50  p95`. Uninvoked rows show dashes.
- **Council `dry_run` mode** — `POST /shikigami/council` with `dry_run: true` returns the planned ability list without invoking. Useful for previewing what the council would consult before paying the latency cost. Response: `{dry_run, planned_abilities, ability_count, observations_in_context, verdict}`. Verified via metrics snapshot that no ability is actually called.
- **Council perf benchmark** (`tests/test_council_perf.py`) — Three benchmarks: all-21-abilities under 15s (≈11s observed, ~525ms/ability), parsed-observations variant (~7s — ~30% speedup), 270-observation scaling case under 20s. Loose bounds so it doesn't flake on slow CI, with printed timings for visibility.
- **New tests** (~17): `test_abilities_health_endpoint.py` (4), `test_summon_cli_council.py` health suite (+4), `test_council_endpoint.py` dry-run suite (+2), `test_council_perf.py` (3).
- **Full suite**: 9912 passed, 16 skipped, 0 failures, 6 minutes parallel. Daemon remains offline.

### Round 8 (2026-05-17) — contradictions, audit retention, council CLI, code review

- **Cross-ability contradiction detection** (`shikigami/narrative.py:_detect_contradictions`) — Surfaces dissonant pairs that the user might not see on their own: mood rising while compass alignment declines (drifting feel-good), mood falling while alignment rises (right work, wrong cost), constructive self-talk paired with very-negative tide (your inner narrator is kinder than your day), flow state at energy trough (rare, protect it). Wired into the council endpoint response (`narrative.contradictions[]`) and the rendered verdict text.
- **Audit log retention** (`shadowd/ability_audit.py`) — Deletes audit JSONL files older than `retention_days` (default 30). Runs at most once per UTC day, throttled on the write hot path. Configurable per AuditLog instance. Cleanup never raises — broken cleanup must not break the request path.
- **summon_cli --council flag** (`summon/summon_cli.py`) — `summon --council "what's my state?"` fans out to the daemon's `/shikigami/council`. Optional `--ability <name>` (repeatable) limits the fan-out set. Renders contradictions + recommendations cleanly under the verdict.
- **Auto-routing coverage gap closed** — `shadowd/shadow_ai.py` `_ABILITY_MAP` and `ABILITY_KEYWORDS` were stuck at 18 entries — Round 6's three new abilities (lantern, tide, echo) were unreachable through natural-language routing. Both maps now cover all 21 abilities. (omni_sense intentionally has no keywords; it's the catch-all fallback.)
- **Code review pass on `enhanced_abilities.py`** (2702 lines): removed unused imports (`copy`, `dataclass`, `field`); broadened `_llm` error catch to include `httpx.HTTPError` so a transport-level failure falls back to algorithmic synthesis instead of bubbling up. The other six broad `except Exception` blocks are all justified optional-feature fallbacks (sentiment, bias detection, council synthesis, semantic search, safe_runner) — left intentionally.
- **New tests** (~25): `test_narrative.py` contradictions suite (+4), `test_ability_audit.py` retention suite (+5), `test_summon_cli_council.py` (5), `test_audit_endpoint.py` (already 4).
- **Full suite**: 9899 passed, 16 skipped, 0 failures. Daemon remains offline.

### Round 7 (2026-05-17) — narrative deepening, audit retrieval, git-aware Lantern

- **`GET /audit/recent`** — Returns the last N (≤1000) audit entries when `SHADOW_ABILITY_AUDIT=1`. Each entry has a hashed query (sha256 prefix), latency, success, confidence, obs_count. Auth-gated.
- **Council endpoint test now asserts narrative + audit wiring**:
  - `test_narrative_block_present` — confirms `response.narrative` has all of `summary / signals / themes / health_score / recommendations` and verdict text mentions the gauge.
  - `test_council_runs_all_21_abilities` — fans out to every registered ability in 10-at-a-time chunks; every member must return a structured response.
  - `test_audit_logs_council_members_when_enabled` — verifies SHADOW_ABILITY_AUDIT writes through the summon path.
- **Lantern git-repo detection** (`shikigami/lantern.py:_git_repo_root`) — Walks up the path tree (up to 32 levels) looking for `.git/`. When found, returns the basename of the repo root, which beats the heuristic for monorepos and `Documents/projects/...` layouts. Falls back to the existing segment-skip heuristic when no repo ancestor exists. `compute_lantern(use_git=False)` for tests/sandbox.
- **Burnout cross-signal detector in the narrative weaver** — When ≥3 of {falling mood, anxiety/doubt theme rising, alignment < 0.35, mostly-dormant projects with no active ones, harsh self-talk < -0.25} are co-active, emit a priority-1 recommendation titled "🚨 Burnout risk — pause and reassess" naming the specific triggering signals. Same threshold tested both ways: fires on heavy clusters, stays quiet on isolated signals.
- **New tests** (~30): `test_audit_endpoint.py` (4), `test_lantern.py` git suite (+5), `test_narrative.py` burnout suite (+3), `test_council_endpoint.py` extensions (+3).
- **Full suite**: 9885 passed, 16 skipped, 0 failures, 5 minutes parallel. Daemon remains offline.

### Round 6 (2026-05-17) — 21 abilities total, cross-ability narrative, audit log

- **3 NEW abilities (now 21 total)**:
  - **🏮 Lantern** (`shikigami/lantern.py`) — Project state tracker. Maps file_change observations to project roots (heuristic skips `src`/`projects`/`code`/`workspace`/etc.), classifies each as active (≤7d), idle (≤30d), or dormant (>30d), and surfaces revisit candidates: dormant projects with ≥3 prior changes that are worth retiring or reviving before the context evaporates.
  - **🌊 Tide** (`shikigami/tide.py`) — Emotional sentiment trend. Buckets text-bearing observations (notes/messages/clipboard/journal) by ISO week, fits an OLS slope to weekly averages → rising / falling / stable, tracks longest positive/negative streaks, and surfaces best/worst weeks with their dominant emotional words.
  - **🌀 Echo** (`shikigami/echo.py`) — Counterfactual analysis. Identifies recent decision points (DecisionTracker entries + decision-like observations), finds historical decisions that chose differently in similar situations, computes the satisfaction delta, ranks by *regret pressure* (|Δ| × √n_alt) so a single outlier doesn't dominate. Produces past-tense "what would have happened" narratives.
- **Cross-ability narrative weaver** (`shikigami/narrative.py`) — When the council fans out to multiple abilities, `weave()` extracts compact signals from each (~12 across all current abilities), detects recurring themes, scores an overall health gauge (0–1) from tide/mirror/compass/lantern/focus signals, and emits cross-signal recommendations like *"Take a reflective break"* (tide falling + mirror harsh self-talk) or *"Audit dormant projects"* (≥3 dormant + compass<0.4). Wired into `POST /shikigami/council` — the response now includes a `narrative` block with signals, themes, recommendations, and a health gauge.
- **Ability invocation audit log** (`shadowd/ability_audit.py`) — Opt-in via `SHADOW_ABILITY_AUDIT=1`. Writes JSONL to `data/audit/abilities-YYYY-MM-DD.jsonl`, rotates daily. Persists ts / ability / **hashed** query (never raw text) / success / latency / confidence / obs_count / error_type. Wired into `/shikigami/summon` alongside the existing `ability_metrics` recorder. Audit failures never break the request path.
- **Observation pre-parse cache** (`shikigami/parsed_observations.py`) — A list-compatible `ParsedObservations` facade that memoizes timestamp parsing and indexes observations by type. The council endpoint pre-wraps the observation list once and hands it to all members; abilities that aren't aware of the facade still iterate it as a normal list. Verified all 21 abilities accept it transparently.
- **New tests** (~145): `test_lantern.py` (23), `test_tide.py` (16), `test_echo.py` (16), `test_narrative.py` (17), `test_ability_audit.py` (14), `test_parsed_observations.py` (18). All previous count assertions (18 abilities) updated to 21.
- **Full suite**: 9870 passed, 16 skipped, 0 failures, ~7 minutes parallel. Daemon remains offline.

### Round 5 (2026-05-16) — 18 abilities total, full suite stays green

- **4 NEW abilities (now 18 total)**:
  - **🧭 Compass** (`shikigami/compass.py`) — Goal alignment over time. Tracks a `Goal` persistent store; computes alignment from frequency of keyword matches in observations vs the pre-goal baseline; detects rising / declining / dormant trends; honors target deadlines for urgency boost.
  - **🏠 Hearth** (`shikigami/hearth.py`) — Location patterns. Classifies WiFi/GPS/calendar observations into home/work/cafe/public/school/gym; reports time per place, top transitions, hour-of-day mode, and anomalies ("at hour X you're usually home but recently at work").
  - **⏳ Time Witness** (`shikigami/time_witness.py`) — Where the hours actually went. Categorizes the last N days of observations into coding / communication / browsing / meetings / media / system, with delta-vs-prior-window. ASCII bar chart in the output.
  - **🪞 Mirror** (`shikigami/mirror.py`) — Self-reflection from first-person text. Detects 9 inner-state themes (anxiety, motivation, frustration, burnout, growth, doubt, gratitude, stuck, purpose) in clipboard/note/journal observations; reports rising/declining themes and avg self-talk sentiment.
- **Cognitive bias detector** (`shikigami/bias_detector.py`) — surfaces confirmation / recency / sunk-cost / bandwagon biases. Wired into Pattern Sight so each pattern report now includes a "⚠️ COGNITIVE BIAS SIGNALS" section with strength + suggestion.
- **Per-ability metrics** (`shadowd/ability_metrics.py`) — `record(ability, success, duration)` tracked in-memory with a 100-sample rolling window. `/metrics` now emits `shadow_ability_invocations`, `shadow_ability_successes`, `shadow_ability_failures`, `shadow_ability_latency_p95` per ability label. Wired into `/shikigami/summon`.
- **Structured JSON logging** — `SHADOW_LOG_JSON=true` switches the daemon to one-record-per-line JSON. Friendly for log shippers.
- **Auto-routing keywords + ABILITY_MAP updated** for all 18 abilities.
- **Pre-existing tests updated for the wider ability count** — `test_lists_all_known`, `test_list_shikigami`, robustness sweep, integration test.
- **Quality pass**: Reality Forge no-output state now explains how to get artifacts instead of dumping raw LLM text.
- **New tests** (~100): `test_compass.py` (19), `test_hearth.py` (14), `test_time_witness.py` (12), `test_mirror.py` (15), `test_bias_detector.py` (14), `test_ability_metrics.py` (10), `test_all_abilities_integration.py` (21).

### Round 4.5 (2026-05-16) — features added

- **3 new observers** (`shadowd/observers_extra.py`): process activity (`psutil`), browser history (Chromium/Firefox sqlite), and calendar (.ics files). Each opt-in via env vars (`SHADOW_OBSERVE_PROCESSES`, `SHADOW_OBSERVE_BROWSER`, `SHADOW_OBSERVE_CALENDAR`), all best-effort.
- **Natural-language time windows in Perfect Memory** (`shikigami/time_parser.py`): "yesterday", "this week", "last 3 days", "past hour", "since yesterday", word-numbers ("two days"). Wired into `_perfect_memory` ahead of the weekday-name fallback.
- **4 NEW abilities** — README originally promised 10, we shipped 14:
  - **🌀 Habit Tracker**: detects recurring behaviors by (weekday, hour) bucket; tracks streaks, consistency, broken-streak alerts.
  - **⚡ Energy Flow**: circadian energy curve from observation density. Reports current level + 6-hour forecast + ASCII chart of the 24h curve.
  - **🎯 Focus Sight**: finds deep-work blocks (contiguous spans of one activity type) + classifies current state (`in_flow` / `scattered` / `idle`).
  - **🔗 Connection**: per-tracked-person relationship strength scoring (recency + frequency + sentiment + reciprocity). Surfaces rising stars & dormant relationships.
- **Council of Abilities** (`POST /shikigami/council`): fan-out endpoint that consults multiple abilities in parallel for one question, then surfaces the highest-confidence verdict plus brief mentions from each member. Default council: Oracle + Shadow Council + Pattern Sight + Social Radar + Habit Tracker.
- **Auto-routing keywords updated**: new abilities are routable via natural language ("habit", "energy", "focus", "connection") through `shadow_ai.auto_select`.
- **Life Pulse**: zero-state output now surfaces unknown nearby devices instead of going silent.
- **New tests** (~140 lines): `test_time_parser.py`, `test_habit_tracker.py`, `test_observers_extra.py`, `test_new_abilities.py`, `test_council_endpoint.py`. All green.

### Round 3 (2026-05-16) — daemon was taken offline for this round

- **MCP server live**: `pip install mcp` (added to requirements). `shadow-mcp --transport streamable-http --port 8779` now binds correctly and answers the `initialize` handshake with proper capabilities. Both `shadowd/mcp_server.py` and `shadowd/mcp_server_enhanced.py` were patched to drop the `version=` kwarg that mcp 1.x removed from `FastMCP.__init__` — they introspect the signature and pass it conditionally.
- **`shadowctl` and `shadow-mcp` console scripts**: added to `pyproject.toml` so `pip install -e .` exposes them in PATH. `shadowctl` is an alias for `summon`.
- **Minimal v1-focused dashboard**: `web/shadow.html` — clean static HTML hitting only the working endpoints (`/status`, `/shikigami`, `/observers/status`, `/insights/recent`). Live WebSocket updates. `/dashboard` now serves this; the old v2-oriented PWA is at `/dashboard/full`.
- **Anticipatory bridge**: proactive engine now also calls cognition v2's `AnticipatoryEngine.predict()` and emits `anticipation` insights with confidence > 0.65. Bridges v1 ↔ v2.
- **`/config` endpoint**: sanitized snapshot of env vars, feature flags (LLM configured, MCP installed, auth on, proactive on), and paths. No credentials.
- **Defensive ability execution**: `/shikigami/summon` wraps `registry.execute` in try/except so a buggy ability returns a structured error response instead of crashing the daemon.
- **Better insufficient-data responses**: replaced "Continue monitoring X for clearer patterns as more data accumulates" with specific actionable hints (e.g. "Council needs more tracked decisions. POST /track/decision …").
- **Bugs fixed**:
  - **Markov RL deadlock**: `MarkovBehaviorModel.get_value_estimate` recursively called itself while holding a non-reentrant lock — any value-estimate query hung forever. Now snapshots state under the lock and recurses outside it.
  - **Intent Seer crash on `type: None` observations**: `' → '.join()` blew up when an observation explicitly had `type=None`. Coerced to `str`.
  - **EnvironmentModel can't resolve concatenated SSIDs**: `_infer_location` required strict word boundaries, so `HomeWiFi` came out as `unknown`. Added a substring-match fallback at 0.65 confidence.
  - **Pre-existing test bug**: `_predict_activities` was renamed to `_predict_activities_unlocked` but the test still called the old name. Made the test tolerant of either.
- **Robustness test suite**: `tests/test_abilities_robustness.py` — 107 tests covering every ability with empty contexts, malformed questions (empty, whitespace, 5000-char, unicode, emoji, shell metas, JSON-injection), concurrent invocations, weird observation shapes (None type, garbage type, missing timestamp). All abilities must succeed or fail gracefully — never throw.
- **`pytest-timeout` added** to requirements: required to protect against future deadlock regressions in the RL/Markov code.

---

## Architecture

```
~/digital-shadow/
├── shadowd/           # v1 daemon (the canonical entry point)
│   ├── shadowd.py     # FastAPI app, 80+ routes
│   ├── mind.py        # LLM client (Ollama Cloud), retry/sanitization
│   └── cognition/     # v2 platform — 91 modules, 642 /os/* routes (see below)
├── shikigami/         # the 10 abilities
│   ├── enhanced_abilities.py  ← canonical registry (used by daemon)
│   ├── abilities.py           ← helper algos (Monte Carlo, fingerprint, …)
│   ├── superhuman.py          ← more helper algos (Bayes, Markov, Kalman, …)
│   ├── shikigami.py           ← LEGACY v1 registry (deprecated; tests only)
│   ├── semantic_memory.py     ← TF-IDF for Perfect Memory
│   ├── embedding_index.py     ← Ollama embeddings + TF-IDF fallback
│   ├── sentiment.py           ← lexicon sentiment + manipulation detection
│   ├── council_synthesis.py   ← Shadow Council action-category synthesis
│   └── safe_runner.py         ← Reality Forge allowlist command runner
├── shadowd/
│   ├── shadowd.py             ← FastAPI app, observers, abilities
│   ├── proactive_engine.py    ← background insight scanner
│   └── observers.py           ← terminal/fs/clipboard/window/git watchers
├── phone/             # phone-side scanners + REST API
│   ├── bluetooth_scanner.py   ← real bluetoothctl discovery
│   ├── wifi_scanner.py        ← real nmcli/iwlist scan
│   ├── phone_api.py           ← richer v1 API used by start-all.sh
│   └── phone_api_v2.py        ← slimmer v2 used by tests
├── sensors/
│   └── native_sensors.py      ← audio/battery/load probes
├── web/
│   ├── shadow_pwa.html        ← PWA dashboard (served at /dashboard)
│   ├── manifest.json
│   └── sw.js
└── summon/
    └── summon_cli.py          ← CLI; talks to /shikigami/summon, /insights, etc.
```

## API endpoint cheat sheet

| Endpoint | Purpose |
|---|---|
| `GET /` | Daemon name + link to /dashboard |
| `GET /dashboard` | PWA dashboard (HTML) |
| `GET /status` | Daemon status: observations, abilities, decisions, people |
| `GET /shikigami` | List all 10 abilities |
| `POST /shikigami/summon` | Invoke an ability (with `shikigami:` field) |
| `GET /observations?limit=N` | Recent observations |
| `POST /observe` | Submit a new observation |
| `POST /track/person` | Track a person by their devices |
| `POST /track/decision` | Record a decision for Council learning |
| `GET /insights/recent?limit=N` | Recent proactive insights (manipulation, council, anomaly) |
| `GET /observers/status` | Per-observer task health |
| `POST /auth/login` | JWT login when `SHADOW_AUTH_ENABLED=true` |
| `WS  /ws` | Real-time event stream (insights, observations) |
| `/os/*` (642 routes) | Cognition v2 layer (anticipatory, anomaly, oracle, …) |

---

## Cognition v2 (`shadowd/cognition/`)

91 modules implementing a parallel "cognitive platform" with modules for
ShadowGraph, DarkMemory, NeuralMesh, PersonalityEngine, AnticipatoryEngine,
EscalationManager, Oracle, AnomalyDetector, etc.

**Mounted on the same daemon as `/os/*` routes** — currently 642 of them.
You can call them directly. Examples that return real data:

```bash
curl -s -X POST http://127.0.0.1:8765/os/anticipate/predict -d '{}'   # generates a real signal
curl -s -X POST http://127.0.0.1:8765/os/act -H 'Content-Type: application/json' \
    -d '{"action": "summarize this", "user_id": "demo"}'              # 4-agent mission
curl -s http://127.0.0.1:8765/os/anomaly/stats
```

**Not running as an autonomous daemon by default.** `shadowd/cognition/shadowosd.py`
is a separate process you'd run alongside `shadowd.shadowd` to drive
self-improvement / event loops. The HTTP surface works without it.

**Pruning recommendation**: don't. The code isn't dead — it just isn't fed.
Bringing v2 fully online (observation pipeline → cognition modules → outputs
fed back) is a multi-week project.

---

## Known limitations

- **Single-user**: stores everything in `data/*.json` under `$HOME`. No multi-tenant model.
- **No GPS on Linux laptops**: `sensors/sensor_hub.py` tries `gpsd` (optional), then IP geolocation. Real GPS requires the Termux phone integration (`phone/android/`).
- **`/shikigami/summon` always returns 200**: even if no observations are tracked. Output is sparse but the call succeeds.
- **LLM-backed abilities (Social Radar, Shadow Council fallback, Reality Forge) need `OLLAMA_API_KEY`**: configured in `.env`. Without it those parts of responses degrade to "no LLM available".
- **Reality Forge does not auto-execute commands**: it parses them out of LLM output but only logs; you run manually if trusted.
- **Tests**: 3 files needed `pip install hypothesis` to collect (160 tests). Done in venv.

---

## What's deferred / known scope creep

- AGENTS.md (250 KB) reads as a future-tense design doc. It does not match the
  current daemon's actual endpoint list. Treat as wish list, not docs.
- Multiple "session" entries in user memory describe past bugfix sessions; they
  are no longer load-bearing.
- The empty `=` file is gone.
- `phone/phone_api_v2_old.py` was orphaned — gone.
