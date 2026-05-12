# AgentMesh — CLAUDE.md

Project-level instructions and context for Claude Code sessions on this repo.

---

## What This Project Is

AgentMesh is a peer-to-peer multi-agent discovery and routing framework built as a solo portfolio project for new-grad AI/ML engineer job applications. Agents register capabilities with a central registry (phone book, not orchestrator), discover each other at runtime via semantic search, negotiate task contracts over WebSocket, and earn ELO-style trust scores that influence future routing.

**GitHub:** https://github.com/arshadvani3/AgentMesh  
**Author:** Arsh Advani (arshadvani3)  
**Status:** Phase 10 complete — 126 tests passing, ruff clean, mypy clean, CI green.

---

## Architecture

```
Registry (port 8000)          ← FastAPI, agent phone-book + routing
ResearchAgent (port 9001)     ← LangGraph orchestrator
DataAgent (port 9002)         ← pandas/numpy CSV analysis
CodeAgent (port 9003)         ← code generation + sandboxed execution
WriterAgent (port 9004)       ← report synthesis
Dashboard (port 3000)         ← optional React UI
```

Agents never talk to each other directly — all discovery/routing goes through the registry.

### Routing formula
```
score = (semantic_match × 0.35) + (trust_score × 0.35) + (availability × 0.15) + (cost_factor × 0.15)
```
- `semantic_match`: cosine similarity via sentence-transformers (all-MiniLM-L6-v2)
- `trust_score`: ELO-style update: `clip(old + 0.1 × (quality − old), 0, 1)`, starts at 0.5
- `availability`: `max(0.0, 1.0 − active_tasks / max_concurrent_tasks)`, live via heartbeat WS
- `cost_factor`: `1.0 − min(cost / max_cost, 1.0)`, agents over budget excluded entirely

### Circuit breaker
- 3 consecutive failures → `status="degraded"`, scored at 30% availability (not excluded)
- Auto-recovery after 60 s → `status="healthy"`
- State tracked in registry: `_failure_streaks`, `_degraded_since`

### Counter-proposal negotiation
- Agent at capacity returns `COUNTERED` with proposed deadline
- SDK `delegate()` retries with proposed deadline if ≤ 1.5× original, else tries next candidate

---

## Key Files

| File | Purpose |
|------|---------|
| `mesh/models.py` | Pydantic v2 schemas — protocol source of truth |
| `mesh/registry.py` | FastAPI: register, discover, trust, circuit breaker, WS heartbeat, traces, memory endpoints |
| `mesh/router.py` | 4-factor composite routing with semantic search |
| `mesh/db.py` | asyncpg + full in-memory fallback (no Docker needed for tests) |
| `mesh/memory.py` | AgentMemory — Redis-backed session state with in-memory fallback, TTL=1hr |
| `sdk/agent.py` | MeshAgent base class: heartbeat, WS task server, discover(), delegate(), counter-proposal retry |
| `agents/research_agent.py` | LangGraph StateGraph: plan→discover→delegate→synthesize, writes memory at each node |
| `agents/data_agent.py` | Real pandas/numpy CSV computation; passes actual stats to LLM for narrative |
| `agents/code_agent.py` | Code generation via Groq + sandboxed subprocess execution (opt-in, `execute: true`) |
| `agents/writer_agent.py` | Report synthesis via Groq |
| `data/startups.csv` | 100-row canonical demo dataset (company, sector, amount_usd, year, country, stage) |
| `demo.py` | One-command demo: registry + 4 agents + research task + Rich UI |
| `tests/` | 126 tests across all layers |

---

## Phase History

| Phase | What was built |
|-------|---------------|
| 1–8 (initial) | Core registry, routing, trust scoring, circuit breaker, negotiation, LangGraph pipeline, JWT auth, CI/CD |
| 9 | Dynamic capability announcements, JWT auth between agents, latency tracking, robust JSON parsing |
| 10 | Real DataAgent (pandas CSV stats), CodeAgent sandbox execution, AgentMemory (Redis), new demo, 126 tests |

---

## DataAgent (Phase 10)

`agents/data_agent.py` — real computation, not LLM guessing.

- `compute_csv_stats(content)` → tries pandas first, falls back to numpy+stdlib csv
- Returns: `rows, cols, columns, numeric_stats (mean/std/min/max/p25/p50/p75), missing_values, categorical_summaries, engine`
- `_load_csv(input_data)` → async, handles `csv_path` (local file) and `csv_url` (HTTP fetch)
- When CSV provided: computes real stats, injects into LLM prompt as ground truth. LLM narrates, doesn't invent.
- Fallback: if no CSV, falls through to LLM-only analysis (old behaviour preserved)

---

## CodeAgent (Phase 10)

`agents/code_agent.py` — generates code and optionally runs it.

- `_run_in_sandbox(code)` → sync function:
  1. `ast.parse(code)` — syntax check, fast fail before subprocess
  2. `subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)`
  3. Returns `{executed, stdout (capped 2000), stderr (capped 1000), exit_code, error}`
- 4 cases: success, syntax error (pre-subprocess), runtime error, timeout (killed after 5s)
- Execution is **opt-in**: `input_data.get("execute", False)` — default False
- Why subprocess not exec(): isolated process, killable, captures stdout cleanly

---

## AgentMemory (Phase 10)

`mesh/memory.py` — cross-agent session state.

```python
memory = AgentMemory()              # Redis if REDIS_URL set, else in-memory
await memory.set(sid, "plan", {...})
await memory.get(sid, "plan")
await memory.get_session(sid)       # all keys for session
await memory.clear(sid)
await memory.list_sessions()
get_memory()                        # module-level singleton
```

- Redis keys: `agentmesh:session:{session_id}:{key}`
- In-memory fallback: `dict[session_id, dict[key, (value, expires_at)]]`
- TTL default 3600 s — sessions expire automatically
- ResearchAgent writes: `plan` after planning, `result_{subtask}` after each delegation, `final_report` at end
- Registry endpoints: `GET /memory/{session_id}`, `GET /memory`

---

## Running the Project

```bash
# Install deps
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run demo (starts registry + 4 agents, runs funding analysis task)
python3 demo.py

# Run against custom CSV
python3 demo.py --csv-path /path/to/data.csv

# Lint / type check
ruff check mesh/ sdk/ agents/ tests/
mypy mesh/ sdk/ agents/
```

---

## Test Suite

```
tests/test_models.py          — Pydantic schema validation
tests/test_registry.py        — FastAPI endpoint tests
tests/test_router.py          — routing score logic
tests/test_trust.py           — ELO trust update
tests/test_circuit_breaker.py — failure/recovery state machine
tests/test_negotiation.py     — counter-proposal flow
tests/test_e2e.py             — full registry + agent integration
tests/test_memory.py          — AgentMemory set/get/TTL/clear/sessions (15 tests)
tests/test_data_agent.py      — numpy path, pandas path, real CSV, async _load_csv (10 tests)
tests/test_code_agent.py      — print capture, syntax error, timeout, isolation (10 tests)
```

**Total: 126 tests. All async memory tests use `@pytest.mark.asyncio` + `async def`.**

---

## Conventions and Constraints

- **Pydantic v2** throughout — use `model_validate`, not `parse_obj`
- **asyncpg + in-memory fallback** in `mesh/db.py` — tests never need Docker/Postgres
- **Redis + in-memory fallback** in `mesh/memory.py` — same pattern
- **No comments unless non-obvious** — well-named identifiers are the documentation
- **ruff** for lint (line length 100), **mypy** for types — both must be clean before commit
- Python version: 3.13 (note: `asyncio.get_event_loop()` raises outside async context — always use `@pytest.mark.asyncio`)
- Registry is discovery-only — it never orchestrates or owns business logic
- All data models in `mesh/models.py` — never duplicate Pydantic schemas elsewhere

---

## Honest Limitations

- Not benchmarked at scale (100+ agents)
- No auth between agents (assumes trusted internal network)
- Redis pub/sub not implemented (only key-value memory is used)
- Agents use simple Groq Llama 3.3 70B prompts — not production LLM pipelines
- Dashboard exists but is minimal
