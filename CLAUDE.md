# AgentMesh — CLAUDE.md

Project-level instructions and context for Claude Code sessions on this repo.

---

## What This Project Is

AgentMesh is a peer-to-peer multi-agent discovery and routing framework built as a solo portfolio project for new-grad AI/ML engineer job applications. Agents register capabilities with a central registry (phone book, not orchestrator), discover each other at runtime via semantic search, negotiate task contracts over WebSocket, and earn EMA-based trust scores that influence future routing.

**GitHub:** https://github.com/arshadvani3/AgentMesh  
**Author:** Arsh Advani (arshadvani3)  
**Status:** Phase 11 complete — 164 tests passing, ruff clean, mypy clean, CI green.

---

## Architecture

```
Registry (port 8000)          ← FastAPI, agent phone-book + routing
ResearchAgent (port 9001)     ← LangGraph orchestrator
DataAgent (port 9002)         ← pandas/numpy CSV analysis
CodeAgent (port 9003)         ← code generation + subprocess execution
WriterAgent (port 9004)       ← report synthesis
WebSearchAgent (port 9005)    ← MCP-backed web search (brave-search or mock)
Dashboard (port 3000)         ← React 18 + force-directed graph, live data
```

Agents never talk to each other directly — all discovery/routing goes through the registry.

### Routing formula
```
score = (semantic_match × 0.35) + (trust_score × 0.35) + (availability × 0.15) + (cost_factor × 0.15)
```
- `semantic_match`: cosine similarity via sentence-transformers (all-MiniLM-L6-v2), query encoded once per `match()` call, run in `loop.run_in_executor` to avoid blocking the event loop
- `trust_score`: EMA update: `clip(old + 0.1 × (quality − old), 0, 1)`, starts at 0.5. Quality scored by `OutputEvaluator` (not hardcoded).
- `availability`: `max(0.0, 1.0 − active_tasks / max_concurrent_tasks)`, live via heartbeat WS
- `cost_factor`: `1.0 − min(cost / max_cost, 1.0)`, agents over budget excluded entirely

### Circuit breaker
- 3 consecutive failures → `status=AgentStatus.DEGRADED`, scored at 30% availability (not excluded)
- Auto-recovery after 60 s → `status=AgentStatus.HEALTHY`
- State in `_failure_streaks`, `_degraded_since`, guarded by `_circuit_lock` (asyncio.Lock)

### Counter-proposal negotiation
- Agent at capacity returns `COUNTERED` with proposed deadline
- SDK `delegate()` retries with proposed deadline if ≤ 1.5× original, else tries next candidate

### Auth
- JWT via `AGENT_SECRET` env var. If unset → dev mode ("anonymous"), all endpoints open.
- `POST /agents/register` and `DELETE /agents/{id}` validate token matches the agent being modified.
- `POST /auth/token` takes `{agent_id, secret}` as POST body (not query params — avoids log exposure).
- `GET /memory/{session_id}` requires auth — session data may contain sensitive workflow state.

---

## Key Files

| File | Purpose |
|------|---------|
| `mesh/models.py` | Pydantic v2 schemas — protocol source of truth. `AgentStatus`, `TaskStatus`, `TraceEventType` are all StrEnums. |
| `mesh/registry.py` | FastAPI: register, discover, trust, circuit breaker, WS heartbeat, traces, memory, stats, trust history endpoints |
| `mesh/router.py` | 4-factor composite routing; `model.encode()` runs in thread executor |
| `mesh/db.py` | asyncpg + full in-memory fallback; singleton guarded by `asyncio.Lock`; traces capped at `deque(maxlen=10_000)` |
| `mesh/memory.py` | AgentMemory — Redis-backed session state with in-memory fallback, TTL=1hr; `list_sessions()` uses Redis set index (no `KEYS *`) |
| `mesh/evaluator.py` | OutputEvaluator — scores agent output 0.0–1.0; deterministic for CSV/code, LLM-as-judge for reports |
| `mesh/mcp_client.py` | MCPToolClient — wraps MCP server subprocess connections; mock fallback when server not configured |
| `sdk/agent.py` | MeshAgent base class: heartbeat, WS task server, discover(), delegate(), counter-proposal retry, MCP tool access. All `httpx.AsyncClient` calls use `timeout=10.0`. |
| `agents/research_agent.py` | LangGraph StateGraph: plan→discover→delegate→synthesize, writes memory at each node |
| `agents/data_agent.py` | Real pandas/numpy CSV computation; `csv_url` SSRF-protected (blocks RFC-1918); `csv_path` restricted to `CSV_SAFE_DIR` |
| `agents/code_agent.py` | Code generation via Groq + subprocess execution (opt-in, `execute: true`); uses `get_running_loop()` |
| `agents/writer_agent.py` | Report synthesis via Groq |
| `agents/web_search_agent.py` | Web search via brave-search MCP server; graceful mock fallback |
| `data/startups.csv` | 100-row canonical demo dataset (company, sector, amount_usd, year, country, stage) |
| `demo.py` | One-command demo: registry + 4 agents + research task + Rich UI |
| `examples/` | Standalone runnable examples: `hello_agent.py`, `translation_agent.py`, `two_agent_pipeline.py` |
| `tests/` | 164 tests across all layers |

---

## Phase History

| Phase | What was built |
|-------|---------------|
| 1–8 (initial) | Core registry, routing, trust scoring, circuit breaker, negotiation, LangGraph pipeline, JWT auth, CI/CD |
| 9 | Dynamic capability announcements, JWT auth between agents, latency tracking, robust JSON parsing |
| 10 | Real DataAgent (pandas CSV stats), CodeAgent sandbox execution, AgentMemory (Redis), new demo, 126 tests |
| 11 | MCP tool integration, OutputEvaluator (real trust scoring), live dashboard (MemoryPanel, trust history), PyPI metadata + examples |
| 11+ (security/quality pass) | SSRF + path traversal protection, auth on register/deregister/memory, asyncio.Lock singletons, StrEnums, executor for embeddings, deque trace cap, httpx timeouts, docker hardening |

---

## DataAgent

`agents/data_agent.py` — real computation, not LLM guessing.

- `compute_csv_stats(content)` → tries pandas first, falls back to numpy+stdlib csv
- Returns: `rows, cols, columns, numeric_stats (mean/std/min/max/p25/p50/p75), missing_values, categorical_summaries, engine`
- `_load_csv(input_data)` → async; `csv_path` restricted to `CSV_SAFE_DIR` env var (default `/data/uploads`); `csv_url` blocks private/link-local IPs via `_is_ssrf_url()`
- When CSV provided: computes real stats, injects into LLM prompt as ground truth. LLM narrates, doesn't invent.

---

## CodeAgent

`agents/code_agent.py` — generates code and optionally runs it.

- `_run_in_sandbox(code)` → process isolation only (same OS user — not a true sandbox):
  1. `ast.parse(code)` — syntax check, fast fail
  2. `subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)`
  3. Returns `{executed, stdout (capped 2000), stderr (capped 1000), exit_code, error}`
- Execution is **opt-in**: `input_data.get("execute", False)` — default False
- Uses `asyncio.get_running_loop().run_in_executor(None, _run_in_sandbox, code)`

---

## OutputEvaluator (Phase 11)

`mesh/evaluator.py` — replaces hardcoded `quality=0.8`.

- `analyze_csv`: deterministic grounding check — verifies reported numbers against computed stats (±10% tolerance)
- `fetch_code`: checks `exit_code == 0` from execution result
- `web_search`: checks non-empty content in results
- `write_report` / unknown: LLM-as-judge via Groq (returns 0.5 as fallback if API key not set)
- Score feeds directly into EMA trust update in `sdk/agent.py`

---

## MCPToolClient (Phase 11)

`mesh/mcp_client.py` — wraps MCP server connections.

- Config from `~/.agentmesh/mcp_servers.json` (`{command, args, env}` per server name)
- Lazy connection — only connects when `call_tool()` is first invoked
- Mock fallback when server not configured — returns plausible stubs for search/filesystem
- `MeshAgent.call_mcp_tool(server, tool, args)` is the public API

---

## AgentMemory

`mesh/memory.py` — cross-agent session state.

```python
memory = AgentMemory()              # Redis if REDIS_URL set, else in-memory
await memory.set(sid, "plan", {...})
await memory.get(sid, "plan")
await memory.get_session(sid)       # all keys for session
await memory.clear(sid)
await memory.list_sessions()        # uses Redis SMEMBERS on session index set, not KEYS *
get_memory()                        # module-level singleton (threading.Lock guarded)
```

- Redis keys: `agentmesh:session:{session_id}:{key}`; session IDs tracked in `agentmesh:session_index` set
- In-memory fallback: `dict[session_id, dict[key, (value, expires_at)]]`
- TTL default 3600 s

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
tests/test_trust.py           — EMA trust update
tests/test_circuit_breaker.py — failure/recovery state machine
tests/test_negotiation.py     — counter-proposal flow
tests/test_e2e.py             — full registry + agent integration
tests/test_auth.py            — JWT auth, token endpoint, protected endpoints
tests/test_memory.py          — AgentMemory set/get/TTL/clear/sessions
tests/test_data_agent.py      — numpy path, pandas path, real CSV, SSRF/path protection
tests/test_code_agent.py      — print capture, syntax error, timeout, isolation
tests/test_evaluator.py       — OutputEvaluator per-capability scoring
tests/test_mcp_client.py      — MCPToolClient tool listing, calling, mock fallback
```

**Total: 164 tests. All async tests use `@pytest.mark.asyncio` + `async def`.**

---

## Conventions and Constraints

- **Pydantic v2** throughout — use `model_validate`, not `parse_obj`
- **asyncpg + in-memory fallback** in `mesh/db.py` — tests never need Docker/Postgres
- **Redis + in-memory fallback** in `mesh/memory.py` — same pattern
- **No comments unless non-obvious** — well-named identifiers are the documentation
- **ruff** for lint (line length 100, `datetime.UTC` alias), **mypy** for types — both must be clean before commit
- Python version: 3.13 — use `asyncio.get_running_loop()` not `get_event_loop()`; use `datetime.now(UTC)` not `datetime.utcnow()`
- Registry is discovery-only — it never orchestrates or owns business logic
- All data models in `mesh/models.py` — never duplicate Pydantic schemas elsewhere
- Status fields use `AgentStatus` StrEnum — never bare strings `"healthy"/"offline"/"degraded"`
- `TraceEvent.event_type` uses `TraceEventType` StrEnum — rejects unknown values at Pydantic boundary

---

## Security Model

- **Auth is opt-in**: set `AGENT_SECRET` env var to enable JWT enforcement. Without it, all endpoints accept anonymous callers (dev mode).
- **Registration ownership**: token's `sub` must match `manifest.agent_id` — prevents spoofed registrations.
- **Trust update**: protected endpoint; caller identity validated by token.
- **csv_path**: restricted to `CSV_SAFE_DIR` (default `/data/uploads`) — prevents path traversal.
- **csv_url**: blocks RFC-1918, loopback, link-local IPs — prevents SSRF against internal services.
- **CodeAgent subprocess**: process isolation only — same OS user, full filesystem access. Not a true sandbox. Wrap with Docker/bubblewrap before enabling `execute=true` on untrusted input.
- **`/memory/{session_id}`**: requires auth — session data contains intermediate task results.

---

## Honest Limitations

- Not benchmarked at scale (100+ agents)
- Embedding index is in-process — registry restart loses index, requires agent re-registration
- Module-level runtime state (`_active_task_counts`, `_circuit_lock`, etc.) is not persisted — lost on restart
- Redis pub/sub not implemented (only key-value memory is used)
- Agents use Groq Llama 3.3 70B — not production LLM pipelines
- CodeAgent subprocess is not a true OS-level sandbox
- No request deduplication — task retries can cause double-execution
