# AgentMesh

**A peer-to-peer protocol for AI agent discovery, negotiation, and reputation.**

[![CI](https://github.com/arshadvani3/AgentMesh/actions/workflows/ci.yml/badge.svg)](https://github.com/arshadvani3/AgentMesh/actions)
[![PyPI](https://img.shields.io/pypi/v/agentmesh-proto.svg)](https://pypi.org/project/agentmesh-proto/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **See it in action →** [SHOWCASE.md](SHOWCASE.md) — 9-agent incident response demo with screenshots and analysis

---

## What AgentMesh Is

AgentMesh is a **framework for connecting AI agents at runtime** — not a specific pipeline or LLM tool.

It has two parts:

**The core framework** (`mesh/` + `sdk/`) — a registry, a router, and a base class. Agents register their capabilities, discover each other via semantic search, negotiate tasks over WebSocket, and earn trust scores that influence future routing. The framework has no opinion about what your agents do or which LLM they use.

**Bundled example agents** (`agents/`) — five agents (research, data analysis, code generation, report writing, web search) built with Groq + LangChain to demonstrate a complete working mesh. These are demos. You don't need them to use AgentMesh, and you don't need Groq or LangChain to write your own agents.

---

## Why AgentMesh

Most multi-agent frameworks hardwire agent relationships at build time. You write a graph, a crew, or a pipeline — and it's fixed. Adding a new agent means editing the orchestrator. A flaky agent silently degrades the whole pipeline. There's no way to ask "find the best available agent for this task right now."

AgentMesh treats agents like services on a network:

- Any agent can **join or leave** the mesh at runtime
- Tasks are routed to the **best available** agent by capability, trust, load, and cost
- Agents that fail consistently are **automatically deprioritized** via circuit breaking
- Overloaded agents can **counter-propose** a later deadline instead of failing silently
- Quality is **measured, not assumed** — an OutputEvaluator scores every result and feeds real scores back into trust

The registry is a phone book, not an orchestrator. Agents talk directly to each other.

---

## How the Protocol Works

### 1. Register
An agent joins the mesh by posting a capability manifest:

```json
{
  "agent_id": "agent-data-7f3a",
  "name": "Data Agent",
  "endpoint": "ws://localhost:9002",
  "max_concurrent_tasks": 3,
  "capabilities": [{
    "name": "analyze_csv",
    "description": "Statistical analysis of CSV or tabular data",
    "avg_latency_ms": 8000,
    "cost_per_call_usd": 0.002
  }],
  "tags": ["data", "analysis", "csv"]
}
```

### 2. Discover
Any agent queries the mesh with a natural-language description or an exact name:

```python
agents = await self.discover(
    description="analyze and summarize tabular data",
    min_trust=0.4,
    max_cost_usd=0.005,
)
```

The router ranks candidates by a composite score with **dynamic weights** — the weights shift based on what the caller signals:

```
score = (semantic_match × w_match)   # exact name hit = 1.0; else cosine similarity
      + (trust_score    × w_trust)   # confidence-weighted: new agents start humble
      + (availability   × w_avail)   # based on current task load
      + (cost_factor    × w_cost)    # cheaper = higher score within budget
```

Base weights: `match=0.40  trust=0.30  availability=0.15  cost=0.15`

Caller signals adjust automatically:
- `max_latency_ms` set → availability +0.10, trust -0.10 (latency-sensitive)
- `max_cost_usd` set  → cost +0.10, trust -0.10 (budget-sensitive)

Tags are a **soft boost**, not a hard filter — partial overlap still contributes to score.

Degraded agents (circuit open) are included but scored at 30% availability.

Results are **diversity re-ranked**: a second agent on the same hostname must score 0.15 higher than a fresh-host candidate to be selected, spreading load across distinct hosts.

### 3. Negotiate
The requester opens a WebSocket to the chosen agent and sends a `task.request`. The agent responds with one of three outcomes:

| Response | Meaning |
|---|---|
| `accepted` | Agent takes the task, returns estimated latency |
| `rejected` | Agent can't do it (wrong capability, etc.) |
| `countered` | Agent is at capacity — proposes a later deadline |

On `countered`, the SDK automatically retries with the proposed deadline if it's within 1.5× the original. Otherwise it falls back to the next discovery candidate.

### 4. Execute
The agent runs the task using whatever it has — an LLM, a database, an MCP tool, or a subprocess. Result comes back as a structured `TaskResult`.

### 5. Trust Update
After every task, the requester evaluates the output and reports a quality score. Trust updates via an ELO-style algorithm:

```python
new_score = old_score + 0.1 × (actual_quality − expected_quality)
```

Scores are bounded to `[0.0, 1.0]`. After **3 consecutive failures**, the circuit opens and the agent moves to `degraded` status. It auto-recovers after 60 seconds (half-open probe).

**Quality is measured, not assumed.** The `OutputEvaluator` scores each result deterministically where possible:
- `analyze_csv` — cross-checks reported numbers against actual pandas/numpy computed stats (±10% tolerance)
- `fetch_code` — checks subprocess exit code (`0` = correct, syntax error = `0.0`)
- `web_search` — checks non-empty result content
- `write_report` — LLM-as-judge via Groq (generic rubric, returns `0.5` if API unavailable)

This closes the feedback loop: better agents accumulate trust over time, poor ones decline and get routed around.

---

## Building Your Own Agent

Subclass `MeshAgent`, decorate methods with `@capability`, call `start()`. No LLM required — your agent can do anything.

```python
from sdk.agent import MeshAgent, capability
import asyncio, os

class TranslationAgent(MeshAgent):
    @capability(
        name="translate_text",
        description="Translates text between languages. Supports 50+ languages.",
        input_schema={
            "type": "object",
            "properties": {
                "text":        {"type": "string"},
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
        avg_latency_ms=1200,
        cost_per_call_usd=0.001,
    )
    async def translate_text(self, input_data: dict) -> dict:
        # call any model, API, or database here
        return {"translated": "...", "detected_source": "en"}


async def main():
    agent = TranslationAgent(
        name="Translation Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8080"),
        ws_port=9010,
        max_concurrent_tasks=5,
        tags=["translation", "nlp", "language"],
    )
    await agent.start()

asyncio.run(main())
```

Once running, any other agent on the mesh can find and use it:

```python
agents = await self.discover(description="translate text between languages")
result = await self.delegate("translate_text", {
    "text": "Hello, world!",
    "target_lang": "es",
}, target=agents[0])
print(result.output["translated"])  # "¡Hola, mundo!"
```

More examples in [`examples/`](examples/) — including a minimal 15-line hello agent and a 2-agent pipeline without LangGraph.

### Using MCP Tools

Agents can call MCP tools directly. Configure servers in `~/.agentmesh/mcp_servers.json`:

```json
{
  "brave-search": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
    "env": {"BRAVE_API_KEY": "your-key"}
  }
}
```

Then call from any agent:

```python
result = await self.call_mcp_tool("brave-search", "brave_web_search", {"query": "..."})
```

A mock fallback activates automatically when the server isn't configured.

---

## Try the Demo

The bundled demo starts the registry and launches 5 example agents (Research, Data, Code, Writer, WebSearch) backed by Groq Llama 3.3 70B. It runs a full research pipeline against the included startup funding dataset and prints the final report.

**Requirements:** Python 3.11+, a free [Groq API key](https://console.groq.com)

```bash
# Clone and install (includes example agent deps)
git clone https://github.com/arshadvani3/AgentMesh
cd AgentMesh
pip install -e ".[all,dev]"

# Configure
cp .env.example .env
# Set GROQ_API_KEY=gsk_...

# Run
python3 demo.py
```

**Custom query:**
```bash
python3 demo.py --query "Compare vector databases for RAG: Pinecone vs Weaviate vs Chroma"
```

**Custom dataset:**
```bash
python3 demo.py --csv-path /path/to/your/data.csv --query "Analyze the trends in this dataset"
```

**Install options:**

| Command | What you get |
|---|---|
| `pip install agentmesh-proto` | Core framework only — registry + SDK, no LLM deps |
| `pip install "agentmesh-proto[agents]"` | + LangChain, LangGraph, Groq, Pandas (needed for the bundled example agents) |
| `pip install "agentmesh-proto[database]"` | + asyncpg for PostgreSQL |
| `pip install "agentmesh-proto[cache]"` | + Redis for session memory |
| `pip install "agentmesh-proto[mcp]"` | + MCP tool client |
| `pip install "agentmesh-proto[all]"` | Everything above |

> **Note:** `[agents]` is only needed to run the bundled example agents. Your own agents only need the core install.

---

## Example Use Cases

### Autonomous Research Pipeline
A research agent receives a complex query, decomposes it into subtasks, discovers specialists dynamically, and synthesizes results — all without knowing in advance which agents exist.

```
Query: "Competitive analysis of vector databases"
  → discovers web_search_agent    → delegates retrieval
  → discovers data_analysis_agent → delegates metric comparison
  → discovers report_writer_agent → delegates synthesis
  → merges results into final report
```

New specialists can join the mesh at any time and get discovered automatically on the next query.

### Code Review Mesh
Push a PR → trigger a fleet of independent review agents:
- `security_scanner` — OWASP vulnerability checks
- `style_checker` — linting and formatting
- `test_coverage` — gap analysis
- `documentation` — missing docstring detection

Each agent is independent. Add or remove reviewers by registering/deregistering. No orchestrator rewrite needed.

### Budget-Constrained Routing
Multiple LLM backends registered with different cost profiles:

```python
agents = await self.discover(
    description="generate structured JSON from unstructured text",
    max_cost_usd=0.005,
    min_trust=0.6,
)
```

The router excludes agents over budget and ranks cheaper ones higher within budget. As trust scores accumulate, the system learns which model performs best per capability.

### Resilient Data Pipelines
Three ETL agents registered with the same capability. If the primary hits 3 consecutive failures, its circuit opens and the next-best agent takes over automatically — no manual failover configuration.

### Enterprise Tool Composition
Teams register their internal tooling as agents:
- `salesforce_agent` — CRM queries
- `analytics_agent` — data warehouse SQL
- `slack_agent` — formatted channel notifications

An orchestrator composes them into workflows at runtime, discovering what's available rather than having it hardwired.

---

## Comparison

| Feature | LangGraph / CrewAI | Google A2A | **AgentMesh** |
|---|---|---|---|
| Topology | Centralized graph | Spec only | Decentralized mesh |
| Discovery | Hardwired at build time | Static agent cards | Semantic search at runtime |
| Negotiation | None | Defined, not implemented | Accept / reject / counter |
| Trust & Reputation | None | None | ELO scoring, auto-updates |
| Circuit Breaker | None | None | Built-in, auto-recovery |
| Cost-Aware Routing | None | None | Budget filter + cost factor |
| Load-Aware Routing | None | None | Live task count per agent |
| Quality Evaluation | None | None | OutputEvaluator per-capability |
| MCP Tool Support | Via LangChain | None | Native MCPToolClient |
| Working Runtime | Yes | No (spec only) | Yes |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              AgentMesh Core Framework               │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │           Registry  (port 8080)               │  │
│  │  Registration · Discovery · Trust · Traces    │  │
│  │  Semantic routing · Circuit breaker · Memory  │  │
│  └────────────────────────┬──────────────────────┘  │
│                           │  REST + WebSocket        │
│  ┌────────────────────────▼──────────────────────┐  │
│  │              MeshAgent SDK                    │  │
│  │   register · heartbeat · discover · delegate  │  │
│  │   negotiate · trust report · MCP tools        │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │   React Dashboard  (port 3000)                │  │
│  │   Mesh Graph · Trace Timeline · Memory Panel  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│      Bundled Example Agents  (demo only)            │
│  Any agent you write sits here too — using any      │
│  LLM, database, API, or no AI at all.               │
│                                                     │
│  ResearchAgent :9001  ── LangGraph orchestrator     │
│  DataAgent     :9002  ── pandas/numpy + Groq        │
│  CodeAgent     :9003  ── code gen + subprocess      │
│  WriterAgent   :9004  ── report synthesis + Groq    │
│  WebSearchAgent:9005  ── brave-search MCP           │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
agentmesh/
│
├── mesh/                  # Core framework — registry, router, protocol
│   ├── models.py          # Pydantic schemas — protocol source of truth
│   ├── registry.py        # FastAPI service: register, discover, trust, traces, memory, stats
│   ├── router.py          # 4-factor semantic routing (match/trust/load/cost)
│   ├── db.py              # asyncpg + full in-memory fallback (no Docker needed)
│   ├── evaluator.py       # OutputEvaluator: deterministic + LLM-as-judge quality scoring
│   ├── mcp_client.py      # MCPToolClient: subprocess MCP server wrapper, mock fallback
│   ├── memory.py          # AgentMemory: Redis-backed session state, in-memory fallback
│   ├── cli.py             # Click CLI
│   └── migrations/        # SQL migrations 001–004
│
├── sdk/                   # Core framework — agent base class
│   └── agent.py           # MeshAgent base class + @capability decorator
│                          # Handles: registration, heartbeat, WebSocket task server,
│                          # discover(), delegate(), counter-proposal retry, MCP tools
│
├── agents/                # Example agents — bundled demo implementations
│   │                      # Not part of the core framework. Replace with your own.
│   ├── research_agent.py  # LangGraph orchestrator: plan→discover→delegate→synthesize
│   ├── data_agent.py      # Real pandas/numpy CSV computation; LLM narrates, doesn't invent
│   ├── code_agent.py      # Code generation via Groq + optional subprocess execution
│   ├── writer_agent.py    # Markdown report synthesis via Groq
│   └── web_search_agent.py # Web search via brave-search MCP (mock fallback built-in)
│
├── dashboard/             # Vite + React 18 + TypeScript + Tailwind
│   └── src/
│       ├── components/
│       │   ├── MeshGraph.tsx       # SVG physics graph: nodes, edges, drag-to-pin, agent detail
│       │   ├── TraceTimeline.tsx   # Swim-lane timeline with scrubber and event log
│       │   ├── MemoryPanel.tsx     # Live session state inspector with collapsible JSON
│       │   ├── CommandPalette.tsx  # ⌘K command palette
│       │   ├── TaskWaterfall.tsx   # Per-task waterfall modal
│       │   └── JsonTree.tsx        # Syntax-colored collapsible JSON tree
│       └── hooks/
│           ├── useDashboardSocket.ts  # Auto-reconnecting WebSocket for live events
│           ├── useAgents.ts           # Registry polling hook
│           └── useStats.ts            # Aggregate mesh stats polling
│
├── examples/              # Minimal standalone examples to get started
│   ├── hello_agent.py          # 15-line agent — register and respond
│   ├── translation_agent.py    # Full translation agent (runnable standalone)
│   └── two_agent_pipeline.py   # 2-agent mini-mesh without LangGraph
│
├── tests/
│   ├── test_models.py          # Schema validation, JSON roundtrips
│   ├── test_registry.py        # Registration, discovery, trust endpoints
│   ├── test_router.py          # Semantic match, cost filter, ranking
│   ├── test_trust.py           # ELO convergence and bounds
│   ├── test_circuit_breaker.py # Degraded status, recovery, ranking penalty
│   ├── test_negotiation.py     # Accept/reject/counter over real WebSocket
│   ├── test_e2e.py             # Full register→discover→delegate→verify
│   ├── test_auth.py            # JWT auth, token endpoint, protected endpoints
│   ├── test_memory.py          # AgentMemory set/get/TTL/clear/sessions
│   ├── test_data_agent.py      # Real CSV computation, SSRF/path protection
│   ├── test_code_agent.py      # Print capture, syntax error, timeout, isolation
│   ├── test_evaluator.py       # OutputEvaluator per-capability scoring
│   ├── test_mcp_client.py      # MCPToolClient tool listing, calling, mock fallback
│   └── test_latency.py         # Latency tracking and reporting
│
├── data/
│   └── startups.csv    # 100-row demo dataset (company/sector/funding/year/country/stage)
│
├── demo.py         # One-command demo: registry + 5 example agents + research task
├── .env.example    # Environment variable template
└── pyproject.toml  # Package metadata + dependencies
```

---

## API Reference

### Registry Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/agents/register` | Register an agent with its capability manifest |
| `GET` | `/agents` | List all registered agents |
| `GET` | `/agents/{agent_id}` | Get a specific agent's record |
| `DELETE` | `/agents/{agent_id}` | Deregister an agent |
| `POST` | `/discover` | Find agents matching a DiscoveryQuery |
| `POST` | `/trust/update` | Report task outcome and update trust score |
| `GET` | `/agents/{agent_id}/trust_history` | Per-agent trust score history |
| `GET` | `/traces` | Query stored trace events |
| `POST` | `/traces` | Emit a trace event |
| `GET` | `/stats` | Aggregate mesh stats (agents, tasks, avg trust, sessions) |
| `GET` | `/memory` | List active session IDs |
| `GET` | `/memory/{session_id}` | Inspect session state (auth required) |
| `POST` | `/auth/token` | Issue a JWT for an agent |
| `WS` | `/ws/heartbeat/{agent_id}` | Agent heartbeat + task load reporting |
| `WS` | `/ws/dashboard` | Real-time event stream for the dashboard |

### DiscoveryQuery Fields

```python
DiscoveryQuery(
    capability_name="analyze_csv",       # exact match
    capability_description="...",        # semantic match
    min_trust_score=0.4,                 # reputation filter
    max_cost_usd=0.005,                  # budget filter
    tags=["data", "analysis"],           # tag overlap boost (soft — partial match still scores)
    top_k=5,                             # max results (1–20)
)
```

---

## Running Tests

```bash
pip install -e ".[all,dev]"
pytest tests/ -v
```

176 tests, no external dependencies required. The in-memory database and memory fallbacks mean no PostgreSQL, Redis, or Docker needed.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Demo agents only | — | Groq API key ([free at console.groq.com](https://console.groq.com)) — not needed for your own agents |
| `REGISTRY_URL` | No | `http://localhost:8080` | Registry base URL for agents |
| `REGISTRY_PORT` | No | `8080` | Port for the registry server |
| `DATABASE_URL` | No | in-memory | PostgreSQL connection string |
| `REDIS_URL` | No | in-memory | Redis URL for AgentMemory session state |
| `AGENT_SECRET` | No | dev mode | JWT signing secret — set to enable auth enforcement |
| `CSV_SAFE_DIR` | No | `/data/uploads` | Allowed base path for csv_path (path traversal protection) |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

---

## Tech Stack

**Core framework** — no LLM dependency:

| Layer | Technology |
|---|---|
| Registry API | FastAPI + asyncpg + PostgreSQL |
| Semantic Routing | sentence-transformers (all-MiniLM-L6-v2) |
| Agent Transport | WebSocket (websockets) |
| MCP Tools | mcp>=1.0.0 — stdio subprocess client |
| Session Memory | Redis (in-memory fallback) |
| Dashboard | React 18 + TypeScript + Vite + Tailwind (custom SVG physics, no charting lib) |
| Testing | pytest + pytest-asyncio |
| CLI | Click |

**Bundled example agents only** — swap these for whatever your agents use:

| Layer | Technology |
|---|---|
| LLM | Groq API — Llama 3.3 70B via langchain-groq |
| Orchestration | LangGraph StateGraph (ResearchAgent only) |
| Data Analysis | pandas + numpy (DataAgent only) |

---

## Security

- **JWT authentication** — set `AGENT_SECRET` to enable. `POST /auth/token` issues tokens; register/deregister/trust endpoints validate ownership (token `sub` must match `agent_id`)
- **SSRF protection** — `csv_url` blocks RFC-1918, loopback, and link-local IPs before fetching
- **Path traversal protection** — `csv_path` restricted to `CSV_SAFE_DIR`; paths outside this dir are rejected with HTTP 400
- **Session memory auth** — `GET /memory/{session_id}` requires a valid JWT; session data may contain intermediate task state
- **Input length limits** — capability descriptions truncated to 1,000 chars before embedding (DoS prevention)
- **Endpoint scheme validation** — agents only connect to `ws://` or `wss://` endpoints
- **Field length constraints** — agent name (128 chars) and description (1,024 chars) capped in the data model
- **Input validation** — trust quality scores outside `[0.0, 1.0]` rejected with HTTP 422; negative costs rejected
- **CodeAgent subprocess** — process isolation only (same OS user). Not a true sandbox — wrap with Docker/bubblewrap before enabling `execute=true` on untrusted input

---

## License

MIT — [Arsh Advani](https://github.com/arshadvani3)
