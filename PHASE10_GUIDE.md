# Phase 10 — What We Built and Why

This doc explains everything added in Phase 10 in plain language.
No assumed knowledge — just what it is, why it exists, and how it fits together.

---

## The Problem We Were Solving

Before Phase 10, all 4 agents were doing the same thing underneath:
**ask Groq a question, get text back, return it as JSON.**

That means:
- The DataAgent never actually touched any data. It just asked an LLM "what do you think this data says?"
- The CodeAgent generated code but never ran it — you had no idea if it actually worked
- Redis was listed as a dependency but zero lines of code ever used it (looked like a lie in a code review)
- The demo query ("compare LangChain vs CrewAI") is something any chatbot answers — it didn't need a mesh at all

Phase 10 fixes all four of these.

---

## Thing 1: DataAgent Now Does Real Computation

### What changed
The DataAgent can now accept a `csv_path` (a file on disk) or `csv_url` (a link to a CSV online).
When it gets one, it:
1. Loads the actual file
2. Computes real statistics using **pandas** (or numpy if pandas isn't available)
3. Gets numbers like: mean funding = $675M, AI sector = 20 companies, median = $225M
4. Then gives those **real computed numbers** to Groq and says "write a narrative about this"

### Why this matters
Before: LLM guesses what the data "probably" says. Those numbers are hallucinated.
After: LLM is given real computed numbers and asked to explain them. The numbers are true.

Think of it like this — before, you were asking someone who'd never seen your spreadsheet to describe it. Now you're showing them the spreadsheet and asking them to explain what they see.

### The fallback
If no CSV is provided, it still works the old way (LLM-only). Nothing breaks.
If pandas isn't installed, it falls back to numpy + Python's built-in csv module. Same computation, slightly less powerful.

### The dataset we added: `data/startups.csv`
A real 100-row CSV with:
- 9 sectors: AI, Fintech, Data, Cybersecurity, Climatetech, DevTools, Design, Biotech, Logistics
- Columns: company, sector, amount_usd, year, country, stage
- Real companies: OpenAI, Anthropic, Stripe, Databricks, Wiz, etc.
- Real funding rounds from 2019–2023

This is the dataset the demo runs against. The final report now contains actual numbers from this file.

---

## Thing 2: CodeAgent Now Actually Runs Code

### What changed
The CodeAgent generates code the same way it did before (asks Groq).
But now, if you pass `"execute": true` in the request, each snippet gets:
1. Syntax-checked first (fast fail — no subprocess needed)
2. Run in a **completely separate subprocess** with a 5-second timeout
3. Real `stdout`, `stderr`, and `exit_code` captured and returned

### Why subprocess and not just exec()?
`exec()` runs code inside the same Python process as the agent. That means:
- It can access the agent's variables and secrets
- It can't be killed if it runs forever
- stdout isn't captured cleanly

A subprocess is a fresh, isolated process. It has no idea it's inside an agent. You can kill it. It produces real output. It's safe.

### What you get back per example
```
{
  "title": "Sector breakdown chart",
  "code": "import collections\n...",
  "executed": true,
  "execution_result": {
    "stdout": "AI: 20\nFintech: 10\n...",
    "stderr": "",
    "exit_code": 0
  }
}
```

### The opt-in design
Execution is off by default (`"execute": false`). You have to explicitly ask for it.
This means the CodeAgent is safe to use in contexts where running arbitrary code is a bad idea.
Only enable it when you control what's being generated.

### The 4 cases it handles
| Scenario | What happens |
|---|---|
| Code works perfectly | `executed: true`, real stdout returned |
| Syntax error | Caught before subprocess, `executed: false`, error message in stderr |
| Runtime error (e.g. 1/0) | Runs, `exit_code != 0`, traceback in stderr |
| Infinite loop / slow code | Killed after 5 seconds, `executed: false`, "timed out" error |

---

## Thing 3: AgentMemory — Redis Session Storage

### The problem it solves
Every agent task was completely stateless. When the ResearchAgent finished:
- Plan: gone
- Intermediate results: gone
- No way to ask "what did this workflow actually do?"

Also, if the ResearchAgent crashed halfway through delegating subtasks, all progress was lost.

### What AgentMemory is
A key-value store where agents can save and retrieve data during a workflow.
Think of it like a shared notepad that all agents in a workflow can read and write.

```
session_id = "task-demo-1234"

memory.set(session_id, "plan", {"subtasks": [...]})        # save the plan
memory.set(session_id, "result_analyze_csv", {...})        # save a result as it arrives
memory.get(session_id, "plan")                             # read it back
memory.get_session(session_id)                             # get everything for this session
memory.clear(session_id)                                   # clean up when done
```

### The storage backend
- **If Redis is running** (set `REDIS_URL` env var): data goes to Redis. Survives process restarts. Multiple agents across machines can share it.
- **If Redis is not running**: falls back to an in-memory Python dict. Works fine for development and tests. Data lives only as long as the registry process.

This is the same pattern as the database layer — Postgres if available, in-memory if not. Consistent, no surprises.

### TTL (Time to Live)
Every key expires after 1 hour by default. You don't need to manually clean up sessions — they disappear on their own. This prevents memory from filling up with old workflow state.

### Why Redis was already in pyproject.toml
Redis was declared as a dependency from the very beginning but never used. In a code review this looks like abandoned work — "they planned to use this but never got there." Now it's fully wired. The dependency is honest.

---

## Thing 4: ResearchAgent Uses Memory

### What changed in the workflow
The ResearchAgent's LangGraph pipeline now has memory at 3 points:

**After planning:**
Saves the decomposed subtask plan to memory.
```
memory.set(task_id, "plan", {"query": "...", "subtasks": [...]})
```
If the agent crashes after planning but before delegating, a retry can check if a plan already exists instead of re-planning from scratch.

**After each delegation:**
As each subtask finishes, its result is saved immediately.
```
memory.set(task_id, "result_analyze_csv", {real data agent output})
memory.set(task_id, "result_fetch_code", {real code agent output})
```
This means partial progress survives a crash. If 2 of 3 subtasks completed before something went wrong, a retry only needs to redo the 1 that failed.

**After the final report:**
The complete output is saved for later inspection.
```
memory.set(task_id, "final_report", {"report": "...", "agents_consulted": [...]})
```

### session_id in the state
A new `session_id` field was added to `ResearchState` (the LangGraph state dict). This is just the `task_id` passed through from the original request. Every node in the graph can access it to read/write memory.

---

## Thing 5: New Registry Endpoints

Two new HTTP endpoints were added to the registry so you can inspect memory from outside:

### `GET /memory/{session_id}`
Returns everything stored for a workflow session.
```json
{
  "session_id": "task-demo-1234",
  "keys": ["plan", "result_analyze_csv", "result_fetch_code", "final_report"],
  "data": {
    "plan": {"query": "...", "subtasks": [...]},
    "result_analyze_csv": {"analysis": "...", "key_findings": [...]}
  }
}
```
Useful for debugging: "what exactly did each agent produce?"

### `GET /memory`
Lists all active session IDs.
```json
{
  "sessions": ["task-demo-1234", "task-demo-5678"],
  "count": 2
}
```

---

## Thing 6: The New Demo

### Old query
> "Write a competitive analysis of LangChain vs CrewAI vs AutoGen"

**Problem:** Any chatbot answers this. You don't need a mesh. There's nothing real happening.

### New query
> "Analyze the startup funding dataset — find which sectors attracted the most capital, identify funding trends by year, generate Python code to visualize the sector breakdown, and write a strategic investment report with specific numbers from the data."

**Why this is better:**
1. DataAgent loads `data/startups.csv` and computes real stats (not guessed)
2. CodeAgent generates visualization code (and can run it if you want)
3. WriterAgent receives those real numbers and writes a report citing actual figures
4. The final report says things like "AI attracted $X across Y companies" — verifiably true from the dataset

### The `--csv-path` flag
You can now run the demo against any CSV:
```bash
python demo.py --csv-path /path/to/your/data.csv
```
The query adapts automatically because the ResearchAgent discovers capabilities dynamically and passes the file path through to the DataAgent.

### The Run Summary now shows
```
Data source:      data/startups.csv
Session memory:   GET http://localhost:8000/memory/task-demo-1234
```
After the demo completes, you can hit that URL and see every intermediate result stored during the workflow.

---

## How It All Fits Together — The Full Flow

Here's what happens when you run `python demo.py`:

```
1. Registry starts (FastAPI server, port 8000)

2. 4 agents start and register themselves:
   - ResearchAgent (port 9001) — orchestrator
   - DataAgent (port 9002) — real pandas computation
   - CodeAgent (port 9003) — code generation + optional execution
   - WriterAgent (port 9004) — report synthesis

3. Demo sends a task to ResearchAgent via WebSocket:
   {capability: "research", input_data: {query: "...", csv_path: "data/startups.csv"}}

4. ResearchAgent runs its LangGraph pipeline:

   PLAN node:
   - Calls GET /discover to find what agents are on the mesh right now
   - Passes live capability list to Groq: "decompose this query into subtasks"
   - Gets back: [{analyze_csv}, {fetch_code}, {write_report}]
   - Writes plan to memory: memory.set(task_id, "plan", {...})

   DISCOVER node:
   - For each subtask, calls POST /discover
   - Finds the right agent for each capability
   - Router scores: semantic match × 0.35 + trust × 0.35 + availability × 0.15 + cost × 0.15

   DELEGATE node (parallel):
   - Sends analyze_csv task to DataAgent via WebSocket
     → DataAgent loads startups.csv, computes real stats with pandas
     → Returns: {analysis: "...", key_findings: [...], computed_stats: {real numbers}}
   - Sends fetch_code task to CodeAgent via WebSocket (parallel)
     → CodeAgent generates visualization code
     → Returns: {examples: [{code, explanation, execution_result}]}
   - Both run at the same time (asyncio.gather)
   - Each result saved to memory as it arrives

   SYNTHESIZE node:
   - Sends write_report to WriterAgent with all the real data
   - WriterAgent receives actual computed numbers, not guesses
   - Returns a report that quotes real statistics

5. Final report printed to terminal with Rich formatting

6. Run summary shows:
   - Total time elapsed
   - Data source used
   - URL to inspect all memory for this session
```

---

## The 3 New Files

| File | What it is |
|---|---|
| `mesh/memory.py` | The AgentMemory class — Redis + fallback, set/get/clear/TTL |
| `data/startups.csv` | The real 100-row dataset the demo runs against |
| `tests/test_memory.py` | 15 tests: set/get/TTL/clear/sessions/registry endpoints |
| `tests/test_data_agent.py` | 10 tests: numpy path, pandas path, real CSV, missing files |
| `tests/test_code_agent.py` | 10 tests: print capture, syntax error, timeout, isolation |

---

## Numbers

| Metric | Before Phase 10 | After Phase 10 |
|---|---|---|
| Tests passing | 86 | 126 |
| Agents doing real computation | 0 | 1 (DataAgent) |
| Agents executing code | 0 | 1 (CodeAgent, opt-in) |
| Redis usage | 0 lines | ~200 lines |
| Demo produces real numbers | No | Yes |
| Session state survives crash | No | Yes (partial results in memory) |

---

## Interview Talking Points

**"What does the DataAgent actually do?"**
It loads a CSV, computes statistics with pandas (mean, std, percentiles, correlations, value counts), and passes those real numbers to the LLM for narrative synthesis. The LLM explains the data — it doesn't invent it.

**"Why subprocess for code execution?"**
exec() shares process state with the agent and can't be killed. subprocess is isolated, killable, and captures stdout cleanly. It's the same reason you don't eval() untrusted input in web apps.

**"What's AgentMemory for?"**
Two things: crash recovery (partial results survive mid-workflow failures because each result is persisted as it arrives) and observability (operators can inspect exactly what each agent produced via the /memory endpoint). It also activates Redis which was a declared but unused dependency.

**"Why is the new demo better?"**
The old demo asked a question any single LLM can answer. The new demo requires the mesh: real data computation (DataAgent's unique capability), code generation (CodeAgent's unique capability), and synthesis across both — things that genuinely require coordination between specialized agents.
