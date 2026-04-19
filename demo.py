"""AgentMesh end-to-end demo script.

Starts the registry and all four demo agents, waits for them to register,
submits a complex research task, streams the trace of inter-agent interactions,
and prints the final report.

Usage:
    python demo.py
    python demo.py --query "Compare FastAPI vs Django for building AI microservices"
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import click
import httpx
import websockets
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://localhost:8000")
REGISTRY_PORT = 8000

AGENTS = [
    {"name": "Research Agent", "module": "agents.research_agent", "port": 9001},
    {"name": "Data Agent",     "module": "agents.data_agent",     "port": 9002},
    {"name": "Code Agent",     "module": "agents.code_agent",     "port": 9003},
    {"name": "Writer Agent",   "module": "agents.writer_agent",   "port": 9004},
]

DEFAULT_QUERY = (
    "Write a competitive analysis of LangChain vs CrewAI vs AutoGen for building "
    "multi-agent systems. Include code examples showing how each framework handles "
    "tool calling."
)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_processes: list[subprocess.Popen] = []


def _cleanup():
    """Terminate all spawned subprocesses."""
    console.print("\n[bold yellow]Shutting down all agents and registry...[/]")
    for proc in _processes:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in _processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    console.print("[green]Cleanup complete.[/]")


def _start_process(module: str, env_extra: dict | None = None) -> subprocess.Popen:
    """Spawn a Python module as a subprocess.

    Args:
        module: Python module path (e.g. 'agents.data_agent').
        env_extra: Additional environment variables to inject.

    Returns:
        The running subprocess.Popen handle.
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    _processes.append(proc)
    return proc


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

async def _wait_for_registry(timeout: int = 30) -> bool:
    """Poll the registry until it is responsive.

    Args:
        timeout: Maximum seconds to wait.

    Returns:
        True if registry came up, False if timed out.
    """
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{REGISTRY_URL}/agents", timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def _wait_for_agents(expected_names: list[str], timeout: int = 60) -> dict[str, bool]:
    """Wait until all expected agents appear in the registry.

    Args:
        expected_names: Agent names to look for.
        timeout: Maximum seconds to wait.

    Returns:
        Dict mapping agent name -> True if registered.
    """
    deadline = time.time() + timeout
    status = {name: False for name in expected_names}

    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{REGISTRY_URL}/agents", timeout=5)
                registered = {r["manifest"]["name"] for r in resp.json()}
                for name in expected_names:
                    status[name] = name in registered
                if all(status.values()):
                    return status
            except Exception:
                pass
            await asyncio.sleep(2)

    return status


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

def _make_status_table(agent_status: dict[str, bool], registry_up: bool) -> Table:
    """Build a Rich table showing live agent startup status.

    Args:
        agent_status: Mapping of agent name -> registered boolean.
        registry_up: Whether the registry is responsive.

    Returns:
        Formatted Rich Table.
    """
    table = Table(
        title="[bold cyan]AgentMesh Startup[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Component", style="cyan", min_width=20)
    table.add_column("Status", justify="center", min_width=12)

    reg_status = "[green]ONLINE[/]" if registry_up else "[yellow]STARTING...[/]"
    table.add_row("Registry (port 8000)", reg_status)

    for name, ready in agent_status.items():
        st = "[green]REGISTERED[/]" if ready else "[yellow]STARTING...[/]"
        table.add_row(name, st)

    return table


# ---------------------------------------------------------------------------
# Trace streaming
# ---------------------------------------------------------------------------

class TraceCollector:
    """Collects TraceEvents from the registry WebSocket dashboard stream."""

    def __init__(self):
        """Initialize an empty trace buffer."""
        self.events: list[dict[str, Any]] = []
        self._ws = None
        self._task: asyncio.Task | None = None

    async def start(self):
        """Connect to the registry dashboard WebSocket and begin collecting."""
        self._task = asyncio.create_task(self._listen())

    async def stop(self):
        """Cancel the listener task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listen(self):
        """Background loop consuming trace events from the dashboard stream."""
        uri = f"ws://localhost:{REGISTRY_PORT}/ws/dashboard"
        try:
            async with websockets.connect(uri) as ws:
                self._ws = ws
                async for message in ws:
                    try:
                        event = json.loads(message)
                        self.events.append(event)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass  # Registry may not be up yet


def _render_trace_table(events: list[dict[str, Any]]) -> Table:
    """Render collected trace events as a Rich timeline table.

    Args:
        events: List of TraceEvent dicts from the registry.

    Returns:
        Formatted Rich Table.
    """
    table = Table(
        title="[bold cyan]Agent Interaction Timeline[/]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Time", style="dim", min_width=12)
    table.add_column("Event", min_width=14)
    table.add_column("From", style="cyan", min_width=16)
    table.add_column("To", style="magenta", min_width=16)
    table.add_column("Task", style="yellow", min_width=12)

    for ev in events[-20:]:  # Show last 20 events
        ts = ev.get("timestamp", "")[:19].replace("T", " ")
        event_type = ev.get("event_type", "")
        color = {
            "request_sent": "blue",
            "accepted": "green",
            "executing": "yellow",
            "completed": "bright_green",
            "failed": "red",
            "rejected": "red",
        }.get(event_type, "white")

        table.add_row(
            ts,
            f"[{color}]{event_type}[/]",
            ev.get("from_agent", "")[:20],
            ev.get("to_agent", "")[:20],
            ev.get("task_id", "")[:12],
        )

    return table


# ---------------------------------------------------------------------------
# Main demo flow
# ---------------------------------------------------------------------------

async def run_demo(query: str):
    """Execute the full end-to-end AgentMesh demonstration.

    Args:
        query: The research task to submit to the Research Agent.
    """
    console.print(Panel.fit(
        "[bold cyan]AgentMesh[/] [white]-- Peer-to-Peer Agent Discovery Demo[/]",
        border_style="cyan",
    ))

    # -- Step 1: Start registry --
    console.print("\n[bold]Step 1:[/] Starting registry...")
    registry_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "mesh.registry:app",
            "--host", "0.0.0.0",
            "--port", str(REGISTRY_PORT),
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _processes.append(registry_proc)

    # -- Step 2: Wait for registry --
    with console.status("[yellow]Waiting for registry...[/]"):
        ok = await _wait_for_registry(timeout=30)

    if not ok:
        console.print("[red]Registry failed to start. Aborting.[/]")
        return

    console.print("[green]Registry online.[/]")

    # -- Step 3: Start agents --
    console.print("\n[bold]Step 2:[/] Starting agents...")
    for ag in AGENTS:
        _start_process(ag["module"])
        await asyncio.sleep(0.5)  # stagger starts slightly

    # -- Step 4: Wait for agents with live table --
    console.print("\n[bold]Step 3:[/] Waiting for agents to register...\n")
    agent_names = [ag["name"] for ag in AGENTS]
    agent_status = {name: False for name in agent_names}

    with Live(console=console, refresh_per_second=2) as live:
        deadline = time.time() + 120
        while time.time() < deadline:
            live.update(_make_status_table(agent_status, registry_up=True))
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{REGISTRY_URL}/agents", timeout=5)
                    registered = {r["manifest"]["name"] for r in resp.json()}
                    for name in agent_names:
                        agent_status[name] = name in registered
            except Exception:
                pass

            if all(agent_status.values()):
                live.update(_make_status_table(agent_status, registry_up=True))
                break
            await asyncio.sleep(2)

    if not all(agent_status.values()):
        missing = [n for n, ok in agent_status.items() if not ok]
        console.print(f"[yellow]Warning: These agents did not register: {missing}[/]")
        console.print("[yellow]Proceeding anyway...[/]")
    else:
        console.print("[green]All agents registered and healthy.[/]\n")

    # -- Step 5: Start trace collector --
    tracer = TraceCollector()
    await tracer.start()
    await asyncio.sleep(1)

    # -- Step 6: Submit task to Research Agent --
    console.print(Panel(
        f"[bold]{query}[/]",
        title="[bold cyan]Research Task",
        border_style="blue",
    ))

    # Find the research agent endpoint
    research_agent_ws = None
    research_agent_id = None

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{REGISTRY_URL}/agents", timeout=10)
        for rec in resp.json():
            if rec["manifest"]["name"] == "Research Agent":
                research_agent_ws = rec["manifest"]["endpoint"]
                research_agent_id = rec["manifest"]["agent_id"]
                break

    if not research_agent_ws:
        console.print("[red]Research Agent not found on mesh. Aborting.[/]")
        await tracer.stop()
        return

    console.print(f"\n[bold]Step 4:[/] Delegating task to Research Agent at {research_agent_ws}...")
    start_time = datetime.utcnow()

    task_payload = {
        "method": "task.request",
        "params": {
            "task_id": f"task-demo-{int(time.time())}",
            "capability": "research",
            "input_data": {"query": query},
            "requester_id": "demo-script",
            "target_id": research_agent_id,
            "deadline_ms": 120000,
            "priority": 5,
            "context": "End-to-end demo run",
            "created_at": datetime.utcnow().isoformat(),
        },
    }

    final_result = None
    tokens_used = 0

    try:
        async with websockets.connect(research_agent_ws, open_timeout=15) as ws:
            await ws.send(json.dumps(task_payload))

            # Negotiation response
            with console.status("[yellow]Negotiating with Research Agent...[/]"):
                neg_raw = await asyncio.wait_for(ws.recv(), timeout=15)
            neg = json.loads(neg_raw)
            neg_status = neg.get("result", {}).get("status", "unknown")
            console.print(f"[green]Negotiation: {neg_status}[/]")

            if neg_status in ("rejected", "countered"):
                reason = neg.get("result", {}).get("reason", "")
                console.print(f"[red]Task rejected: {reason}[/]")
                await tracer.stop()
                return

            # Wait for result (up to 2 minutes)
            console.print("[yellow]Research in progress (this may take 30-90 seconds)...[/]")
            with console.status("[cyan]Agents collaborating on the mesh...[/]"):
                result_raw = await asyncio.wait_for(ws.recv(), timeout=120)

            result_data = json.loads(result_raw).get("params", {})
            final_result = result_data
            tokens_used = result_data.get("tokens_used", 0)

    except TimeoutError:
        console.print("[red]Timeout waiting for Research Agent result.[/]")
    except Exception as e:
        console.print(f"[red]Error communicating with Research Agent: {e}[/]")

    elapsed = (datetime.utcnow() - start_time).total_seconds()

    # -- Step 7: Print trace --
    await asyncio.sleep(2)  # collect any lingering trace events
    await tracer.stop()

    console.print(f"\n[bold]Step 5:[/] Interaction Trace ({len(tracer.events)} events)\n")
    if tracer.events:
        console.print(_render_trace_table(tracer.events))
    else:
        console.print("[dim]No trace events captured (registry trace streaming may not have fired)[/]")

    # -- Step 8: Summary stats --
    console.print()
    stats = Table(box=box.SIMPLE, show_header=False)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="white")
    stats.add_row("Total elapsed", f"{elapsed:.1f}s")
    stats.add_row("Tokens used", str(tokens_used) if tokens_used else "n/a")
    stats.add_row("Trace events", str(len(tracer.events)))
    stats.add_row("Status", final_result.get("status", "unknown") if final_result else "no result")
    console.print(Panel(stats, title="[bold cyan]Run Summary", border_style="cyan"))

    # -- Step 9: Print final report --
    if final_result and final_result.get("status") == "completed":
        output = final_result.get("output", {})
        report = output.get("report", "")
        agents_consulted = output.get("agents_consulted", [])
        sources = output.get("sources_used", [])

        console.print(f"\n[dim]Agents consulted: {', '.join(agents_consulted)}[/]")
        console.print(f"[dim]Sources used: {', '.join(sources)}[/]\n")

        if report:
            console.print(Panel(
                Markdown(report),
                title="[bold green]Final Report",
                border_style="green",
                padding=(1, 2),
            ))
        else:
            console.print("[yellow]No report content returned.[/]")
    else:
        console.print("[yellow]Task did not complete successfully.[/]")
        if final_result:
            console.print(f"[dim]{json.dumps(final_result, indent=2)[:500]}...[/]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--query",
    default=DEFAULT_QUERY,
    help="Research task to submit to the mesh",
)
def main(query: str):
    """Run the AgentMesh end-to-end demonstration.

    Starts the registry and all demo agents, submits a research task,
    and prints the full trace and final report.
    """
    # Register cleanup handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (_cleanup(), sys.exit(0)))

    try:
        asyncio.run(run_demo(query))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
