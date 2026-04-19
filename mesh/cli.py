"""AgentMesh CLI -- command-line interface for managing the mesh.

Usage:
    agentmesh registry start           Start the registry server
    agentmesh agent list               List all registered agents
    agentmesh demo run                 Run the end-to-end demo
    agentmesh demo run --query "..."   Run demo with a custom query
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import click
import httpx
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://localhost:8000")


@click.group()
def main():
    """AgentMesh -- peer-to-peer agent discovery and communication network."""


# ---------------------------------------------------------------------------
# Registry commands
# ---------------------------------------------------------------------------

@main.group()
def registry():
    """Manage the AgentMesh registry."""


@registry.command("start")
@click.option("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
@click.option("--port", default=8000, type=int, help="Bind port (default: 8000)")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload")
@click.option(
    "--log-level",
    default=os.environ.get("LOG_LEVEL", "info").lower(),
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Uvicorn log level",
)
def registry_start(host: str, port: int, reload: bool, log_level: str):
    """Start the AgentMesh registry server with uvicorn.

    Args:
        host: Interface to bind.
        port: Port to listen on.
        reload: Enable hot reload (development only).
        log_level: Uvicorn log verbosity.
    """
    console.print(f"[cyan]Starting AgentMesh registry on {host}:{port}...[/]")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "mesh.registry:app",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level,
    ]
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Agent commands
# ---------------------------------------------------------------------------

@main.group()
def agent():
    """Inspect agents registered on the mesh."""


@agent.command("list")
@click.option(
    "--registry-url",
    default=REGISTRY_URL,
    envvar="REGISTRY_URL",
    help="Registry URL (default: http://localhost:8000)",
)
@click.option(
    "--status",
    type=click.Choice(["all", "healthy", "degraded", "offline"]),
    default="all",
    help="Filter by agent status",
)
def agent_list(registry_url: str, status: str):
    """List all agents registered on the mesh.

    Args:
        registry_url: URL of the registry service.
        status: Status filter.
    """
    async def _fetch():
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{registry_url}/agents", timeout=10)
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError:
                console.print(f"[red]Cannot connect to registry at {registry_url}[/]")
                return []

    agents = asyncio.run(_fetch())

    if status != "all":
        agents = [a for a in agents if a["manifest"]["status"] == status or a["status"] == status]

    if not agents:
        console.print("[yellow]No agents found.[/]")
        return

    table = Table(
        title=f"[bold cyan]Registered Agents ({len(agents)} total)[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="dim font-mono", min_width=20)
    table.add_column("Name", style="cyan", min_width=18)
    table.add_column("Status", justify="center", min_width=10)
    table.add_column("Trust", justify="right", min_width=8)
    table.add_column("Capabilities", min_width=20)
    table.add_column("Tags", style="dim", min_width=16)

    for rec in agents:
        m = rec["manifest"]
        agent_status = rec.get("status", "unknown")
        status_display = {
            "healthy": "[green]healthy[/]",
            "degraded": "[yellow]degraded[/]",
            "offline": "[red]offline[/]",
        }.get(agent_status, agent_status)

        cap_names = ", ".join(c["name"] for c in m.get("capabilities", []))
        tags = ", ".join(m.get("tags", []))
        trust = f"{rec.get('trust_score', 0):.2f}"

        table.add_row(
            m["agent_id"][:22],
            m["name"],
            status_display,
            trust,
            cap_names[:30] or "[dim]none[/]",
            tags[:20] or "[dim]none[/]",
        )

    console.print(table)

    # Summary
    healthy = sum(1 for a in agents if a.get("status") == "healthy")
    console.print(
        f"\n[dim]{healthy}/{len(agents)} healthy | "
        f"Avg trust: {sum(a.get('trust_score', 0) for a in agents)/len(agents):.2f}[/]"
    )


# ---------------------------------------------------------------------------
# Demo commands
# ---------------------------------------------------------------------------

@main.group()
def demo():
    """Run the AgentMesh end-to-end demonstration."""


@demo.command("run")
@click.option(
    "--query",
    default=(
        "Write a competitive analysis of LangChain vs CrewAI vs AutoGen for building "
        "multi-agent systems. Include code examples showing how each framework handles "
        "tool calling."
    ),
    help="Research query to submit to the mesh",
)
def demo_run(query: str):
    """Start all agents and run a live multi-agent research task.

    Args:
        query: The research task to submit.
    """
    # Import here to avoid importing demo.py's subprocess logic at module load
    import importlib.util
    import pathlib

    demo_path = pathlib.Path(__file__).parent.parent / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", demo_path)
    demo_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(demo_module)  # type: ignore[union-attr]

    asyncio.run(demo_module.run_demo(query))
