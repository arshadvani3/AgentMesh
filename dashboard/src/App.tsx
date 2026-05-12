/**
 * App -- AgentMesh observability dashboard root component.
 *
 * Three views accessible via tab navigation:
 *   1. Mesh Graph    -- live force-directed agent network
 *   2. Trace Timeline -- swimlane view of cross-agent workflows
 *   3. Memory        -- live session state inspector
 *
 * Data sources:
 *   - REST polling: GET /agents (every 5s via useAgents)
 *   - REST polling: GET /stats  (every 3s via useStats)
 *   - WebSocket: ws://localhost:8000/ws/dashboard (via useDashboardSocket)
 */

import { useState } from "react";
import { Activity, Database, GitBranch, Network, RefreshCw, Wifi, WifiOff } from "lucide-react";
import AgentCard from "./components/AgentCard";
import MemoryPanel from "./components/MemoryPanel";
import MeshGraph from "./components/MeshGraph";
import TraceTimeline from "./components/TraceTimeline";
import { useAgents } from "./hooks/useAgents";
import { useDashboardSocket } from "./hooks/useDashboardSocket";
import { useStats } from "./hooks/useStats";
import type { AgentRecord } from "./types";

type Tab = "graph" | "timeline" | "memory";

export default function App() {
  const { agents, loading, error } = useAgents();
  const traces = useDashboardSocket();
  const stats = useStats();
  const [activeTab, setActiveTab] = useState<Tab>("graph");
  const [selectedAgent, setSelectedAgent] = useState<AgentRecord | null>(null);

  const degradedCnt = agents.filter((a) => a.status === "degraded").length;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Top bar */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Network size={20} className="text-cyan-400" />
          <span className="font-bold text-lg text-white">AgentMesh</span>
          <span className="text-gray-500 text-sm">Dashboard</span>
        </div>

        {/* Stats chips */}
        <div className="flex items-center gap-3 ml-4">
          <StatChip
            label="Agents"
            value={String(stats.total_agents || agents.length)}
            color="text-cyan-400"
          />
          <StatChip
            label="Healthy"
            value={String(stats.agents_by_status?.healthy ?? agents.filter((a) => a.status === "healthy").length)}
            color="text-green-400"
          />
          {degradedCnt > 0 && (
            <StatChip
              label="Degraded"
              value={String(degradedCnt)}
              color="text-yellow-400"
            />
          )}
          <StatChip
            label="Avg Trust"
            value={
              stats.avg_trust
                ? (stats.avg_trust * 100).toFixed(0) + "%"
                : agents.length > 0
                ? ((agents.reduce((s, a) => s + a.trust_score, 0) / agents.length) * 100).toFixed(0) + "%"
                : "—"
            }
            color="text-yellow-400"
          />
          <StatChip
            label="Tasks Done"
            value={String(stats.total_tasks_completed)}
            color="text-purple-400"
          />
          <StatChip
            label="Sessions"
            value={String(stats.active_sessions)}
            color="text-blue-400"
          />
          <StatChip
            label="Traces"
            value={String(traces.length)}
            color="text-gray-400"
          />
        </div>

        {/* Connection status */}
        <div className="ml-auto flex items-center gap-2 text-xs">
          {error ? (
            <>
              <WifiOff size={14} className="text-red-400" />
              <span className="text-red-400">Registry unreachable</span>
            </>
          ) : loading ? (
            <>
              <RefreshCw size={14} className="text-gray-400 animate-spin" />
              <span className="text-gray-400">Connecting...</span>
            </>
          ) : (
            <>
              <Wifi size={14} className="text-green-400" />
              <span className="text-green-400">Connected</span>
            </>
          )}
        </div>
      </header>

      {/* Tab nav */}
      <nav className="border-b border-gray-800 px-6 flex gap-1">
        <TabButton
          active={activeTab === "graph"}
          onClick={() => setActiveTab("graph")}
          icon={<Network size={14} />}
          label="Mesh Graph"
        />
        <TabButton
          active={activeTab === "timeline"}
          onClick={() => setActiveTab("timeline")}
          icon={<Activity size={14} />}
          label="Trace Timeline"
        />
        <TabButton
          active={activeTab === "memory"}
          onClick={() => setActiveTab("memory")}
          icon={<Database size={14} />}
          label="Memory"
          badge={stats.active_sessions > 0 ? String(stats.active_sessions) : undefined}
        />
      </nav>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* View area */}
        <div
          className={`flex-1 overflow-hidden p-4 ${selectedAgent && activeTab !== "memory" ? "pr-2" : ""}`}
        >
          {activeTab === "graph" && (
            <div className="h-full relative">
              <MeshGraph
                agents={agents}
                traces={traces}
                onSelectAgent={setSelectedAgent}
              />
            </div>
          )}

          {activeTab === "timeline" && (
            <div className="h-full overflow-hidden">
              <TraceTimeline agents={agents} traces={traces} />
            </div>
          )}

          {activeTab === "memory" && (
            <div className="h-full overflow-hidden -m-4">
              <MemoryPanel />
            </div>
          )}
        </div>

        {/* Agent detail panel -- hidden on memory tab */}
        {selectedAgent && activeTab !== "memory" && (
          <aside className="w-80 shrink-0 border-l border-gray-800 p-4 overflow-auto">
            <div className="flex items-center gap-2 mb-3">
              <GitBranch size={14} className="text-cyan-400" />
              <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                Agent Detail
              </span>
            </div>
            <AgentCard
              record={selectedAgent}
              traces={traces}
              onClose={() => setSelectedAgent(null)}
            />
          </aside>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-2 flex items-center text-xs text-gray-600">
        <span>AgentMesh v0.1.0</span>
        <span className="mx-2">|</span>
        <span>Registry: http://localhost:8000</span>
        <span className="mx-2">|</span>
        <span>WS: ws://localhost:8000/ws/dashboard</span>
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------

function StatChip({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-1.5 bg-gray-900 rounded-lg px-3 py-1">
      <span className="text-gray-500 text-xs">{label}</span>
      <span className={`font-bold text-sm ${color}`}>{value}</span>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  label,
  badge,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  badge?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
        active
          ? "border-cyan-400 text-cyan-400"
          : "border-transparent text-gray-500 hover:text-gray-300"
      }`}
    >
      {icon}
      {label}
      {badge && (
        <span className="bg-cyan-900 text-cyan-300 text-xs font-bold px-1.5 py-0.5 rounded-full leading-none">
          {badge}
        </span>
      )}
    </button>
  );
}
