/**
 * App -- AgentMesh observability dashboard root component.
 *
 * Three views accessible via tab navigation:
 *   1. Mesh Graph -- live force-directed agent network
 *   2. Trace Timeline -- swimlane view of cross-agent workflows
 *   3. Agent Detail -- selected agent stats and history
 *
 * Data sources:
 *   - REST polling: GET /agents (every 5s via useAgents)
 *   - WebSocket: ws://localhost:8000/ws/dashboard (via useDashboardSocket)
 */

import { useState } from "react";
import { Activity, GitBranch, Network, RefreshCw, Wifi, WifiOff } from "lucide-react";
import AgentCard from "./components/AgentCard";
import MeshGraph from "./components/MeshGraph";
import TraceTimeline from "./components/TraceTimeline";
import { useAgents } from "./hooks/useAgents";
import { useDashboardSocket } from "./hooks/useDashboardSocket";
import type { AgentRecord } from "./types";

type Tab = "graph" | "timeline";

export default function App() {
  const { agents, loading, error } = useAgents();
  const traces = useDashboardSocket();
  const [activeTab, setActiveTab] = useState<Tab>("graph");
  const [selectedAgent, setSelectedAgent] = useState<AgentRecord | null>(null);

  const healthyCnt = agents.filter((a) => a.status === "healthy").length;
  const offlineCnt = agents.filter((a) => a.status === "offline").length;
  const avgTrust =
    agents.length > 0
      ? (agents.reduce((s, a) => s + a.trust_score, 0) / agents.length).toFixed(2)
      : "0.00";

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
            value={String(agents.length)}
            color="text-cyan-400"
          />
          <StatChip
            label="Healthy"
            value={String(healthyCnt)}
            color="text-green-400"
          />
          {offlineCnt > 0 && (
            <StatChip
              label="Offline"
              value={String(offlineCnt)}
              color="text-red-400"
            />
          )}
          <StatChip label="Avg Trust" value={avgTrust} color="text-yellow-400" />
          <StatChip
            label="Trace Events"
            value={String(traces.length)}
            color="text-purple-400"
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
      </nav>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* View area */}
        <div
          className={`flex-1 overflow-hidden p-4 ${selectedAgent ? "pr-2" : ""}`}
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
        </div>

        {/* Agent detail panel */}
        {selectedAgent && (
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
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
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
    </button>
  );
}
