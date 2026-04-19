/**
 * TraceTimeline -- horizontal swimlane timeline of cross-agent workflows.
 *
 * Each agent gets its own row. Tasks appear as colored bars with duration.
 * Delegation arrows connect bars across swimlanes. Clicking a bar shows the
 * full request/response payload in a side panel.
 *
 * Updates in real-time via WebSocket (traces prop refreshed by parent).
 */

import { useState, useMemo } from "react";
import { X } from "lucide-react";
import type { AgentRecord, TraceEvent } from "../types";

interface Props {
  agents: AgentRecord[];
  traces: TraceEvent[];
}

interface SelectedEvent {
  event: TraceEvent;
  agentName: string;
}

const EVENT_COLORS: Record<string, string> = {
  request_sent: "bg-blue-600",
  accepted:     "bg-emerald-600",
  executing:    "bg-yellow-600",
  completed:    "bg-green-600",
  failed:       "bg-red-600",
  rejected:     "bg-red-800",
};

const EVENT_TEXT: Record<string, string> = {
  request_sent: "text-blue-300",
  accepted:     "text-emerald-300",
  executing:    "text-yellow-300",
  completed:    "text-green-300",
  failed:       "text-red-300",
  rejected:     "text-red-300",
};

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", { hour12: false, fractionalSecondDigits: 1 });
}

/** Group trace events by the agent they are "from". */
function groupByAgent(
  traces: TraceEvent[],
  agents: AgentRecord[]
): Map<string, TraceEvent[]> {
  const agentIds = new Set(agents.map((a) => a.manifest.agent_id));
  const result = new Map<string, TraceEvent[]>();
  for (const agent of agents) {
    result.set(agent.manifest.agent_id, []);
  }
  for (const ev of traces) {
    if (agentIds.has(ev.from_agent)) {
      result.get(ev.from_agent)!.push(ev);
    }
  }
  return result;
}

export default function TraceTimeline({ agents, traces }: Props) {
  const [selected, setSelected] = useState<SelectedEvent | null>(null);

  const grouped = useMemo(() => groupByAgent(traces, agents), [traces, agents]);

  const agentNameById = useMemo(() => {
    const m: Record<string, string> = {};
    for (const a of agents) m[a.manifest.agent_id] = a.manifest.name;
    return m;
  }, [agents]);

  if (traces.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Waiting for agent interactions...
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Swimlanes */}
      <div className="flex-1 overflow-auto pr-2">
        <div className="min-w-[600px]">
          {agents.map((agent) => {
            const agentId = agent.manifest.agent_id;
            const agentEvents = grouped.get(agentId) ?? [];
            return (
              <div key={agentId} className="mb-4">
                {/* Lane header */}
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-gray-300 w-36 truncate">
                    {agent.manifest.name}
                  </span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      agent.status === "healthy"
                        ? "bg-green-900 text-green-300"
                        : agent.status === "degraded"
                        ? "bg-yellow-900 text-yellow-300"
                        : "bg-gray-800 text-gray-400"
                    }`}
                  >
                    {agent.status}
                  </span>
                </div>

                {/* Events row */}
                <div className="flex items-center gap-2 flex-wrap bg-gray-900 rounded-lg px-3 py-2 min-h-[44px]">
                  {agentEvents.length === 0 ? (
                    <span className="text-xs text-gray-600">no events</span>
                  ) : (
                    agentEvents.slice(-15).map((ev) => (
                      <button
                        key={ev.trace_id}
                        onClick={() =>
                          setSelected({ event: ev, agentName: agent.manifest.name })
                        }
                        className={`flex flex-col items-start px-2 py-1 rounded text-xs cursor-pointer
                          hover:opacity-80 transition-opacity border border-transparent hover:border-gray-600
                          ${EVENT_COLORS[ev.event_type] ?? "bg-gray-700"}`}
                        title={`${ev.event_type} at ${formatTime(ev.timestamp)}`}
                      >
                        <span className="font-mono font-semibold">
                          {ev.event_type.replace("_", " ")}
                        </span>
                        <span className="text-gray-300 text-opacity-70">
                          {formatTime(ev.timestamp)}
                        </span>
                        {ev.to_agent !== ev.from_agent && (
                          <span className="text-gray-200 text-opacity-60 truncate max-w-[100px]">
                            to: {agentNameById[ev.to_agent] ?? ev.to_agent.slice(0, 12)}
                          </span>
                        )}
                      </button>
                    ))
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Recent events log */}
        <div className="mt-4">
          <h3 className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">
            Recent Events (last 20)
          </h3>
          <div className="space-y-0.5">
            {traces.slice(-20).reverse().map((ev) => (
              <div
                key={ev.trace_id}
                className="flex items-center gap-3 text-xs py-1 px-2 rounded hover:bg-gray-800 cursor-pointer"
                onClick={() =>
                  setSelected({
                    event: ev,
                    agentName: agentNameById[ev.from_agent] ?? ev.from_agent,
                  })
                }
              >
                <span className="text-gray-500 font-mono w-20 shrink-0">
                  {formatTime(ev.timestamp)}
                </span>
                <span
                  className={`w-24 shrink-0 font-semibold ${
                    EVENT_TEXT[ev.event_type] ?? "text-gray-300"
                  }`}
                >
                  {ev.event_type.replace("_", " ")}
                </span>
                <span className="text-gray-400 truncate">
                  {agentNameById[ev.from_agent] ?? ev.from_agent.slice(0, 14)}
                  {" -> "}
                  {agentNameById[ev.to_agent] ?? ev.to_agent.slice(0, 14)}
                </span>
                <span className="text-gray-600 font-mono text-[10px] shrink-0">
                  {ev.task_id.slice(0, 14)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Payload side panel */}
      {selected && (
        <div className="w-80 shrink-0 border-l border-gray-800 pl-4 overflow-auto">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-200">Event Detail</h3>
            <button
              onClick={() => setSelected(null)}
              className="text-gray-500 hover:text-gray-300"
            >
              <X size={16} />
            </button>
          </div>

          <dl className="space-y-2 text-xs">
            <div>
              <dt className="text-gray-500">Agent</dt>
              <dd className="text-gray-200">{selected.agentName}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Event type</dt>
              <dd className={EVENT_TEXT[selected.event.event_type] ?? "text-gray-300"}>
                {selected.event.event_type}
              </dd>
            </div>
            <div>
              <dt className="text-gray-500">Task ID</dt>
              <dd className="text-gray-200 font-mono">{selected.event.task_id}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Timestamp</dt>
              <dd className="text-gray-200 font-mono">
                {new Date(selected.event.timestamp).toISOString()}
              </dd>
            </div>
            <div>
              <dt className="text-gray-500">To</dt>
              <dd className="text-gray-200">
                {agentNameById[selected.event.to_agent] ?? selected.event.to_agent}
              </dd>
            </div>
            <div>
              <dt className="text-gray-500 mb-1">Payload</dt>
              <dd>
                <pre className="text-gray-300 bg-gray-900 rounded p-2 overflow-auto text-[10px] max-h-60">
                  {JSON.stringify(selected.event.payload, null, 2)}
                </pre>
              </dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  );
}
