/**
 * AgentCard -- detailed view for a single mesh agent.
 *
 * Shows: name, status, capabilities, trust score over time (recharts),
 * MCP servers, live task stats, and recent task history (from trace events).
 */

import {
  Activity,
  CheckCircle,
  ChevronRight,
  Clock,
  Code2,
  Server,
  Shield,
  Tag,
  XCircle,
} from "lucide-react";
import { useMemo } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AgentRecord, TraceEvent } from "../types";

interface Props {
  record: AgentRecord;
  traces: TraceEvent[];
  onClose: () => void;
}

function trustColorClass(score: number): string {
  if (score < 0.5) return "text-red-400";
  if (score < 0.8) return "text-yellow-400";
  return "text-green-400";
}

function statusBadge(status: AgentRecord["status"]): JSX.Element {
  const styles = {
    healthy:  "bg-green-900 text-green-300",
    degraded: "bg-yellow-900 text-yellow-300",
    offline:  "bg-gray-800 text-gray-400",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full ${styles[status]}`}>
      {status}
    </span>
  );
}

/** Build a synthetic trust-score-over-time series from trace events. */
function buildTrustHistory(
  agentId: string,
  traces: TraceEvent[],
  currentTrust: number
): { time: string; trust: number }[] {
  const completions = traces
    .filter(
      (t) =>
        t.to_agent === agentId &&
        (t.event_type === "completed" || t.event_type === "failed")
    )
    .slice(-10);

  if (completions.length === 0) {
    return [{ time: "now", trust: currentTrust }];
  }

  // Simple backward simulation -- each completion nudges trust by +/-0.05
  let score = currentTrust;
  const points: { time: string; trust: number }[] = [
    { time: "now", trust: currentTrust },
  ];

  for (let i = completions.length - 1; i >= 0; i--) {
    const ev = completions[i];
    const delta = ev.event_type === "completed" ? 0.05 : -0.05;
    score = Math.max(0, Math.min(1, score - delta));
    const label = new Date(ev.timestamp).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
    });
    points.unshift({ time: label, trust: Math.round(score * 100) / 100 });
  }

  return points;
}

export default function AgentCard({ record, traces, onClose }: Props) {
  const { manifest, trust_score, status, tasks_completed, tasks_failed } = record;

  const trustHistory = useMemo(
    () => buildTrustHistory(manifest.agent_id, traces, trust_score),
    [manifest.agent_id, traces, trust_score]
  );

  const recentTasks = useMemo(
    () =>
      traces
        .filter(
          (t) =>
            t.to_agent === manifest.agent_id || t.from_agent === manifest.agent_id
        )
        .slice(-8)
        .reverse(),
    [traces, manifest.agent_id]
  );

  const successRate =
    tasks_completed + tasks_failed > 0
      ? Math.round((tasks_completed / (tasks_completed + tasks_failed)) * 100)
      : null;

  return (
    <div className="h-full overflow-auto text-sm">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-white">{manifest.name}</h2>
          <div className="flex items-center gap-2 mt-1">
            {statusBadge(status)}
            <span className="text-gray-500 text-xs font-mono">
              {manifest.agent_id}
            </span>
          </div>
          {manifest.description && (
            <p className="text-gray-400 text-xs mt-1">{manifest.description}</p>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 ml-4"
        >
          <XCircle size={18} />
        </button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center gap-1 text-gray-400 text-xs mb-1">
            <Shield size={12} />
            Trust
          </div>
          <div className={`text-xl font-bold ${trustColorClass(trust_score)}`}>
            {(trust_score * 100).toFixed(0)}%
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center gap-1 text-gray-400 text-xs mb-1">
            <CheckCircle size={12} />
            Completed
          </div>
          <div className="text-xl font-bold text-green-400">{tasks_completed}</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center gap-1 text-gray-400 text-xs mb-1">
            <XCircle size={12} />
            Failed
          </div>
          <div className="text-xl font-bold text-red-400">{tasks_failed}</div>
        </div>
      </div>

      {successRate !== null && (
        <div className="flex items-center gap-2 mb-4">
          <Activity size={14} className="text-gray-400" />
          <span className="text-gray-400 text-xs">Success rate:</span>
          <span className="text-white font-semibold text-xs">{successRate}%</span>
          <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full"
              style={{ width: `${successRate}%` }}
            />
          </div>
        </div>
      )}

      {/* Trust score chart */}
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Trust Over Time
        </h3>
        <div className="h-28 bg-gray-900 rounded-lg p-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trustHistory}>
              <XAxis
                dataKey="time"
                tick={{ fill: "#6b7280", fontSize: 9 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: "#6b7280", fontSize: 9 }}
                tickLine={false}
                axisLine={false}
                width={28}
              />
              <Tooltip
                contentStyle={{
                  background: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "6px",
                  fontSize: "11px",
                  color: "#f9fafb",
                }}
              />
              <Area
                type="monotone"
                dataKey="trust"
                stroke="#22c55e"
                fill="#14532d"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Capabilities */}
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1">
          <Code2 size={12} />
          Capabilities ({manifest.capabilities.length})
        </h3>
        <div className="space-y-2">
          {manifest.capabilities.map((cap) => (
            <div key={cap.name} className="bg-gray-800 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 font-mono text-xs text-cyan-400 mb-0.5">
                <ChevronRight size={12} />
                {cap.name}
              </div>
              <p className="text-gray-400 text-xs">{cap.description}</p>
              {cap.avg_latency_ms && (
                <div className="flex items-center gap-1 text-gray-600 text-xs mt-1">
                  <Clock size={10} />
                  ~{cap.avg_latency_ms}ms avg
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Tags */}
      {manifest.tags.length > 0 && (
        <div className="mb-5">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1">
            <Tag size={12} />
            Tags
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {manifest.tags.map((tag) => (
              <span
                key={tag}
                className="bg-gray-800 text-gray-300 text-xs px-2 py-0.5 rounded-full"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* MCP Servers */}
      {manifest.mcp_servers.length > 0 && (
        <div className="mb-5">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1">
            <Server size={12} />
            MCP Servers
          </h3>
          <div className="space-y-1">
            {manifest.mcp_servers.map((srv) => (
              <div
                key={srv}
                className="text-xs text-emerald-400 bg-gray-800 rounded px-2 py-1 font-mono"
              >
                {srv}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent task history */}
      {recentTasks.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Recent Interactions
          </h3>
          <div className="space-y-1">
            {recentTasks.map((ev) => (
              <div
                key={ev.trace_id}
                className="flex items-center gap-2 text-xs text-gray-400 py-1 border-b border-gray-800 last:border-0"
              >
                <span
                  className={`w-2 h-2 rounded-full shrink-0 ${
                    ev.event_type === "completed"
                      ? "bg-green-500"
                      : ev.event_type === "failed"
                      ? "bg-red-500"
                      : ev.event_type === "executing"
                      ? "bg-yellow-500"
                      : "bg-blue-500"
                  }`}
                />
                <span className="font-mono text-gray-600 shrink-0">
                  {new Date(ev.timestamp).toLocaleTimeString("en-US", {
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <span className="truncate">{ev.event_type.replace("_", " ")}</span>
                <span className="text-gray-600 font-mono shrink-0">
                  {ev.task_id.slice(0, 10)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
