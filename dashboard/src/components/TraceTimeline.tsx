import { useEffect, useMemo, useRef, useState } from "react";
import { JsonTree } from "./JsonTree";
import type { AgentRecord, TraceEvent } from "../types";

interface Props {
  agents: AgentRecord[];
  traces: TraceEvent[];
}

// ─── event color system ──────────────────────────────────────────────────────

const EVENT_DOT: Record<string, string> = {
  request_sent: "var(--sig-blue)",
  accepted:     "var(--sig-emerald)",
  executing:    "var(--sig-yellow)",
  completed:    "var(--sig-green)",
  failed:       "var(--sig-red)",
  rejected:     "var(--sig-red)",
};

const EVENT_TAG_BG: Record<string, string> = {
  request_sent: "var(--sig-blue-soft)",
  accepted:     "oklch(0.68 0.14 160 / 0.14)",
  executing:    "oklch(0.75 0.14 80 / 0.14)",
  completed:    "var(--sig-green-soft)",
  failed:       "var(--sig-red-soft)",
  rejected:     "var(--sig-red-soft)",
};

const EVENT_TAG_COLOR: Record<string, string> = {
  request_sent: "var(--sig-blue)",
  accepted:     "var(--sig-emerald)",
  executing:    "var(--sig-yellow)",
  completed:    "var(--sig-green)",
  failed:       "var(--sig-red)",
  rejected:     "var(--sig-red)",
};

// ─── types ───────────────────────────────────────────────────────────────────

interface TLEvent {
  raw: TraceEvent;
  ts: number;
  durMs: number;
  isSpan: boolean;
  lane: number;
}

// ─── greedy interval scheduler ───────────────────────────────────────────────

function assignLanes(events: TLEvent[]): TLEvent[] {
  const sorted = [...events].sort((a, b) => a.ts - b.ts);
  const lanes: number[] = [];
  for (const ev of sorted) {
    let placed = false;
    for (let i = 0; i < lanes.length; i++) {
      if (ev.ts >= lanes[i]) {
        ev.lane = i;
        lanes[i] = ev.ts + ev.durMs + 200;
        placed = true;
        break;
      }
    }
    if (!placed) {
      ev.lane = lanes.length;
      lanes.push(ev.ts + ev.durMs + 200);
    }
  }
  return sorted;
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function fmtTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtDur(ms: number) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// ─── EventTag ────────────────────────────────────────────────────────────────

function EventTag({ type }: { type: string }) {
  return (
    <span style={{
      fontFamily: "var(--font-mono)", fontSize: 10, padding: "1px 5px", borderRadius: 2,
      background: EVENT_TAG_BG[type] ?? "var(--bg-elev)",
      color: EVENT_TAG_COLOR[type] ?? "var(--fg-tertiary)",
      textTransform: "uppercase", letterSpacing: "0.04em", whiteSpace: "nowrap",
    }}>
      {type.replace(/_/g, "_")}
    </span>
  );
}

// ─── EventDetailPanel ────────────────────────────────────────────────────────

function EventDetailPanel({ ev, agentName, onClose, onOpenWaterfall }: {
  ev: TraceEvent;
  agentName: string;
  onClose: () => void;
  onOpenWaterfall: (taskId: string) => void;
}) {
  return (
    <div style={{ width: 320, flexShrink: 0, borderLeft: "1px solid var(--border-subtle)", background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ height: 44, display: "flex", alignItems: "center", padding: "0 14px", borderBottom: "1px solid var(--border-subtle)", gap: 8 }}>
        <EventTag type={ev.event_type} />
        <span style={{ flex: 1, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-tertiary)", overflow: "hidden", textOverflow: "ellipsis" }}>
          {ev.trace_id.slice(0, 16)}…
        </span>
        <button onClick={onClose} style={{ color: "var(--fg-muted)", fontSize: 11, padding: 2, borderRadius: 2 }}>✕</button>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "12px 14px", display: "flex", flexDirection: "column", gap: 14 }}>
        {/* stat row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
          {[
            { l: "From",  v: agentName || ev.from_agent.replace(/^agt_/, "") },
            { l: "To",    v: ev.to_agent.replace(/^agt_/, "") },
            { l: "Time",  v: fmtTime(ev.timestamp) },
          ].map(({ l, v }) => (
            <div key={l} style={{ background: "var(--bg-elev)", borderRadius: "var(--radius)", padding: "7px 8px" }}>
              <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 3 }}>{l}</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis" }}>{v}</div>
            </div>
          ))}
        </div>

        {/* task section */}
        <div style={{ background: "var(--bg-elev)", borderRadius: "var(--radius)", padding: "10px 10px" }}>
          <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 6 }}>Task</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-secondary)", marginBottom: 8, overflow: "hidden", textOverflow: "ellipsis" }}>
            {ev.task_id}
          </div>
          <button
            onClick={() => onOpenWaterfall(ev.task_id)}
            style={{ fontSize: 11, padding: "5px 8px", background: "var(--accent-soft)", border: "1px solid var(--accent-border)", borderRadius: "var(--radius)", color: "var(--accent)", display: "inline-flex", alignItems: "center", gap: 5 }}
          >
            <span>↗</span> View full task waterfall
          </button>
        </div>

        {/* payload */}
        {Object.keys(ev.payload).length > 0 && (
          <div>
            <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 8 }}>Payload</div>
            <div style={{ background: "var(--bg-elev)", borderRadius: "var(--radius)", padding: "10px 12px" }}>
              <JsonTree data={ev.payload} defaultOpen={true} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── SwimRow ─────────────────────────────────────────────────────────────────

const SWIM_ROW_H = 56;
const LANE_H = 22;

function SwimRow({ agent, events, windowStart, windowEnd, onSelect, selectedId, isNow }: {
  agent: AgentRecord;
  events: TLEvent[];
  windowStart: number;
  windowEnd: number;
  onSelect: (ev: TraceEvent) => void;
  selectedId: string | null;
  isNow: boolean;
}) {
  const span = windowEnd - windowStart;
  const xPct = (ts: number) => ((ts - windowStart) / span) * 100;

  const laned = useMemo(() => assignLanes(
    events.filter((e) => e.ts >= windowStart - e.durMs && e.ts <= windowEnd)
  ), [events, windowStart, windowEnd]);

  const maxLane = laned.reduce((m, e) => Math.max(m, e.lane), 0);
  const rowH = Math.max(SWIM_ROW_H, (maxLane + 1) * LANE_H + 14);

  const trust = agent.trust_score;
  const dotColor = trust >= 0.8 ? "var(--sig-green)" : trust >= 0.5 ? "var(--sig-amber)" : "var(--sig-red)";

  const nowPct = xPct(Date.now());
  const showNow = isNow && nowPct >= 0 && nowPct <= 100;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", borderBottom: "1px solid var(--border-subtle)", minHeight: rowH }}>
      {/* label */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 7, padding: "8px 12px", borderRight: "1px solid var(--border-subtle)" }}>
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: dotColor, display: "inline-block", marginTop: 4, flexShrink: 0 }} />
        <div style={{ overflow: "hidden" }}>
          <div style={{ fontSize: 12, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {agent.manifest.name}
          </div>
          <div style={{ fontSize: 10, color: "var(--fg-tertiary)", fontFamily: "var(--font-mono)" }}>
            {events.length} events
          </div>
        </div>
      </div>

      {/* swim track */}
      <div style={{ position: "relative", overflow: "hidden" }}>
        {/* NOW line */}
        {showNow && (
          <div style={{ position: "absolute", top: 0, bottom: 0, left: `${nowPct}%`, width: 1, background: "var(--accent)", zIndex: 5, pointerEvents: "none" }} />
        )}

        {laned.map((ev) => {
          const left = xPct(ev.ts);
          const right = xPct(ev.ts + ev.durMs);
          const top = 7 + ev.lane * LANE_H;
          const isSelected = ev.raw.trace_id === selectedId;

          if (ev.isSpan) {
            const widthPct = Math.max(0.5, right - left);
            const hasFail = ev.raw.event_type === "failed";
            const bg = hasFail ? "var(--sig-red-soft)" : "oklch(0.72 0.15 150 / 0.15)";
            const border = hasFail ? "oklch(0.68 0.18 25 / 0.4)" : "oklch(0.74 0.15 150 / 0.35)";
            const color = hasFail ? "var(--sig-red)" : "var(--sig-green)";
            return (
              <div
                key={ev.raw.trace_id}
                onClick={() => onSelect(ev.raw)}
                title={`executing · ${fmtTime(ev.raw.timestamp)}`}
                style={{
                  position: "absolute",
                  left: `${left}%`,
                  width: `${widthPct}%`,
                  top, height: LANE_H - 4,
                  background: bg,
                  border: `1px solid ${border}`,
                  borderRadius: 3,
                  display: "flex", alignItems: "center", padding: "0 5px",
                  fontFamily: "var(--font-mono)", fontSize: 9.5, color,
                  cursor: "pointer", whiteSpace: "nowrap", overflow: "hidden",
                  outline: isSelected ? `1px solid ${color}` : "none",
                  zIndex: 2,
                }}
              >
                exec
              </div>
            );
          }

          return (
            <div
              key={ev.raw.trace_id}
              onClick={() => onSelect(ev.raw)}
              title={`${ev.raw.event_type} · ${fmtTime(ev.raw.timestamp)}`}
              style={{
                position: "absolute",
                left: `${left}%`,
                top: top + 3,
                width: 8, height: 8, borderRadius: "50%",
                background: isSelected ? "var(--fg)" : (EVENT_DOT[ev.raw.event_type] ?? "var(--fg-tertiary)"),
                transform: "translateX(-50%)",
                cursor: "pointer", zIndex: 2,
                boxShadow: isSelected ? `0 0 0 2px ${EVENT_DOT[ev.raw.event_type] ?? "var(--fg-tertiary)"}` : "none",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

// ─── main ─────────────────────────────────────────────────────────────────────

type FilterType = "all" | "running" | "errors";

export default function TraceTimeline({ agents, traces }: Props) {
  const [offsetSec, setOffsetSec] = useState(0);
  const [filter, setFilter] = useState<FilterType>("all");
  const [taskSearch, setTaskSearch] = useState("");
  const [selectedEv, setSelectedEv] = useState<TraceEvent | null>(null);
  const [now, setNow] = useState(Date.now());
  const scrubberRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);


  const agentNameById = useMemo(() => {
    const m: Record<string, string> = {};
    for (const a of agents) m[a.manifest.agent_id] = a.manifest.name;
    return m;
  }, [agents]);

  // Convert traces to TLEvent
  const allEvents = useMemo<TLEvent[]>(() => traces.map((t) => ({
    raw: t,
    ts: new Date(t.timestamp).getTime(),
    durMs: t.event_type === "executing" ? 3000 : 0,
    isSpan: t.event_type === "executing",
    lane: 0,
  })), [traces]);

  // Window: last 120 seconds ending at now - offsetSec
  const windowEnd = now - offsetSec * 1000;
  const windowStart = windowEnd - 120_000;

  // Filter events for log table
  const logEvents = useMemo(() => {
    let evts = [...traces].reverse();
    if (filter === "errors") evts = evts.filter((e) => e.event_type === "failed" || e.event_type === "rejected");
    if (filter === "running") evts = evts.filter((e) => e.event_type === "executing");
    if (taskSearch.trim()) evts = evts.filter((e) => e.task_id.toLowerCase().includes(taskSearch.toLowerCase()) || e.from_agent.toLowerCase().includes(taskSearch.toLowerCase()));
    return evts.slice(0, 80);
  }, [traces, filter, taskSearch]);

  const runningCount = traces.filter((e) => e.event_type === "executing").length;
  const errorCount = traces.filter((e) => e.event_type === "failed" || e.event_type === "rejected").length;

  const isLive = offsetSec === 0;

  // Time axis ticks (10s intervals, 13 ticks over 120s)
  const tickCount = 7;
  const ticks = Array.from({ length: tickCount }, (_, i) => {
    const msAgo = ((tickCount - 1 - i) / (tickCount - 1)) * 120_000;
    const ts = windowEnd - msAgo;
    const secAgo = Math.round((now - ts) / 1000);
    const label = secAgo === 0 ? "now" : `-${secAgo}s`;
    return { pct: ((ts - windowStart) / (windowEnd - windowStart)) * 100, label };
  });

  // Only show agents that have events or are registered
  const activeAgents = agents.length > 0 ? agents : [];

  const openWaterfall = (taskId: string) => {
    if (typeof (window as unknown as { __openTask?: (id: string) => void }).__openTask === "function") {
      (window as unknown as { __openTask: (id: string) => void }).__openTask(taskId);
    }
  };

  const selectedAgentName = selectedEv ? (agentNameById[selectedEv.from_agent] ?? selectedEv.from_agent.replace(/^agt_/, "")) : "";

  if (traces.length === 0) {
    return (
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, color: "var(--fg-tertiary)" }}>
        <div style={{ width: 44, height: 44, border: "1px dashed var(--border)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--fg-muted)" }}>
          <ActivityIcon />
        </div>
        <div style={{ fontSize: 13, color: "var(--fg-secondary)", fontWeight: 500 }}>No traces yet</div>
        <div style={{ fontSize: 12, color: "var(--fg-tertiary)", maxWidth: 280, textAlign: "center", lineHeight: 1.5 }}>
          Run a task to see agent interactions in the timeline.
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* scrubber */}
      <div style={{ height: 36, display: "flex", alignItems: "center", gap: 10, padding: "0 14px", borderBottom: "1px solid var(--border-subtle)", background: "var(--bg-panel)", flexShrink: 0 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)", whiteSpace: "nowrap" }}>–120s</span>
        <div style={{ flex: 1, position: "relative" }}>
          <input
            ref={scrubberRef}
            type="range" min={0} max={120} step={1}
            value={offsetSec}
            onChange={(e) => setOffsetSec(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent)", cursor: "pointer" }}
          />
        </div>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)", whiteSpace: "nowrap" }}>now</span>
        <div
          onClick={() => setOffsetSec(0)}
          style={{
            display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 8px", borderRadius: "var(--radius)",
            fontSize: 10.5, fontFamily: "var(--font-mono)", cursor: "pointer",
            background: isLive ? "var(--sig-green-soft)" : "var(--bg-elev)",
            color: isLive ? "var(--sig-green)" : "var(--fg-tertiary)",
            border: isLive ? "1px solid oklch(0.74 0.15 150 / 0.3)" : "1px solid var(--border-subtle)",
          }}
        >
          <span style={{ width: 5, height: 5, borderRadius: "50%", background: "currentColor", display: "inline-block" }} />
          {isLive ? "live" : `−${offsetSec}s`}
        </div>
      </div>

      {/* toolbar */}
      <div style={{ height: 36, display: "flex", alignItems: "center", gap: 8, padding: "0 14px", borderBottom: "1px solid var(--border-subtle)", background: "var(--bg-panel)", flexShrink: 0 }}>
        {(["all", "running", "errors"] as FilterType[]).map((f) => {
          const counts: Record<FilterType, number | null> = { all: traces.length, running: runningCount, errors: errorCount };
          const isActive = filter === f;
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: "3px 9px", borderRadius: "var(--radius)", fontSize: 11, fontWeight: 500, cursor: "pointer",
                background: isActive ? "var(--accent-soft)" : "transparent",
                border: isActive ? "1px solid var(--accent-border)" : "1px solid transparent",
                color: isActive ? "var(--accent)" : "var(--fg-tertiary)",
                display: "inline-flex", alignItems: "center", gap: 5,
              }}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
              <span style={{ fontFamily: "var(--font-mono)", fontSize: 9.5, opacity: 0.7 }}>{counts[f]}</span>
            </button>
          );
        })}

        <div style={{ width: 1, height: 14, background: "var(--border-subtle)", margin: "0 4px" }} />

        <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 8px", background: "var(--bg-base)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)", flex: "0 1 180px" }}>
          <SearchIcon />
          <input
            placeholder="Search task ID or agent…"
            value={taskSearch}
            onChange={(e) => setTaskSearch(e.target.value)}
            style={{ flex: 1, fontSize: 11, color: "var(--fg)", background: "none", border: "none", outline: "none", fontFamily: "var(--font-sans)" }}
          />
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {Object.entries(EVENT_DOT).slice(0, 5).map(([type, color]) => (
            <span key={type} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 10, color: "var(--fg-tertiary)", fontFamily: "var(--font-mono)" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, display: "inline-block" }} />
              {type.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      </div>

      {/* swim lanes + log + detail */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "row" }}>
          {/* lanes */}
          <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column" }}>
            {/* time axis header */}
            <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", borderBottom: "1px solid var(--border-subtle)", position: "sticky", top: 0, background: "var(--bg-panel)", zIndex: 10, flexShrink: 0 }}>
              <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, padding: "0 12px", display: "flex", alignItems: "center", height: 28, borderRight: "1px solid var(--border-subtle)" }}>
                Agent
              </div>
              <div style={{ position: "relative", height: 28 }}>
                {ticks.map((t, i) => (
                  <span key={i}>
                    <span style={{ position: "absolute", bottom: 5, left: `${t.pct}%`, fontFamily: "var(--font-mono)", fontSize: 9.5, color: "var(--fg-tertiary)", transform: "translateX(-50%)", background: "var(--bg-panel)", padding: "0 3px" }}>
                      {t.label}
                    </span>
                    <div style={{ position: "absolute", top: 0, bottom: 0, left: `${t.pct}%`, width: 1, background: "var(--border-subtle)", opacity: 0.5 }} />
                  </span>
                ))}
                {/* NOW label on first row */}
                {isLive && (
                  <div style={{ position: "absolute", top: 0, bottom: 0, left: "100%", width: 1, background: "var(--accent)", zIndex: 5 }}>
                    <span style={{ position: "absolute", bottom: 3, right: 3, fontFamily: "var(--font-mono)", fontSize: 8.5, color: "var(--accent)", fontWeight: 600 }}>NOW</span>
                  </div>
                )}
              </div>
            </div>

            {/* rows */}
            {activeAgents.map((agent, i) => {
              const agEvents = allEvents.filter((e) => e.raw.from_agent === agent.manifest.agent_id);
              return (
                <SwimRow
                  key={agent.manifest.agent_id}
                  agent={agent}
                  events={agEvents}
                  windowStart={windowStart}
                  windowEnd={windowEnd}
                  onSelect={setSelectedEv}
                  selectedId={selectedEv?.trace_id ?? null}
                  isNow={i === 0 && isLive}
                />
              );
            })}
          </div>

          {/* event detail panel */}
          {selectedEv && (
            <EventDetailPanel
              ev={selectedEv}
              agentName={selectedAgentName}
              onClose={() => setSelectedEv(null)}
              onOpenWaterfall={openWaterfall}
            />
          )}
        </div>

        {/* events log */}
        <div style={{ height: 240, borderTop: "1px solid var(--border-subtle)", display: "flex", flexDirection: "column", flexShrink: 0 }}>
          {/* log header */}
          <div style={{ height: 32, display: "grid", gridTemplateColumns: "80px 110px 130px 1fr 130px 130px 70px", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", background: "var(--bg-panel)", position: "sticky", top: 0, zIndex: 5, flexShrink: 0 }}>
            {["Time", "Type", "Task", "Task ID", "From", "To", "Dur"].map((h) => (
              <div key={h} style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, padding: "0 10px" }}>{h}</div>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto" }}>
            {logEvents.map((ev) => {
              const isSelected = selectedEv?.trace_id === ev.trace_id;
              return (
                <div
                  key={ev.trace_id}
                  onClick={() => setSelectedEv(isSelected ? null : ev)}
                  style={{
                    display: "grid", gridTemplateColumns: "80px 110px 130px 1fr 130px 130px 70px",
                    alignItems: "center", cursor: "pointer", fontSize: 12,
                    background: isSelected ? "var(--accent-soft)" : "transparent",
                    borderBottom: "1px solid var(--border-subtle)",
                  }}
                  onMouseEnter={(el) => { if (!isSelected) (el.currentTarget as HTMLElement).style.background = "var(--bg-elev)"; }}
                  onMouseLeave={(el) => { if (!isSelected) (el.currentTarget as HTMLElement).style.background = "transparent"; }}
                >
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)", padding: "7px 10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {fmtTime(ev.timestamp)}
                  </div>
                  <div style={{ padding: "7px 10px" }}>
                    <EventTag type={ev.event_type} />
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-secondary)", padding: "7px 10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <span
                      onClick={(e) => { e.stopPropagation(); openWaterfall(ev.task_id); }}
                      style={{ cursor: "pointer", borderBottom: "1px dashed var(--border-subtle)", color: "var(--fg-secondary)" }}
                      title="Open waterfall"
                    >
                      {ev.task_id.slice(0, 14)}…
                    </span>
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)", padding: "7px 10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {ev.task_id}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--fg-secondary)", padding: "7px 10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {agentNameById[ev.from_agent] ?? ev.from_agent.replace(/^agt_/, "")}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--fg-secondary)", padding: "7px 10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {agentNameById[ev.to_agent] ?? ev.to_agent.replace(/^agt_/, "")}
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)", padding: "7px 10px" }}>
                    {ev.event_type === "executing" ? fmtDur(3000) : "—"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── icons ────────────────────────────────────────────────────────────────────

function SearchIcon() {
  return <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--fg-muted)", flexShrink: 0 }}><circle cx="11" cy="11" r="6"/><path d="m20 20-3.5-3.5"/></svg>;
}

function ActivityIcon() {
  return <svg width={20} height={20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>;
}
