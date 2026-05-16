import { useEffect, useMemo } from "react";
import type { TraceEvent } from "../types";

interface Props {
  taskId: string;
  traces: TraceEvent[];
  onClose: () => void;
}

const MARKER_COLOR: Record<string, string> = {
  request_sent: "var(--sig-blue)",
  accepted:     "var(--sig-emerald)",
  completed:    "var(--sig-green)",
  failed:       "var(--sig-red)",
};

export default function TaskWaterfall({ taskId, traces, onClose }: Props) {
  const events = useMemo(
    () => traces
      .filter((e) => e.task_id === taskId)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()),
    [traces, taskId]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!events.length) return null;

  const now = Date.now();
  const startTs = new Date(events[0].timestamp).getTime();
  // For executing events, estimate a 3s duration since TraceEvent has no durMs
  const endTs = Math.max(...events.map((e) => {
    const ts = new Date(e.timestamp).getTime();
    return e.event_type === "executing" ? ts + 3000 : ts;
  }));
  const durationMs = endTs - startTs;
  const padMs = Math.max(500, durationMs * 0.04);
  const axisStart = startTs - padMs;
  const axisEnd = endTs + padMs;
  const axisSpan = axisEnd - axisStart;

  const xPct = (ts: number) => ((ts - axisStart) / axisSpan) * 100;

  // One lane per agent (first appearance order)
  const laneOrder: string[] = [];
  const laneIdx = new Map<string, number>();
  for (const e of events) {
    if (e.from_agent && !laneIdx.has(e.from_agent)) {
      laneIdx.set(e.from_agent, laneOrder.length);
      laneOrder.push(e.from_agent);
    }
  }

  const status = events.find((e) => e.event_type === "failed")
    ? "failed"
    : events.some((e) => e.event_type === "completed")
    ? "completed"
    : "running";

  const statusColor = status === "failed" ? "var(--sig-red)" : status === "running" ? "var(--sig-yellow)" : "var(--sig-green)";
  const taskName = events[0].task_id;

  // Time ticks
  const tickCount = 6;
  const ticks = Array.from({ length: tickCount }, (_, i) => {
    const ts = axisStart + (axisSpan * i) / (tickCount - 1);
    const dt = (ts - startTs) / 1000;
    return { x: xPct(ts), label: dt >= 0 ? `+${dt.toFixed(1)}s` : `${dt.toFixed(1)}s` };
  });

  return (
    <div
      onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "oklch(0 0 0 / 0.55)", zIndex: 60, backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: "4vh 4vw" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ width: "100%", maxWidth: 1100, maxHeight: "86vh", background: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "0 30px 80px oklch(0 0 0 / 0.6)", overflow: "hidden", display: "flex", flexDirection: "column" }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", padding: "14px 18px", borderBottom: "1px solid var(--border-subtle)", gap: 12 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: statusColor, boxShadow: `0 0 8px ${statusColor}`, display: "inline-block" }} />
              <span style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}>{taskName}</span>
              <StatusPill status={status} />
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-tertiary)" }}>
              <CopyId value={taskId}>{taskId}</CopyId>
            </div>
          </div>

          <div style={{ display: "flex", marginLeft: "auto", alignItems: "stretch" }}>
            {[
              { l: "Duration", v: `${(durationMs / 1000).toFixed(2)}s` },
              { l: "Agents",   v: String(laneOrder.length) },
              { l: "Events",   v: String(events.length) },
              { l: "Started",  v: `${Math.max(0, Math.round((now - startTs) / 1000))}s ago` },
            ].map(({ l, v }) => (
              <div key={l} style={{ padding: "0 14px", borderLeft: "1px solid var(--border-subtle)", display: "flex", flexDirection: "column", gap: 2 }}>
                <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600 }}>{l}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 500 }}>{v}</div>
              </div>
            ))}
          </div>

          <button onClick={onClose} style={{ color: "var(--fg-tertiary)", marginLeft: 8, width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "var(--radius)" }}>
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflow: "auto", padding: "14px 18px" }}>
          {/* Axis */}
          <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", height: 30, borderBottom: "1px solid var(--border-subtle)", position: "relative", marginBottom: 6 }}>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, display: "flex", alignItems: "center", paddingLeft: 12 }}>Agent</div>
            <div style={{ position: "relative", height: 30 }}>
              {ticks.map((t, i) => (
                <span key={i}>
                  <span style={{ position: "absolute", bottom: 6, left: `${t.x}%`, fontFamily: "var(--font-mono)", fontSize: 9.5, color: "var(--fg-tertiary)", transform: "translateX(-50%)", background: "var(--bg-panel)", padding: "0 4px" }}>{t.label}</span>
                  <div style={{ position: "absolute", top: 22, bottom: -1000, left: `${t.x}%`, width: 1, background: "var(--border-subtle)", opacity: 0.5 }} />
                </span>
              ))}
            </div>
          </div>

          {/* Lanes */}
          {laneOrder.map((agId) => {
            const agEvents = events.filter((e) => e.from_agent === agId);
            return (
              <div key={agId} style={{ display: "grid", gridTemplateColumns: "160px 1fr", minHeight: 30, borderBottom: "1px solid var(--border-subtle)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "0 12px", fontSize: 12 }}>
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--sig-green)", display: "inline-block" }} />
                  {agId.replace(/^agt_/, "")}
                </div>
                <div style={{ position: "relative", padding: "4px 0" }}>
                  {ticks.map((t, i) => (
                    <div key={i} style={{ position: "absolute", top: 0, bottom: 0, left: `${t.x}%`, width: 1, background: "var(--border-subtle)", opacity: 0.4 }} />
                  ))}
                  {agEvents.map((e) => {
                    const leftPct = xPct(new Date(e.timestamp).getTime());
                    if (e.event_type === "executing") {
                      const durMs = 3000;
                      const rightPct = xPct(new Date(e.timestamp).getTime() + durMs);
                      const hasFail = events.some((x) => x.event_type === "failed" && x.from_agent === e.from_agent);
                      const spanStatus = hasFail ? "failed" : "completed";
                      const bgColor = spanStatus === "failed" ? "var(--sig-red-soft)" : "var(--sig-green-soft)";
                      const borderColor = spanStatus === "failed" ? "oklch(0.68 0.18 25 / 0.5)" : "oklch(0.74 0.15 150 / 0.5)";
                      const textColor = spanStatus === "failed" ? "var(--sig-red)" : "var(--sig-green)";
                      return (
                        <div key={e.trace_id} style={{ position: "absolute", height: 18, left: `${leftPct}%`, width: `${rightPct - leftPct}%`, top: 6, background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 3, display: "flex", alignItems: "center", padding: "0 6px", fontFamily: "var(--font-mono)", fontSize: 10, color: textColor, cursor: "pointer", whiteSpace: "nowrap", overflow: "hidden" }}>
                          {(durMs / 1000).toFixed(2)}s
                        </div>
                      );
                    }
                    return (
                      <div key={e.trace_id} style={{ position: "absolute", width: 8, height: 8, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)", background: MARKER_COLOR[e.event_type] ?? "var(--fg-tertiary)", left: `${leftPct}%`, zIndex: 2 }} title={`${e.event_type} · +${((new Date(e.timestamp).getTime() - startTs) / 1000).toFixed(2)}s`} />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ borderTop: "1px solid var(--border-subtle)", padding: "10px 18px", display: "flex", gap: 16, fontSize: 11, color: "var(--fg-tertiary)" }}>
          <span>← waterfall view of one task across {laneOrder.length} agents</span>
          <span style={{ marginLeft: "auto" }}>
            <Kbd>esc</Kbd> to close
          </span>
        </div>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const styles: Record<string, React.CSSProperties> = {
    running:   { color: "var(--sig-yellow)", background: "var(--sig-yellow-soft)" },
    completed: { color: "var(--sig-green)",  background: "var(--sig-green-soft)" },
    failed:    { color: "var(--sig-red)",    background: "var(--sig-red-soft)" },
  };
  return (
    <span style={{ fontFamily: "var(--font-mono)", fontSize: 9.5, padding: "0 4px", borderRadius: 2, textTransform: "uppercase", letterSpacing: "0.04em", ...styles[status] }}>
      {status}
    </span>
  );
}

function CopyId({ children, value }: { children: React.ReactNode; value: string }) {
  const handle = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard?.writeText(value);
  };
  return (
    <span onClick={handle} style={{ cursor: "pointer", borderBottom: "1px dashed var(--border)", color: "var(--fg-tertiary)" }} title="Click to copy">
      {children}
    </span>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, background: "var(--bg-base)", border: "1px solid var(--border-subtle)", borderRadius: 2, padding: "0 4px", height: 16, display: "inline-flex", alignItems: "center", color: "var(--fg-secondary)" }}>
      {children}
    </span>
  );
}
