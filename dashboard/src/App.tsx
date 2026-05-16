import { useEffect, useRef, useState } from "react";
import CommandPalette from "./components/CommandPalette";
import MemoryPanel from "./components/MemoryPanel";
import MeshGraph from "./components/MeshGraph";
import TaskWaterfall from "./components/TaskWaterfall";
import TraceTimeline from "./components/TraceTimeline";
import { useAgents } from "./hooks/useAgents";
import { useDashboardSocket } from "./hooks/useDashboardSocket";
import { useStats } from "./hooks/useStats";
import type { AgentRecord, TraceEvent } from "./types";

type Tab = "graph" | "timeline" | "memory";

// Signal color map for event types
const EVENT_COLORS: Record<string, string> = {
  request_sent: "var(--sig-blue)",
  accepted:     "var(--sig-emerald)",
  executing:    "var(--sig-yellow)",
  completed:    "var(--sig-green)",
  failed:       "var(--sig-red)",
  rejected:     "var(--sig-red)",
};

export default function App() {
  const { agents, loading, error } = useAgents();
  const traces = useDashboardSocket();
  const stats = useStats();

  const [tab, setTab] = useState<Tab>("graph");
  const [selectedAgent, setSelectedAgent] = useState<AgentRecord | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [errorsOpen, setErrorsOpen] = useState(false);
  const [openTaskId, setOpenTaskId] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());

  const goSeqRef = useRef("");

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(true);
        return;
      }
      if (e.key === "Escape") {
        setPaletteOpen(false);
        setErrorsOpen(false);
        setOpenTaskId(null);
      }
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "g") {
        goSeqRef.current = "g";
        setTimeout(() => { goSeqRef.current = ""; }, 800);
        return;
      }
      if (goSeqRef.current === "g") {
        if (e.key === "m") { setTab("graph"); goSeqRef.current = ""; }
        if (e.key === "t") { setTab("timeline"); goSeqRef.current = ""; }
        if (e.key === "s") { setTab("memory"); goSeqRef.current = ""; }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Expose global task opener for child components — stable ref so it's only wired once
  const openTaskRef = useRef(setOpenTaskId);
  useEffect(() => {
    (window as Window & { __openTask?: (id: string) => void }).__openTask = (id) => openTaskRef.current(id);
    return () => { delete (window as Window & { __openTask?: (id: string) => void }).__openTask; };
  }, []);

  // Uptime counter (from page load)
  const uptimeRef = useRef(Date.now());
  const uptimeSec = Math.floor((now - uptimeRef.current) / 1000);
  const uptimeStr = `${String(Math.floor(uptimeSec / 3600)).padStart(2, "0")}:${String(Math.floor((uptimeSec % 3600) / 60)).padStart(2, "0")}:${String(uptimeSec % 60).padStart(2, "0")}`;

  // Derived header stats
  const healthy   = agents.filter((a) => a.status === "healthy").length;
  const atRisk    = agents.filter((a) => a.trust_score < 0.5).length;
  const avgTrust  = agents.length > 0
    ? (agents.reduce((s, a) => s + a.trust_score, 0) / agents.length * 100).toFixed(1)
    : "—";
  const liveCount = traces.filter((e) => e.event_type === "executing").length;
  const failures  = traces.filter((e) => e.event_type === "failed").sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const tickerEvents = [...traces].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  ).slice(0, 14);

  const handleAction = (kind: string, val: string) => {
    if (kind === "tab") setTab(val as Tab);
    if (kind === "toast") {
      setToast(val);
      setTimeout(() => setToast(null), 2800);
    }
  };

  const connected = !error && !loading;

  return (
    <div style={{
      display: "grid",
      gridTemplateRows: "44px 36px 1fr 28px",
      height: "100%",
      background: "var(--bg-base)",
      color: "var(--fg)",
      fontFamily: "var(--font-sans)",
    }}>

      {/* ── Header ── */}
      <header style={{
        display: "flex",
        alignItems: "center",
        padding: "0 12px 0 14px",
        borderBottom: "1px solid var(--border-subtle)",
        background: "var(--bg-panel)",
        gap: 14,
        height: 44,
      }}>
        {/* Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 13, letterSpacing: "-0.01em", paddingRight: 10, borderRight: "1px solid var(--border-subtle)", height: 24 }}>
          <div style={{
            width: 18, height: 18, borderRadius: 3,
            background: "conic-gradient(from 220deg at 50% 50%, var(--accent), oklch(0.62 0.16 200), var(--accent))",
            position: "relative", boxShadow: "0 0 0 1px oklch(0.72 0.13 245 / 0.35) inset",
            flexShrink: 0,
          }}>
            <div style={{ position: "absolute", inset: 4, borderRadius: 2, background: "var(--bg-panel)" }} />
          </div>
          <span>AgentMesh</span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)", background: "var(--bg-elev)", border: "1px solid var(--border-subtle)", padding: "1px 5px", borderRadius: 3, marginLeft: 4 }}>
            local · dev
          </span>
        </div>

        {/* Stat strip */}
        <div style={{ display: "flex", alignItems: "center", flex: 1 }}>
          <StatCol label="Agents" value={String(agents.length)} delta="+2" />
          <StatCol label="Healthy" value={String(healthy)} valueColor="var(--sig-green)" suffix={`/${agents.length}`} />
          <StatCol label="At risk" value={String(atRisk)} valueColor={atRisk > 0 ? "var(--sig-red)" : "var(--fg)"} />
          <StatCol label="Avg trust" value={avgTrust} suffix="%" />
          <StatCol label="Tasks · 24h" value={String((stats.total_tasks_completed / 1000).toFixed(1)) + "k"} delta="+312" />
          <StatColRunning value={liveCount} />
        </div>

        {/* Right side */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: "auto" }}>
          {failures.length > 0 && (
            <div style={{ position: "relative" }}>
              <button
                onClick={() => setErrorsOpen((o) => !o)}
                style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: "var(--radius)", border: "1px solid var(--sig-red)", color: "var(--sig-red)", background: "var(--sig-red-soft)", fontSize: 11, fontWeight: 500 }}
              >
                <PulseDot color="var(--sig-red)" />
                {failures.length} {failures.length === 1 ? "failure" : "failures"}
                <ChevronD />
              </button>
              {errorsOpen && (
                <ErrorsPopover failures={failures} now={now} onClose={() => setErrorsOpen(false)} onOpenTask={(id) => { setOpenTaskId(id); setErrorsOpen(false); }} />
              )}
            </div>
          )}

          <button
            onClick={() => setPaletteOpen(true)}
            style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 8px", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)", color: "var(--fg-secondary)", fontSize: 11, background: "var(--bg-elev)" }}
          >
            <SearchIcon />
            <span>Search</span>
            <Kbd>⌘K</Kbd>
          </button>

          <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "4px 9px", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)", fontSize: 11, fontWeight: 500 }}>
            {connected ? <LiveDot /> : <OfflineDot />}
            <span>{connected ? "Connected" : error ? "Disconnected" : "Connecting…"}</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)" }}>ws://localhost:8080</span>
          </div>

        </div>
      </header>

      {/* ── Tab nav ── */}
      <nav style={{ display: "flex", alignItems: "stretch", background: "var(--bg-base)", borderBottom: "1px solid var(--border-subtle)", padding: "0 14px", gap: 2, height: 36 }}>
        <TabBtn active={tab === "graph"} onClick={() => setTab("graph")} icon={<GraphIcon />} label="Mesh Graph" count={agents.length} />
        <TabBtn active={tab === "timeline"} onClick={() => setTab("timeline")} icon={<TimelineIcon />} label="Trace Timeline" count={traces.length} />
        <TabBtn active={tab === "memory"} onClick={() => setTab("memory")} icon={<MemoryIcon />} label="Memory" count={stats.active_sessions} />
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)", whiteSpace: "nowrap" }}>
            poll <span style={{ color: "var(--fg-secondary)" }}>3s</span> · stream <span style={{ color: "var(--sig-green)" }}>live</span>
          </span>
        </div>
      </nav>

      {/* ── Main view ── */}
      <main style={{ position: "relative", overflow: "hidden" }}>
        {tab === "graph" && (
          <MeshGraph
            agents={agents}
            traces={traces}
            selectedAgent={selectedAgent}
            onSelectAgent={setSelectedAgent}
            now={now}
          />
        )}
        {tab === "timeline" && (
          <TraceTimeline agents={agents} traces={traces} />
        )}
        {tab === "memory" && <MemoryPanel />}
      </main>

      {/* ── Status bar ── */}
      <footer style={{
        display: "flex", alignItems: "center",
        padding: "0 14px",
        borderTop: "1px solid var(--border-subtle)",
        background: "var(--bg-panel)",
        fontFamily: "var(--font-mono)", fontSize: 10.5,
        color: "var(--fg-tertiary)", gap: 16, overflow: "hidden", height: 28,
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: connected ? "var(--sig-green)" : "var(--sig-red)" }} />
          localhost:8000
        </span>
        <span>build <span style={{ color: "var(--fg-secondary)" }}>v0.1.0</span></span>
        <span>uptime <span style={{ color: "var(--fg-secondary)" }}>{uptimeStr}</span></span>

        {/* Live event ticker */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, overflow: "hidden", whiteSpace: "nowrap", WebkitMaskImage: "linear-gradient(90deg, transparent, black 40px, black calc(100% - 40px), transparent)" }}>
          {tickerEvents.map((e) => {
            const c = EVENT_COLORS[e.event_type] ?? "var(--sig-blue)";
            return (
              <span key={e.trace_id} style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--fg-secondary)" }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}`, display: "inline-block" }} />
                <span style={{ color: c }}>{e.event_type}</span>
                <span>·</span>
                <span>{e.from_agent.replace(/^agt_/, "")}</span>
                <span style={{ color: "var(--fg-muted)" }}>→</span>
                <span>{e.to_agent.replace(/^agt_/, "")}</span>
                <span style={{ color: "var(--fg-muted)" }}>·</span>
                <span style={{ color: "var(--fg-muted)" }}>{e.task_id.slice(0, 10)}</span>
              </span>
            );
          })}
        </div>

        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          <Kbd>⌘K</Kbd>
          <span style={{ marginLeft: 6 }}>commands</span>
        </span>
      </footer>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onAction={handleAction} />

      {openTaskId && (
        <TaskWaterfall taskId={openTaskId} traces={traces} onClose={() => setOpenTaskId(null)} />
      )}

      {toast && (
        <div style={{ position: "fixed", bottom: 44, right: 16, zIndex: 60, background: "var(--bg-panel)", border: "1px solid var(--border)", padding: "8px 12px", borderRadius: 6, fontSize: 12, boxShadow: "0 8px 24px oklch(0 0 0 / 0.5)", display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 6px var(--accent)" }} />
          {toast}
        </div>
      )}
    </div>
  );
}

// ─── Small helpers ────────────────────────────────────────────────────────────

function StatCol({ label, value, delta, suffix, valueColor }: { label: string; value: string; delta?: string; suffix?: string; valueColor?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", padding: "0 14px", borderRight: "1px solid var(--border-subtle)", borderLeft: "1px solid var(--border-subtle)", minWidth: 90, marginLeft: -1 }}>
      <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 500 }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 500, letterSpacing: "-0.01em", display: "flex", alignItems: "baseline", gap: 4, color: valueColor ?? "var(--fg)" }}>
        {value}
        {suffix && <span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>{suffix}</span>}
        {delta && <span style={{ fontSize: 10, color: "var(--sig-green)" }}>{delta}</span>}
      </span>
    </div>
  );
}

function StatColRunning({ value }: { value: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", padding: "0 14px", borderRight: "1px solid var(--border-subtle)", minWidth: 90 }}>
      <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 500 }}>Running</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 500, display: "flex", alignItems: "center", gap: 5, color: "var(--sig-yellow)" }}>
        <PulseDot color="var(--sig-yellow)" />
        {value}
      </span>
    </div>
  );
}

function PulseDot({ color }: { color: string }) {
  return (
    <span style={{ position: "relative", display: "inline-block", width: 6, height: 6, flexShrink: 0 }}>
      <span style={{ display: "block", width: 6, height: 6, borderRadius: "50%", background: color }} />
      <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: color, animation: "pulse 2s ease-out infinite" }} />
    </span>
  );
}

function LiveDot() {
  return (
    <span style={{ position: "relative", width: 7, height: 7, display: "inline-block" }}>
      <span style={{ display: "block", width: 7, height: 7, borderRadius: "50%", background: "var(--sig-green)", boxShadow: "0 0 0 3px oklch(0.74 0.15 150 / 0.18)" }} />
      <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "var(--sig-green)", animation: "pulse 2s ease-out infinite" }} />
    </span>
  );
}

function OfflineDot() {
  return <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: "var(--sig-red)" }} />;
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, background: "var(--bg-base)", border: "1px solid var(--border-subtle)", borderRadius: 2, padding: "0 4px", height: 16, display: "inline-flex", alignItems: "center", color: "var(--fg-secondary)" }}>
      {children}
    </span>
  );
}

function TabBtn({ active, onClick, icon, label, count }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string; count: number }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex", alignItems: "center", gap: 7,
        padding: "0 14px", fontSize: 12, fontWeight: 500,
        color: active ? "var(--fg)" : "var(--fg-secondary)",
        position: "relative", cursor: "pointer", whiteSpace: "nowrap",
        transition: "color 120ms",
        background: "none", border: "none",
      }}
    >
      {icon}
      {label}
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, background: "var(--bg-elev)", border: "1px solid var(--border-subtle)", borderRadius: 8, padding: "0 5px", height: 15, display: "inline-flex", alignItems: "center", color: "var(--fg-tertiary)" }}>
        {count}
      </span>
      {active && (
        <span style={{ position: "absolute", left: 8, right: 8, bottom: -1, height: 1.5, background: "var(--accent)", borderRadius: 1 }} />
      )}
    </button>
  );
}

function ErrorsPopover({ failures, now, onClose, onOpenTask }: { failures: TraceEvent[]; now: number; onClose: () => void; onOpenTask: (id: string) => void }) {
  return (
    <div style={{ position: "absolute", top: "calc(100% + 8px)", right: 0, width: 380, background: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: 6, boxShadow: "0 16px 48px oklch(0 0 0 / 0.5)", zIndex: 50, overflow: "hidden" }}>
      <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--sig-red)" }} />
        <span style={{ fontSize: 13, fontWeight: 600 }}>Recent failures</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)", marginLeft: "auto" }}>{failures.length} total</span>
        <button onClick={onClose} style={{ color: "var(--fg-tertiary)" }}>✕</button>
      </div>
      <div style={{ maxHeight: 380, overflow: "auto" }}>
        {failures.slice(0, 10).map((e) => {
          const dt = Math.max(0, Math.round((now - new Date(e.timestamp).getTime()) / 1000));
          return (
            <div key={e.trace_id} onClick={() => onOpenTask(e.task_id)} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "9px 12px", borderBottom: "1px solid var(--border-subtle)", cursor: "pointer" }}
              onMouseEnter={(el) => { (el.currentTarget as HTMLElement).style.background = "var(--bg-elev)"; }}
              onMouseLeave={(el) => { (el.currentTarget as HTMLElement).style.background = ""; }}
            >
              <span style={{ color: "var(--sig-red)", marginTop: 2 }}>✕</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--fg)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.task_id}</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)" }}>{e.from_agent.replace(/^agt_/, "")}</div>
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)", flexShrink: 0 }}>{dt < 60 ? `${dt}s` : `${Math.floor(dt / 60)}m`} ago</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── SVG Icons ────────────────────────────────────────────────────────────────
function SearchIcon() {
  return <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="6"/><path d="m20 20-3.5-3.5"/></svg>;
}
function ChevronD() {
  return <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6"/></svg>;
}
function GraphIcon() {
  return <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M7.8 7.7 10.5 16.2"/><path d="M16.2 7.7 13.5 16.2"/><path d="M8.5 6h7"/></svg>;
}
function TimelineIcon() {
  return <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h18"/><path d="M3 16h18"/><circle cx="8" cy="8" r="1.6" fill="currentColor"/><circle cx="15" cy="16" r="1.6" fill="currentColor"/></svg>;
}
function MemoryIcon() {
  return <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><rect x="3.5" y="6" width="17" height="12" rx="1.5"/><path d="M3.5 10h17"/><path d="M7 14h2"/><path d="M11 14h2"/><path d="M15 14h2"/></svg>;
}
