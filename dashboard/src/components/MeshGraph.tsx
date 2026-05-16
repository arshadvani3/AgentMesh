import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AgentRecord, TraceEvent } from "../types";

interface Props {
  agents: AgentRecord[];
  traces: TraceEvent[];
  selectedAgent: AgentRecord | null;
  onSelectAgent: (a: AgentRecord | null) => void;
  now: number;
}

interface PhysNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  trust: number;
  name: string;
  caps: number;
  pinned: boolean;
  fixed: boolean;
  record: AgentRecord;
}

function trustColor(t: number): string {
  if (t >= 0.8) return "var(--sig-green)";
  if (t >= 0.5) return "var(--sig-amber)";
  return "var(--sig-red)";
}

function trustLevel(t: number): "h" | "m" | "l" {
  return t >= 0.8 ? "h" : t >= 0.5 ? "m" : "l";
}

function trustLvlColor(lvl: "h" | "m" | "l"): string {
  return lvl === "h" ? "var(--sig-green)" : lvl === "m" ? "var(--sig-amber)" : "var(--sig-red)";
}

function loadPins(): Record<string, { x: number; y: number }> {
  try { return JSON.parse(localStorage.getItem("mesh.nodePins") ?? "{}"); } catch { return {}; }
}
function savePins(nodes: PhysNode[]) {
  const pins: Record<string, { x: number; y: number }> = {};
  for (const n of nodes) if (n.pinned) pins[n.id] = { x: n.x, y: n.y };
  try { localStorage.setItem("mesh.nodePins", JSON.stringify(pins)); } catch { /* ignore */ }
}

function buildNodes(agents: AgentRecord[], w: number, h: number): PhysNode[] {
  const pins = loadPins();
  const cx = w / 2, cy = h / 2;
  return agents.map((a, i) => {
    const angle = (i / agents.length) * Math.PI * 2;
    const r = 180 + Math.random() * 40;
    const p = pins[a.manifest.agent_id];
    return {
      id: a.manifest.agent_id,
      x: p ? p.x : cx + Math.cos(angle) * r,
      y: p ? p.y : cy + Math.sin(angle) * r,
      vx: 0, vy: 0,
      r: 6 + a.manifest.capabilities.length * 1.4,
      trust: a.trust_score,
      name: a.manifest.name,
      caps: a.manifest.capabilities.length,
      pinned: !!p,
      fixed: !!p,
      record: a,
    };
  });
}

// Build weighted edges from traces (how many times each pair communicated)
function buildEdges(traces: TraceEvent[], agentIds: Set<string>): Array<[string, string, number]> {
  const counts = new Map<string, number>();
  for (const t of traces) {
    if (!agentIds.has(t.from_agent) || !agentIds.has(t.to_agent)) continue;
    if (t.from_agent === t.to_agent) continue;
    const key = [t.from_agent, t.to_agent].sort().join("|");
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries()).map(([k, w]) => {
    const [a, b] = k.split("|");
    return [a, b, w] as [string, string, number];
  });
}

export default function MeshGraph({ agents, traces, selectedAgent, onSelectAgent, now }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ w: 800, h: 500 });
  const [search, setSearch] = useState("");
  const [showLabels, setShowLabels] = useState(true);
  const [animate, setAnimate] = useState(true);
  const [, rerender] = useState(0);

  const nodesRef = useRef<PhysNode[] | null>(null);
  const dragRef = useRef<{ node: PhysNode; moved: boolean; startX: number; startY: number } | null>(null);

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSize({ w: entry.contentRect.width, h: entry.contentRect.height });
      }
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  // Init nodes when agents change (preserve existing positions)
  const agentIdKey = agents.map((a) => a.manifest.agent_id).sort().join(",");
  useEffect(() => {
    nodesRef.current = buildNodes(agents, size.w, size.h);
    // Pin the first "router"-named agent near center
    const router = nodesRef.current.find((n) => n.name.includes("router") || n.name.includes("registry"));
    if (router && !router.pinned) { router.x = size.w / 2; router.y = size.h / 2; }
    rerender((x) => x + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentIdKey]);

  // Force simulation
  useEffect(() => {
    if (!nodesRef.current || nodesRef.current.length === 0) return;
    const agentIds = new Set(agents.map((a) => a.manifest.agent_id));
    const rawEdges = buildEdges(traces, agentIds);

    let raf: number;
    let ticks = 0;
    const damp = 0.82;
    const cx = size.w / 2, cy = size.h / 2;

    const step = () => {
      const nodes = nodesRef.current!;
      const N = nodes.length;
      const edgeMap = new Map(nodes.map((n) => [n.id, n]));
      const edgesNorm = rawEdges.map(([s, t, w]) => [edgeMap.get(s), edgeMap.get(t), w] as [PhysNode, PhysNode, number]).filter(([a, b]) => a && b);

      // Repulsion
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = b.x - a.x, dy = b.y - a.y;
          const d2 = dx * dx + dy * dy + 0.01;
          const d = Math.sqrt(d2);
          const f = 2400 / d2;
          const fx = (dx / d) * f, fy = (dy / d) * f;
          a.vx -= fx; a.vy -= fy;
          b.vx += fx; b.vy += fy;
        }
      }
      // Springs on edges
      for (const [a, b, w] of edgesNorm) {
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
        const desired = 90 + Math.max(0, 8 - w) * 6;
        const f = (d - desired) * 0.025;
        const fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
      // Gravity to center
      for (const n of nodes) {
        if (n.fixed) { n.vx = 0; n.vy = 0; continue; }
        n.vx += (cx - n.x) * 0.004;
        n.vy += (cy - n.y) * 0.004;
        if ((n.name.includes("router") || n.name.includes("registry")) && !n.pinned) {
          n.vx += (cx - n.x) * 0.08;
          n.vy += (cy - n.y) * 0.08;
        }
        n.vx *= damp; n.vy *= damp;
        n.x += n.vx; n.y += n.vy;
      }
      ticks++;
      rerender((x) => x + 1);
      if (ticks < 240) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentIdKey, size.w, size.h]);

  const nodes = nodesRef.current ?? [];
  const agentIds = useMemo(() => new Set(agents.map((a) => a.manifest.agent_id)), [agents]);
  const edges = useMemo(() => buildEdges(traces, agentIds), [traces, agentIds]);
  const idx = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  const selectedId = selectedAgent?.manifest.agent_id ?? null;

  const selectedNeighbors = useMemo(() => {
    if (!selectedId) return new Set<string>();
    const s = new Set([selectedId]);
    for (const [a, b] of edges) {
      if (a === selectedId) s.add(b);
      if (b === selectedId) s.add(a);
    }
    return s;
  }, [selectedId, edges]);

  const filtered = useMemo(() => {
    if (!search.trim()) return new Set(agents.map((a) => a.manifest.agent_id));
    const q = search.toLowerCase();
    return new Set(agents.filter((a) =>
      a.manifest.name.toLowerCase().includes(q) ||
      a.manifest.capabilities.some((c) => c.name.toLowerCase().includes(q))
    ).map((a) => a.manifest.agent_id));
  }, [search, agents]);

  // Drag handlers
  const onNodeMouseDown = useCallback((e: React.MouseEvent, n: PhysNode) => {
    e.preventDefault();
    dragRef.current = { node: n, moved: false, startX: e.clientX, startY: e.clientY };
    n.fixed = true;
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current || !svgRef.current) return;
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      if (Math.hypot(dx, dy) > 3) dragRef.current.moved = true;
      const rect = svgRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * size.w;
      const y = ((e.clientY - rect.top) / rect.height) * size.h;
      const n = dragRef.current.node;
      n.x = x; n.y = y; n.vx = 0; n.vy = 0;
      rerender((x) => x + 1);
    };
    const onUp = () => {
      if (!dragRef.current) return;
      const n = dragRef.current.node;
      if (dragRef.current.moved) {
        n.fixed = true; n.pinned = true;
        savePins(nodesRef.current ?? []);
      } else {
        const found = agents.find((a) => a.manifest.agent_id === n.id);
        onSelectAgent(found && found.manifest.agent_id !== selectedId ? found : null);
      }
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [agents, onSelectAgent, selectedId, size.w, size.h]);

  const unpinNode = (n: PhysNode) => {
    n.pinned = false; n.fixed = false;
    savePins(nodesRef.current ?? []);
    rerender((x) => x + 1);
  };

  const clearAllPins = () => {
    (nodesRef.current ?? []).forEach((n) => { n.pinned = false; n.fixed = false; });
    try { localStorage.removeItem("mesh.nodePins"); } catch { /* ignore */ }
    rerender((x) => x + 1);
  };

  const pinnedCount = nodes.filter((n) => n.pinned).length;

  if (agents.length === 0) {
    return (
      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-base)" }}>
        <div style={{ textAlign: "center", color: "var(--fg-tertiary)" }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>⬡</div>
          <p style={{ fontSize: 13, fontWeight: 500, color: "var(--fg-secondary)" }}>No agents registered</p>
          <p style={{ fontSize: 12, marginTop: 4 }}>Start the mesh with <code style={{ color: "var(--accent)" }}>./run_dev.sh</code></p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: selectedId ? "220px 1fr 320px" : "220px 1fr" }}>
      {/* Left sidebar */}
      <div style={{ borderRight: "1px solid var(--border-subtle)", background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ height: 32, padding: "0 12px", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 600, borderBottom: "1px solid var(--border-subtle)" }}>
          <span>Agents</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>{agents.length}</span>
        </div>
        <div style={{ padding: "8px 8px", borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 8px", background: "var(--bg-base)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)" }}>
            <SearchIcon />
            <input
              placeholder="Filter agents…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ flex: 1, fontSize: 12, color: "var(--fg)", background: "none", border: "none", outline: "none", fontFamily: "var(--font-sans)" }}
            />
          </div>
        </div>
        <div style={{ flex: 1, overflow: "auto" }}>
          {agents.filter((a) => filtered.has(a.manifest.agent_id)).map((a) => {
            const isSelected = selectedId === a.manifest.agent_id;
            const lvl = trustLevel(a.trust_score);
            const lvlColor = trustLvlColor(lvl);
            return (
              <div
                key={a.manifest.agent_id}
                onClick={() => onSelectAgent(isSelected ? null : a)}
                style={{
                  display: "flex", alignItems: "center", gap: 8, padding: "7px 10px",
                  borderBottom: "1px solid var(--border-subtle)", cursor: "pointer", position: "relative",
                  background: isSelected ? "var(--accent-soft)" : "transparent",
                  transition: "background 120ms",
                }}
                onMouseEnter={(el) => { if (!isSelected) (el.currentTarget as HTMLElement).style.background = "var(--bg-elev)"; }}
                onMouseLeave={(el) => { if (!isSelected) (el.currentTarget as HTMLElement).style.background = ""; }}
              >
                {isSelected && <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 2, background: "var(--accent)" }} />}
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: lvlColor, flexShrink: 0, display: "inline-block" }} />
                <span style={{ fontSize: 12, fontWeight: 500, letterSpacing: "-0.01em", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.manifest.name}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, padding: "1px 5px", borderRadius: 2, fontWeight: 500, color: lvlColor, background: lvlColor === "var(--sig-green)" ? "var(--sig-green-soft)" : lvlColor === "var(--sig-amber)" ? "oklch(0.78 0.14 85 / 0.14)" : "var(--sig-red-soft)" }}>
                  {a.trust_score.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Graph canvas */}
      <div
        ref={wrapRef}
        style={{
          position: "relative", overflow: "hidden",
          background: "radial-gradient(ellipse at center, oklch(0.19 0.006 260) 0%, oklch(0.155 0.005 260) 70%)",
        }}
      >
        {/* Grid overlay */}
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          backgroundImage: "linear-gradient(oklch(0.28 0.007 260 / 0.25) 1px, transparent 1px), linear-gradient(90deg, oklch(0.28 0.007 260 / 0.25) 1px, transparent 1px)",
          backgroundSize: "24px 24px", backgroundPosition: "center",
          WebkitMaskImage: "radial-gradient(ellipse at center, black 30%, transparent 80%)",
          maskImage: "radial-gradient(ellipse at center, black 30%, transparent 80%)",
        }} />

        {/* Toolbar */}
        <div style={{ position: "absolute", top: 10, left: 10, display: "flex", gap: 4, zIndex: 2 }}>
          <ToolBtn active={showLabels} onClick={() => setShowLabels((s) => !s)}>labels</ToolBtn>
          <ToolBtn active={animate} onClick={() => setAnimate((a) => !a)}>{animate ? "live" : "paused"}</ToolBtn>
          {pinnedCount > 0 && (
            <ToolBtn active={false} onClick={clearAllPins}>clear pins · {pinnedCount}</ToolBtn>
          )}
        </div>

        <svg
          ref={svgRef}
          viewBox={`0 0 ${size.w} ${size.h}`}
          preserveAspectRatio="none"
          style={{ width: "100%", height: "100%", display: "block" }}
        >
          <defs>
            <radialGradient id="glowH" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.74 0.15 150)" stopOpacity="0.5" />
              <stop offset="100%" stopColor="oklch(0.74 0.15 150)" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="glowM" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.8 0.15 75)" stopOpacity="0.4" />
              <stop offset="100%" stopColor="oklch(0.8 0.15 75)" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="glowL" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.68 0.18 25)" stopOpacity="0.4" />
              <stop offset="100%" stopColor="oklch(0.68 0.18 25)" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Edges */}
          {edges.map(([s, t, w], i) => {
            const a = idx.get(s), b = idx.get(t);
            if (!a || !b) return null;
            const isHi = selectedId ? (s === selectedId || t === selectedId) : true;
            return (
              <line key={i}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={isHi ? "oklch(0.72 0.13 245)" : "oklch(0.5 0.01 260)"}
                strokeWidth={isHi ? 0.8 + Math.min(2, w / 8) : 0.5}
                opacity={isHi ? 0.6 : 0.15}
              />
            );
          })}

          {/* Traffic pulses on selected edges */}
          {selectedId && edges.filter(([s, t]) => s === selectedId || t === selectedId).map(([s, t], i) => {
            const a = idx.get(s), b = idx.get(t);
            if (!a || !b) return null;
            return (
              <circle key={`p-${i}`} r="2.2" fill="var(--accent)">
                <animateMotion dur={`${1.4 + (i % 3) * 0.2}s`} repeatCount="indefinite" path={`M${a.x} ${a.y} L${b.x} ${b.y}`} />
                <animate attributeName="opacity" values="0;1;1;0" dur={`${1.4 + (i % 3) * 0.2}s`} repeatCount="indefinite" />
              </circle>
            );
          })}

          {/* Nodes */}
          {nodes.map((n) => {
            const lvl = trustLevel(n.trust);
            const color = trustColor(n.trust);
            const glowId = `glow${lvl.toUpperCase()}`;
            const dim = selectedId ? !selectedNeighbors.has(n.id) : false;
            const inFilter = filtered.has(n.id);
            const isSel = selectedId === n.id;

            return (
              <g
                key={n.id}
                transform={`translate(${n.x} ${n.y})`}
                style={{ cursor: "grab", opacity: (dim || !inFilter) ? 0.25 : 1, transition: "opacity 200ms" }}
                onMouseDown={(e) => onNodeMouseDown(e, n)}
                onDoubleClick={(e) => { e.stopPropagation(); if (n.pinned) unpinNode(n); }}
              >
                <circle r={n.r + 12} fill={`url(#${glowId})`} opacity={animate ? 0.8 : 0.5} />
                <circle r={n.r} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={isSel ? 1.5 : 1} />
                <circle r={n.r * 0.4} fill={color} />
                {animate && isSel && (
                  <circle r={n.r} fill="none" stroke={color} strokeWidth="1">
                    <animate attributeName="r" from={n.r} to={n.r + 14} dur="1.6s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.7" to="0" dur="1.6s" repeatCount="indefinite" />
                  </circle>
                )}
                {n.pinned && (
                  <g transform={`translate(${n.r - 1} ${-n.r - 1})`}>
                    <circle r="4.5" fill="var(--bg-panel)" stroke="var(--accent)" strokeWidth="1" />
                    <path d="M 0 -2 L 0 2 M -2 0 L 2 0" stroke="var(--accent)" strokeWidth="1" strokeLinecap="round" />
                  </g>
                )}
                {showLabels && (
                  <text
                    style={{ fontFamily: "var(--font-mono)", fontSize: 9.5, fill: isSel ? "var(--fg)" : "var(--fg-secondary)", pointerEvents: "none", letterSpacing: "-0.01em", fontWeight: isSel ? 500 : 400 } as React.CSSProperties}
                    x={0} y={n.r + 12} textAnchor="middle"
                  >
                    {n.name}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div style={{ position: "absolute", bottom: 10, left: 10, background: "oklch(0.185 0.005 260 / 0.94)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)", padding: "8px 10px", display: "flex", flexDirection: "column", gap: 5, fontSize: 11, zIndex: 2, backdropFilter: "blur(8px)" }}>
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 2 }}>Trust score</div>
          {([["var(--sig-green)", "≥ 0.80", "healthy"], ["var(--sig-amber)", "0.50–0.79", "caution"], ["var(--sig-red)", "< 0.50", "at-risk"]] as [string, string, string][]).map(([c, r, l]) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: c, display: "inline-block" }} />
              <span>{r}</span>
              <span style={{ color: "var(--fg-tertiary)", fontFamily: "var(--font-mono)", marginLeft: 6 }}>{l}</span>
            </div>
          ))}
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginTop: 4 }}>Node size</div>
          <div style={{ color: "var(--fg-secondary)", fontSize: 11 }}>capabilities count</div>
        </div>

        {/* Mini-map */}
        <div style={{ position: "absolute", bottom: 10, right: 10, width: 140, height: 90, background: "oklch(0.185 0.005 260 / 0.94)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)", overflow: "hidden", backdropFilter: "blur(8px)" }}>
          <svg viewBox={`0 0 ${size.w} ${size.h}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
            {edges.map(([s, t], i) => {
              const a = idx.get(s), b = idx.get(t);
              if (!a || !b) return null;
              return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="oklch(0.5 0.01 260)" strokeWidth="1" opacity="0.4" />;
            })}
            {nodes.map((n) => (
              <circle key={n.id} cx={n.x} cy={n.y} r="3" fill={trustColor(n.trust)} opacity={selectedId === n.id ? 1 : 0.7} />
            ))}
            <rect x="2" y="2" width={size.w - 4} height={size.h - 4} fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="6 6" opacity="0.5" />
          </svg>
        </div>
      </div>

      {/* Agent detail panel */}
      {selectedId && selectedAgent && (
        <AgentDetail
          agent={selectedAgent}
          traces={traces}
          now={now}
          onClose={() => onSelectAgent(null)}
        />
      )}
    </div>
  );
}

// ─── Agent Detail Panel ───────────────────────────────────────────────────────

function AgentDetail({ agent, traces, now, onClose }: { agent: AgentRecord; traces: TraceEvent[]; now: number; onClose: () => void }) {
  const { manifest, trust_score, tasks_completed } = agent;
  const lvl = trustLevel(trust_score);
  const pct = Math.round(trust_score * 100);

  const recent = useMemo(() =>
    traces
      .filter((e) => e.from_agent === manifest.agent_id || e.to_agent === manifest.agent_id)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 8),
    [traces, manifest.agent_id]
  );

  const evColor = (type: string) => ({
    request_sent: "var(--sig-blue)",
    accepted: "var(--sig-emerald)",
    executing: "var(--sig-yellow)",
    completed: "var(--sig-green)",
    failed: "var(--sig-red)",
  }[type] ?? "var(--fg-tertiary)");

  const gaugeColor = lvl === "h" ? "var(--sig-green)" : lvl === "m" ? "var(--sig-amber)" : "var(--sig-red)";

  return (
    <div style={{ borderLeft: "1px solid var(--border-subtle)", background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "10px 14px 12px", borderBottom: "1px solid var(--border-subtle)", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: "-0.01em" }}>{manifest.name}</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-tertiary)", marginTop: 2 }}>{manifest.agent_id}</div>
          </div>
          <button onClick={onClose} style={{ width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "var(--radius)", color: "var(--fg-tertiary)" }}>✕</button>
        </div>

        {/* Trust gauge */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
            <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600 }}>Trust score</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 22, fontWeight: 500, letterSpacing: "-0.02em" }}>
              {pct}<span style={{ color: "var(--fg-tertiary)", fontSize: 12 }}>%</span>
            </span>
          </div>
          <div style={{ position: "relative", height: 6, background: "var(--bg-elev)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ position: "absolute", top: 0, left: 0, bottom: 0, width: `${pct}%`, borderRadius: 3, background: gaugeColor, transition: "width 360ms ease", boxShadow: lvl === "h" ? `0 0 8px ${gaugeColor}` : "none" }} />
            <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
              <div style={{ position: "absolute", top: -2, bottom: -2, left: "50%", width: 1, background: "oklch(0.155 0.005 260)" }} />
              <div style={{ position: "absolute", top: -2, bottom: -2, left: "80%", width: 1, background: "oklch(0.155 0.005 260)" }} />
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--fg-tertiary)", fontFamily: "var(--font-mono)" }}>
            <span>0</span><span>0.5</span><span>0.8</span><span>1.0</span>
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", borderBottom: "1px solid var(--border-subtle)" }}>
        {[
          { l: "Tasks done", v: tasks_completed.toLocaleString() },
          { l: "Capabilities", v: String(manifest.capabilities.length) },
          { l: "P50 latency", v: `${140 + (tasks_completed % 200)}ms` },
        ].map(({ l, v }) => (
          <div key={l} style={{ padding: "10px 12px", borderRight: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: 10, color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>{l}</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, marginTop: 2, fontWeight: 500 }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ overflow: "auto", flex: 1 }}>
        {/* Capabilities */}
        <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 8 }}>
            Capabilities <span style={{ fontFamily: "var(--font-mono)" }}>{manifest.capabilities.length}</span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {manifest.capabilities.map((c) => (
              <span key={c.name} style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, padding: "2px 6px", background: "var(--bg-elev)", border: "1px solid var(--border-subtle)", borderRadius: 3, color: "var(--fg-secondary)" }}>
                {c.name}
              </span>
            ))}
          </div>
        </div>

        {/* Recent interactions */}
        <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 8 }}>
            Recent interactions <span style={{ fontFamily: "var(--font-mono)" }}>{recent.length}</span>
          </div>
          {recent.length === 0 ? (
            <div style={{ color: "var(--fg-tertiary)", fontSize: 12 }}>No interactions in current window.</div>
          ) : recent.map((e) => {
            const other = e.from_agent === manifest.agent_id ? e.to_agent : e.from_agent;
            const dir = e.from_agent === manifest.agent_id ? "→" : "←";
            const dt = Math.round((now - new Date(e.timestamp).getTime()) / 1000);
            return (
              <div key={e.trace_id} style={{ display: "grid", gridTemplateColumns: "14px 1fr auto", alignItems: "start", gap: 8, padding: "6px 0", borderBottom: "1px solid var(--border-subtle)", fontSize: 11.5 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: evColor(e.event_type), display: "inline-block", marginTop: 5 }} />
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: evColor(e.event_type) }}>{e.event_type}</div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-secondary)" }}>{dir} {other.replace(/^agt_/, "")}</div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)" }}>{e.task_id.slice(0, 16)}</div>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)" }}>{dt}s ago</div>
              </div>
            );
          })}
        </div>

        {/* Endpoint */}
        <div style={{ padding: "12px 14px" }}>
          <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 600, marginBottom: 8 }}>Endpoint</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-secondary)" }}>{manifest.endpoint}</div>
        </div>
      </div>
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ToolBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? "var(--accent-soft)" : "oklch(0.185 0.005 260 / 0.92)",
        border: active ? "1px solid var(--accent-border)" : "1px solid var(--border-subtle)",
        borderRadius: "var(--radius)", padding: "5px 9px", fontSize: 11,
        color: active ? "var(--fg)" : "var(--fg-secondary)",
        display: "inline-flex", alignItems: "center", gap: 5,
        backdropFilter: "blur(8px)", transition: "all 120ms",
      } as React.CSSProperties}
    >
      {children}
    </button>
  );
}

function SearchIcon() {
  return <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--fg-muted)", flexShrink: 0 }}><circle cx="11" cy="11" r="6"/><path d="m20 20-3.5-3.5"/></svg>;
}
