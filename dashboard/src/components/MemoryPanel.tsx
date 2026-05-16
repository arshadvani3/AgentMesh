import { useEffect, useRef, useState } from "react";
import { JsonTree } from "./JsonTree";
import type { MemorySession } from "../types";

const REGISTRY_URL = import.meta.env.VITE_REGISTRY_URL ?? "http://localhost:8080";
const POLL_MS = 3000;

function useMemorySessions() {
  const [sessionIds, setSessionIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const deletedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    let active = true;
    async function fetch_() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/memory`);
        if (!resp.ok) return;
        const data: { sessions: string[] } = await resp.json();
        if (active) setSessionIds((data.sessions ?? []).filter((s) => !deletedRef.current.has(s)));
      } catch { /* registry not running */ }
      finally { if (active) setLoading(false); }
    }
    fetch_();
    const t = setInterval(fetch_, POLL_MS);
    return () => { active = false; clearInterval(t); };
  }, []);

  async function clearSession(id: string) {
    deletedRef.current.add(id);
    setSessionIds((prev) => prev.filter((s) => s !== id));
    try { await fetch(`${REGISTRY_URL}/memory/${id}`, { method: "DELETE" }); } catch { /* ignore */ }
  }

  return { sessionIds, loading, clearSession };
}

function useSessionDetail(sessionId: string | null): MemorySession | null {
  const [session, setSession] = useState<MemorySession | null>(null);
  useEffect(() => {
    if (!sessionId) { setSession(null); return; }
    let active = true;
    async function fetch_() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/memory/${sessionId}`);
        if (!resp.ok) return;
        const data: MemorySession = await resp.json();
        if (active) setSession(data);
      } catch { /* ignore */ }
    }
    fetch_();
    const t = setInterval(fetch_, POLL_MS);
    return () => { active = false; clearInterval(t); };
  }, [sessionId]);
  return session;
}

export default function MemoryPanel() {
  const { sessionIds, loading, clearSession } = useMemorySessions();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const session = useSessionDetail(selectedId);

  const filtered = q.trim()
    ? sessionIds.filter((s) => s.toLowerCase().includes(q.toLowerCase()))
    : sessionIds;

  return (
    <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: "260px 1fr" }}>
      {/* Session list */}
      <div style={{ borderRight: "1px solid var(--border-subtle)", background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ height: 32, padding: "0 12px", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 600, borderBottom: "1px solid var(--border-subtle)" }}>
          <span>Sessions</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>{loading ? "…" : sessionIds.length}</span>
        </div>

        {/* Search */}
        <div style={{ padding: "8px 8px", borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 8px", background: "var(--bg-base)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)" }}>
            <SearchIcon />
            <input
              placeholder="Search sessions…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              style={{ flex: 1, fontSize: 12, color: "var(--fg)", background: "none", border: "none", outline: "none", fontFamily: "var(--font-sans)" }}
            />
          </div>
        </div>

        <div style={{ flex: 1, overflow: "auto" }}>
          {!loading && filtered.length === 0 && (
            <div style={{ padding: "40px 16px", textAlign: "center", color: "var(--fg-tertiary)", fontSize: 12 }}>
              {q ? `No sessions match "${q}"` : "No sessions. Run the demo to see workflow state."}
            </div>
          )}
          {filtered.map((sid) => {
            const isSelected = sid === selectedId;
            return (
              <div
                key={sid}
                onClick={() => setSelectedId(sid === selectedId ? null : sid)}
                style={{
                  padding: "9px 12px", borderBottom: "1px solid var(--border-subtle)", cursor: "pointer",
                  display: "flex", flexDirection: "column", gap: 3, position: "relative",
                  background: isSelected ? "var(--accent-soft)" : "transparent",
                  transition: "background 120ms",
                }}
                onMouseEnter={(el) => { if (!isSelected) (el.currentTarget as HTMLElement).style.background = "var(--bg-elev)"; }}
                onMouseLeave={(el) => { if (!isSelected) (el.currentTarget as HTMLElement).style.background = ""; }}
              >
                {isSelected && <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 2, background: "var(--accent)" }} />}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {sid.length > 22 ? "…" + sid.slice(-22) : sid}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); if (selectedId === sid) setSelectedId(null); clearSession(sid); }}
                    style={{ color: "var(--fg-muted)", fontSize: 10, flexShrink: 0, padding: 2, borderRadius: 2 }}
                    title="Delete session"
                  >
                    ✕
                  </button>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)" }}>
                  {sid.slice(0, 8)}…
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Session inspector */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {!selectedId ? (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, color: "var(--fg-tertiary)" }}>
            <div style={{ width: 44, height: 44, border: "1px dashed var(--border)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--fg-muted)" }}>
              <MemoryIcon />
            </div>
            <div style={{ fontSize: 13, color: "var(--fg-secondary)", fontWeight: 500 }}>Select a session</div>
            <div style={{ fontSize: 12, maxWidth: 280, lineHeight: 1.5, textAlign: "center" }}>Pick a workflow session on the left to inspect its in-memory state.</div>
          </div>
        ) : !session ? (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--fg-tertiary)", fontSize: 13 }}>
            Loading…
          </div>
        ) : (
          <>
            <div style={{ height: 44, display: "flex", alignItems: "center", padding: "0 16px", borderBottom: "1px solid var(--border-subtle)", gap: 12, background: "var(--bg-panel)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--fg-tertiary)", fontSize: 11 }}>
                <span>sessions</span>
                <ChevronIcon />
                <span style={{ color: "var(--fg-secondary)" }}>session</span>
                <ChevronIcon />
              </div>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 500 }}>{session.session_id}</span>
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)" }}>
                  {session.keys.length} keys
                </span>
                <button style={{ background: "var(--bg-elev)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius)", padding: "5px 9px", fontSize: 11, color: "var(--fg-secondary)", display: "inline-flex", alignItems: "center", gap: 5 }}>
                  <CopyIcon /> snapshot
                </button>
              </div>
            </div>

            <div style={{ flex: 1, overflow: "auto", padding: "14px 16px", display: "flex", flexDirection: "column", gap: 14 }}>
              {session.keys.map((k) => (
                <MemKey key={k} keyName={k} value={session.data[k]} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function MemKey({ keyName, value }: { keyName: string; value: unknown }) {
  const [open, setOpen] = useState(true);

  const typeLabel = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;

  let stat = "";
  if (value === null) stat = "empty";
  else if (Array.isArray(value)) stat = `${(value as unknown[]).length} items`;
  else if (typeof value === "object") stat = `${Object.keys(value as object).length} keys`;
  else if (typeof value === "string") stat = `${(value as string).length} chars`;
  else stat = String(value);

  return (
    <div style={{
      border: open ? "1px solid var(--border-subtle)" : "1px dashed var(--border-subtle)",
      borderRadius: "var(--radius-lg)",
      background: open ? "var(--bg-panel)" : "transparent",
      overflow: "hidden",
      transition: "border-color 120ms",
    }}>
      <div
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex", alignItems: "center", padding: open ? "9px 12px" : "7px 12px",
          borderBottom: open ? "1px solid var(--border-subtle)" : "none",
          background: open ? "var(--bg-elev)" : "transparent",
          cursor: "pointer", gap: 10,
        }}
      >
        <span style={{ color: "var(--fg-tertiary)", transition: "transform 120ms", display: "inline-block", transform: open ? "none" : "rotate(-90deg)" }}>▾</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 12.5, fontWeight: 500, color: open ? "var(--fg)" : "var(--fg-secondary)", flex: 1 }}>{keyName}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-tertiary)", padding: "1px 5px", border: "1px solid var(--border-subtle)", borderRadius: 3 }}>{typeLabel}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, color: "var(--fg-tertiary)" }}>{stat}</span>
      </div>
      {open && (
        <div style={{ padding: "10px 14px", maxHeight: 420, overflow: "auto" }}>
          {value === null ? (
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, color: "var(--fg-muted)", fontStyle: "italic" }}>
              null <span style={{ fontStyle: "normal" }}>(not yet computed)</span>
            </span>
          ) : (
            <JsonTree data={value} defaultOpen={true} />
          )}
        </div>
      )}
    </div>
  );
}

function SearchIcon() {
  return <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--fg-muted)", flexShrink: 0 }}><circle cx="11" cy="11" r="6"/><path d="m20 20-3.5-3.5"/></svg>;
}
function ChevronIcon() {
  return <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="m9 6 6 6-6 6"/></svg>;
}
function MemoryIcon() {
  return <svg width={20} height={20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><rect x="3.5" y="6" width="17" height="12" rx="1.5"/><path d="M3.5 10h17"/><path d="M7 14h2"/><path d="M11 14h2"/><path d="M15 14h2"/></svg>;
}
function CopyIcon() {
  return <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="8" width="12" height="12" rx="1.5"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>;
}
