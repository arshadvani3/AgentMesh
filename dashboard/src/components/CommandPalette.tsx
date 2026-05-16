import { useEffect, useRef, useState } from "react";

interface PaletteItem {
  group: string;
  label: string;
  desc: string;
  action: () => void;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onAction: (kind: string, val: string) => void;
}

export default function CommandPalette({ open, onClose, onAction }: Props) {
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQ("");
      setActive(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const items: PaletteItem[] = [
    { group: "Navigation", label: "Go to Mesh Graph",     desc: "G then M", action: () => onAction("tab", "graph") },
    { group: "Navigation", label: "Go to Trace Timeline", desc: "G then T", action: () => onAction("tab", "timeline") },
    { group: "Navigation", label: "Go to Memory",         desc: "G then S", action: () => onAction("tab", "memory") },
    { group: "Actions",    label: "Send test request",    desc: "mesh.test_request",  action: () => onAction("toast", "Test request enqueued → agt_router_alpha") },
    { group: "Actions",    label: "Snapshot mesh state",  desc: "mesh.snapshot",      action: () => onAction("toast", "Snapshot saved") },
    { group: "Actions",    label: "Pause event stream",   desc: "mesh.pause_stream",  action: () => onAction("toast", "WebSocket stream paused") },
    { group: "Actions",    label: "Reload agent registry",desc: "registry.reload",    action: () => onAction("toast", "Registry reloaded") },
    { group: "Help",       label: "Keyboard shortcuts",   desc: "?",                  action: () => onAction("toast", "gm · gt · gs · ⌘K") },
  ];

  const filtered = q.trim()
    ? items.filter((it) => (it.label + " " + it.desc).toLowerCase().includes(q.toLowerCase()))
    : items;

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") { setActive((a) => Math.min(filtered.length - 1, a + 1)); e.preventDefault(); }
      if (e.key === "ArrowUp")   { setActive((a) => Math.max(0, a - 1)); e.preventDefault(); }
      if (e.key === "Enter")     { filtered[active]?.action(); onClose(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, filtered, active, onClose]);

  if (!open) return null;

  const grouped = filtered.reduce<Record<string, PaletteItem[]>>((acc, it) => {
    (acc[it.group] ??= []).push(it);
    return acc;
  }, {});

  let runIdx = 0;

  return (
    <div
      onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "oklch(0 0 0 / 0.55)", display: "flex", alignItems: "flex-start", justifyContent: "center", paddingTop: "12vh", zIndex: 50, backdropFilter: "blur(4px)" }}
    >
      <div onClick={(e) => e.stopPropagation()} style={{ width: 560, background: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "0 24px 64px oklch(0 0 0 / 0.6)", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderBottom: "1px solid var(--border-subtle)" }}>
          <SearchIcon />
          <input
            ref={inputRef}
            placeholder="Type a command or agent ID…"
            value={q}
            onChange={(e) => { setQ(e.target.value); setActive(0); }}
            style={{ flex: 1, fontSize: 14, color: "var(--fg)", background: "none", border: "none", outline: "none", fontFamily: "var(--font-sans)" }}
          />
          <Kbd>esc</Kbd>
        </div>
        <div style={{ maxHeight: 360, overflow: "auto", padding: 6 }}>
          {Object.entries(grouped).map(([g, list]) => (
            <div key={g}>
              <div style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 600, padding: "6px 10px 4px" }}>{g}</div>
              {list.map((it) => {
                const idx = runIdx++;
                const isActive = idx === active;
                return (
                  <div
                    key={it.label}
                    onMouseEnter={() => setActive(idx)}
                    onClick={() => { it.action(); onClose(); }}
                    style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 10px", borderRadius: "var(--radius)", cursor: "pointer", fontSize: 13, background: isActive ? "var(--bg-elev)" : "transparent", whiteSpace: "nowrap" }}
                  >
                    <span style={{ color: "var(--fg-tertiary)" }}><ChevronIcon /></span>
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>{it.label}</span>
                    <span style={{ color: "var(--fg-tertiary)", fontSize: 11, fontFamily: "var(--font-mono)", flexShrink: 0 }}>{it.desc}</span>
                  </div>
                );
              })}
            </div>
          ))}
          {filtered.length === 0 && (
            <div style={{ padding: "7px 10px", color: "var(--fg-tertiary)" }}>No matches.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, background: "var(--bg-base)", border: "1px solid var(--border-subtle)", borderRadius: 2, padding: "0 4px", height: 16, display: "inline-flex", alignItems: "center", color: "var(--fg-secondary)" }}>
      {children}
    </span>
  );
}

function SearchIcon() {
  return <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--fg-tertiary)", flexShrink: 0 }}><circle cx="11" cy="11" r="6"/><path d="m20 20-3.5-3.5"/></svg>;
}

function ChevronIcon() {
  return <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="m9 6 6 6-6 6"/></svg>;
}
