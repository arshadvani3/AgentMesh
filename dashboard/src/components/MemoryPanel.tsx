/**
 * MemoryPanel -- live inspector for AgentMemory workflow sessions.
 *
 * Polls GET /memory every 3s for active session IDs, then lets the
 * user click a session to expand its keys and values.
 */

import { ChevronDown, ChevronRight, Database, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { MemorySession } from "../types";

const REGISTRY_URL = import.meta.env.VITE_REGISTRY_URL ?? "http://localhost:8080";
const POLL_INTERVAL_MS = 3000;

function useMemorySessions(): { sessionIds: string[]; loading: boolean } {
  const [sessionIds, setSessionIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    async function fetchSessions() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/memory`);
        if (!resp.ok) return;
        const data: { sessions: string[] } = await resp.json();
        if (active) setSessionIds(data.sessions ?? []);
      } catch {
        // registry not running
      } finally {
        if (active) setLoading(false);
      }
    }

    fetchSessions();
    const timer = setInterval(fetchSessions, POLL_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return { sessionIds, loading };
}

function useSessionDetail(sessionId: string | null): MemorySession | null {
  const [session, setSession] = useState<MemorySession | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setSession(null);
      return;
    }

    let active = true;

    async function fetchDetail() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/memory/${sessionId}`);
        if (!resp.ok) return;
        const data: MemorySession = await resp.json();
        if (active) setSession(data);
      } catch {
        // ignore
      }
    }

    fetchDetail();
    const timer = setInterval(fetchDetail, POLL_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [sessionId]);

  return session;
}

// ---------------------------------------------------------------------------
// JSON tree renderer
// ---------------------------------------------------------------------------

function JsonTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const [open, setOpen] = useState(depth < 2);

  if (value === null || value === undefined) {
    return <span className="text-gray-500">null</span>;
  }

  if (typeof value === "boolean") {
    return (
      <span className={value ? "text-green-400" : "text-red-400"}>
        {String(value)}
      </span>
    );
  }

  if (typeof value === "number") {
    return <span className="text-yellow-300">{value}</span>;
  }

  if (typeof value === "string") {
    if (value.length > 200) {
      return (
        <span className="text-orange-300">
          "{value.slice(0, 200)}
          <span className="text-gray-500 italic">…({value.length} chars)</span>"
        </span>
      );
    }
    return <span className="text-orange-300">"{value}"</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0)
      return <span className="text-gray-500">[]</span>;
    return (
      <span>
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-gray-400 hover:text-white inline-flex items-center gap-0.5"
        >
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <span className="text-gray-500">[{value.length}]</span>
        </button>
        {open && (
          <div className="ml-4 border-l border-gray-700 pl-2 mt-0.5 space-y-0.5">
            {value.map((item, i) => (
              <div key={i} className="flex gap-1.5">
                <span className="text-gray-600 text-xs shrink-0">{i}:</span>
                <JsonTree value={item} depth={depth + 1} />
              </div>
            ))}
          </div>
        )}
      </span>
    );
  }

  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>);
    if (keys.length === 0)
      return <span className="text-gray-500">{"{}"}</span>;
    return (
      <span>
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-gray-400 hover:text-white inline-flex items-center gap-0.5"
        >
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <span className="text-gray-500">{"{" + keys.length + "}"}</span>
        </button>
        {open && (
          <div className="ml-4 border-l border-gray-700 pl-2 mt-0.5 space-y-0.5">
            {keys.map((k) => (
              <div key={k} className="flex gap-1.5 flex-wrap">
                <span className="text-cyan-300 text-xs shrink-0">{k}:</span>
                <JsonTree
                  value={(value as Record<string, unknown>)[k]}
                  depth={depth + 1}
                />
              </div>
            ))}
          </div>
        )}
      </span>
    );
  }

  return <span className="text-gray-400">{String(value)}</span>;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function MemoryPanel() {
  const { sessionIds, loading } = useMemorySessions();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const session = useSessionDetail(selectedId);

  return (
    <div className="h-full flex overflow-hidden text-sm">
      {/* Session list */}
      <div className="w-64 shrink-0 border-r border-gray-800 overflow-auto p-3">
        <div className="flex items-center gap-2 mb-3">
          <Database size={14} className="text-cyan-400" />
          <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
            Active Sessions
          </span>
          {loading && (
            <Loader2 size={12} className="text-gray-500 animate-spin ml-auto" />
          )}
        </div>

        {!loading && sessionIds.length === 0 && (
          <p className="text-gray-600 text-xs italic">
            No active sessions. Run the demo to see workflow state here.
          </p>
        )}

        <div className="space-y-1">
          {sessionIds.map((sid) => (
            <button
              key={sid}
              onClick={() => setSelectedId(sid === selectedId ? null : sid)}
              className={`w-full text-left px-2 py-1.5 rounded text-xs font-mono truncate transition-colors ${
                selectedId === sid
                  ? "bg-cyan-900 text-cyan-200"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
              }`}
              title={sid}
            >
              {sid.length > 28 ? sid.slice(0, 28) + "…" : sid}
            </button>
          ))}
        </div>
      </div>

      {/* Session detail */}
      <div className="flex-1 overflow-auto p-4">
        {!selectedId && (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            Select a session to inspect its workflow state
          </div>
        )}

        {selectedId && !session && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 size={14} className="animate-spin" />
            Loading…
          </div>
        )}

        {session && (
          <div className="space-y-4">
            <div>
              <h2 className="text-white font-semibold text-sm mb-0.5">
                Session
              </h2>
              <p className="text-gray-500 font-mono text-xs">{session.session_id}</p>
            </div>

            <div className="flex gap-2 flex-wrap">
              {session.keys.map((k) => (
                <span
                  key={k}
                  className="bg-gray-800 text-cyan-300 text-xs px-2 py-0.5 rounded font-mono"
                >
                  {k}
                </span>
              ))}
            </div>

            <div className="space-y-3">
              {session.keys.map((k) => (
                <div key={k} className="bg-gray-900 rounded-lg p-3">
                  <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-2 font-mono">
                    {k}
                  </h3>
                  <div className="text-xs font-mono leading-relaxed">
                    <JsonTree value={session.data[k]} depth={0} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
