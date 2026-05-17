import { useEffect, useRef, useState } from "react";
import type { TraceEvent } from "../types";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8080/ws/dashboard";
const REGISTRY_URL = import.meta.env.VITE_REGISTRY_URL ?? "http://localhost:8080";
const RECONNECT_DELAY_MS = 3000;

export function useDashboardSocket(): TraceEvent[] {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seenIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    let active = true;

    async function fetchHistory() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/traces?limit=200`);
        if (!resp.ok) return;
        // Registry returns a flat array, not { traces: [...] }
        const historical: TraceEvent[] = await resp.json();
        if (active && historical.length > 0) {
          setEvents(historical.slice(-500));
          for (const e of historical) seenIds.current.add(e.trace_id);
        }
      } catch {
        // registry not up yet
      }
    }

    function connect() {
      if (!active) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (msg) => {
        try {
          const event: TraceEvent = JSON.parse(msg.data);
          if (seenIds.current.has(event.trace_id)) return;
          seenIds.current.add(event.trace_id);
          setEvents((prev) => [...prev.slice(-499), event]);
        } catch {
          // non-JSON keep-alive frame
        }
      };

      ws.onclose = () => {
        if (active) reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      };
      ws.onerror = () => ws.close();
    }

    fetchHistory().then(() => connect());

    return () => {
      active = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return events;
}
