/**
 * useDashboardSocket -- subscribes to the registry WebSocket dashboard stream.
 *
 * Returns an array of TraceEvents received since the hook mounted.
 * Reconnects automatically on disconnect.
 */

import { useEffect, useRef, useState } from "react";
import type { TraceEvent } from "../types";

const WS_URL = "ws://localhost:8000/ws/dashboard";
const RECONNECT_DELAY_MS = 3000;

export function useDashboardSocket(): TraceEvent[] {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let active = true;

    function connect() {
      if (!active) return;

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (msg) => {
        try {
          const event: TraceEvent = JSON.parse(msg.data);
          setEvents((prev) => [...prev.slice(-499), event]);
        } catch {
          // Non-JSON keep-alive frames -- ignore
        }
      };

      ws.onclose = () => {
        if (active) {
          reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      active = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return events;
}
