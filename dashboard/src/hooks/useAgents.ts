/**
 * useAgents -- polls the registry REST API for the live agent list.
 *
 * Refreshes every 5 seconds so the dashboard stays current without
 * requiring a full page reload.
 */

import { useEffect, useState } from "react";
import type { AgentRecord } from "../types";

const REGISTRY_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 5000;

export function useAgents(): {
  agents: AgentRecord[];
  loading: boolean;
  error: string | null;
} {
  const [agents, setAgents] = useState<AgentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function fetchAgents() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/agents`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data: AgentRecord[] = await resp.json();
        if (active) {
          setAgents(data);
          setError(null);
        }
      } catch (err) {
        if (active) setError(String(err));
      } finally {
        if (active) setLoading(false);
      }
    }

    fetchAgents();
    const timer = setInterval(fetchAgents, POLL_INTERVAL_MS);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return { agents, loading, error };
}
