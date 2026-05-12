/**
 * useStats -- polls /stats for aggregate mesh metrics used in the header bar.
 */

import { useEffect, useState } from "react";
import type { MeshStats } from "../types";

const REGISTRY_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 3000;

const DEFAULT_STATS: MeshStats = {
  total_agents: 0,
  agents_by_status: {},
  avg_trust: 0,
  total_tasks_completed: 0,
  active_sessions: 0,
  active_task_counts: {},
};

export function useStats(): MeshStats {
  const [stats, setStats] = useState<MeshStats>(DEFAULT_STATS);

  useEffect(() => {
    let active = true;

    async function fetchStats() {
      try {
        const resp = await fetch(`${REGISTRY_URL}/stats`);
        if (!resp.ok) return;
        const data: MeshStats = await resp.json();
        if (active) setStats(data);
      } catch {
        // registry unreachable — keep last known stats
      }
    }

    fetchStats();
    const timer = setInterval(fetchStats, POLL_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return stats;
}
