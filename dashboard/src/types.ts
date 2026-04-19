/** Shared TypeScript types mirroring the Python Pydantic models. */

export interface CapabilitySchema {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  avg_latency_ms: number | null;
  cost_per_call_usd: number | null;
}

export interface AgentManifest {
  agent_id: string;
  name: string;
  version: string;
  description: string;
  capabilities: CapabilitySchema[];
  mcp_servers: string[];
  max_concurrent_tasks: number;
  endpoint: string;
  tags: string[];
}

export interface AgentRecord {
  manifest: AgentManifest;
  trust_score: number;
  status: "healthy" | "degraded" | "offline";
  registered_at: string;
  last_heartbeat: string;
  tasks_completed: number;
  tasks_failed: number;
}

export interface TraceEvent {
  trace_id: string;
  task_id: string;
  event_type:
    | "request_sent"
    | "accepted"
    | "rejected"
    | "executing"
    | "completed"
    | "failed";
  from_agent: string;
  to_agent: string;
  payload: Record<string, unknown>;
  timestamp: string;
}

/** Graph node for the force-directed mesh visualization. */
export interface GraphNode {
  id: string;
  name: string;
  trustScore: number;
  capabilityCount: number;
  status: AgentRecord["status"];
  record: AgentRecord;
}

/** Graph edge representing an agent-to-agent interaction. */
export interface GraphEdge {
  source: string;
  target: string;
  eventType: TraceEvent["event_type"];
  timestamp: string;
}
