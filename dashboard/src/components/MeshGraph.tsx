/**
 * MeshGraph -- force-directed network visualization of all mesh agents.
 *
 * Nodes:
 *   - Color maps to trust score: red < 0.5 < yellow < 0.8 < green
 *   - Size maps to number of capabilities
 *   - Click a node to surface its detail panel
 *
 * Edges appear when agents communicate (from TraceEvents) with a brief
 * highlight animation. Powered by react-force-graph-2d.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { AgentRecord, GraphEdge, GraphNode, TraceEvent } from "../types";

interface Props {
  agents: AgentRecord[];
  traces: TraceEvent[];
  onSelectAgent: (record: AgentRecord) => void;
}

function trustColor(score: number): string {
  if (score < 0.5) return "#ef4444"; // red-500
  if (score < 0.8) return "#eab308"; // yellow-500
  return "#22c55e";                   // green-500
}

function capabilitySize(count: number): number {
  return 6 + count * 4;
}

/** Convert live trace events to graph edges, keeping only the last 30. */
function tracesToEdges(traces: TraceEvent[], agentIds: Set<string>): GraphEdge[] {
  return traces
    .filter(
      (t) =>
        agentIds.has(t.from_agent) &&
        agentIds.has(t.to_agent) &&
        t.from_agent !== t.to_agent
    )
    .slice(-30)
    .map((t) => ({
      source: t.from_agent,
      target: t.to_agent,
      eventType: t.event_type,
      timestamp: t.timestamp,
    }));
}

export default function MeshGraph({ agents, traces, onSelectAgent }: Props) {
  const fgRef = useRef<InstanceType<typeof ForceGraph2D> | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const agentIds = useMemo(
    () => new Set(agents.map((a) => a.manifest.agent_id)),
    [agents]
  );

  const graphData = useMemo(() => {
    const nodes: GraphNode[] = agents.map((a) => ({
      id: a.manifest.agent_id,
      name: a.manifest.name,
      trustScore: a.trust_score,
      capabilityCount: a.manifest.capabilities.length,
      status: a.status,
      record: a,
    }));
    const links = tracesToEdges(traces, agentIds);
    return { nodes, links };
  }, [agents, traces, agentIds]);

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      onSelectAgent(node.record);
    },
    [onSelectAgent]
  );

  const nodeCanvasObject = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const r = capabilitySize(node.capabilityCount) / globalScale;
      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Outer ring for offline agents
      if (node.status === "offline") {
        ctx.beginPath();
        ctx.arc(x, y, r + 2 / globalScale, 0, 2 * Math.PI);
        ctx.strokeStyle = "#6b7280";
        ctx.lineWidth = 1.5 / globalScale;
        ctx.stroke();
      }

      // Main circle
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle =
        node.status === "offline" ? "#374151" : trustColor(node.trustScore);
      ctx.fill();

      // Label
      const label = node.name;
      const fontSize = Math.max(10 / globalScale, 3);
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#f9fafb";
      ctx.fillText(label, x, y + r + fontSize * 0.8);
    },
    []
  );

  return (
    <div
      ref={containerRef}
      className="w-full h-full bg-gray-950 rounded-xl overflow-hidden"
    >
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData as never}
        nodeId="id"
        linkSource="source"
        linkTarget="target"
        nodeCanvasObject={nodeCanvasObject as never}
        nodeCanvasObjectMode={() => "replace"}
        onNodeClick={handleNodeClick as never}
        linkColor={() => "#4b5563"}
        linkWidth={1.5}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        backgroundColor="#030712"
        cooldownTicks={80}
      />

      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex flex-col gap-1 text-xs text-gray-400 bg-gray-900 bg-opacity-80 rounded-lg px-3 py-2">
        <span className="font-semibold text-gray-300 mb-1">Trust Score</span>
        <span>
          <span className="inline-block w-3 h-3 rounded-full bg-red-500 mr-2" />
          &lt; 0.5
        </span>
        <span>
          <span className="inline-block w-3 h-3 rounded-full bg-yellow-500 mr-2" />
          0.5 - 0.8
        </span>
        <span>
          <span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-2" />
          &ge; 0.8
        </span>
        <span className="mt-1 text-gray-500">Node size = # capabilities</span>
      </div>
    </div>
  );
}
