import { Node, Edge } from '@xyflow/react';
import { PipelineDoc, NodeDef } from '../types';
import { SCHEMA } from '../schema/nodeSchema';

// PipelineDoc → React Flow nodes/edges
export function toFlow(doc: PipelineDoc): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = doc.graph.nodes.map((n, i) => ({
    id: n.id,
    type: 'pipelineNode',
    position: n.position || { x: 120 + (i % 4) * 220, y: 80 + Math.floor(i / 4) * 130 },
    data: { label: n.id, nodeType: n.type },
  }));
  const edges: Edge[] = doc.graph.edges.map((e) => ({
    id: `${e.from}->${e.to}`,
    source: e.from,
    target: e.to,
    markerEnd: { type: 'arrowclosed' as any },
    animated: false,
    style: { strokeWidth: 2 },
  }));
  return { nodes, edges };
}

// React Flow nodes/edges + 各节点 params → PipelineDoc
export function fromFlow(
  pipelineId: string,
  rfNodes: Node[],
  rfEdges: Edge[],
  paramsById: Record<string, NodeDef['params']>,
): PipelineDoc {
  const nodes: NodeDef[] = rfNodes.map((n) => ({
    id: n.id,
    type: (n.data as any).nodeType as string,
    params: paramsById[n.id] || {},
    position: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
  }));
  const edges = rfEdges.map((e) => ({ from: e.source, to: e.target }));
  return { id: pipelineId, graph: { nodes, edges } };
}

export function nodeColor(type: string): string {
  return SCHEMA[type]?.color || '#888';
}
export function nodeLabel(type: string): string {
  return SCHEMA[type]?.label || type;
}
