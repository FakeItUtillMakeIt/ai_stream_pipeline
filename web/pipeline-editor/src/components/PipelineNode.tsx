import { Handle, Position, NodeProps } from '@xyflow/react';
import { nodeColor, nodeLabel } from '../graph/flow';

export default function PipelineNode({ id, data, selected }: NodeProps) {
  const type = (data as any).nodeType as string;
  const color = nodeColor(type);
  return (
    <div
      style={{
        background: '#fff',
        border: `2px solid ${selected ? '#1e88ff' : color}`,
        borderRadius: 8,
        minWidth: 150,
        boxShadow: '0 1px 4px rgba(0,0,0,.15)',
        fontSize: 12,
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ background: color, color: '#fff', padding: '4px 8px', borderRadius: '6px 6px 0 0', fontWeight: 600 }}>
        {nodeLabel(type)}
      </div>
      <div style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{id}</div>
      <div style={{ padding: '0 8px 6px', color: '#888' }}>{type}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
