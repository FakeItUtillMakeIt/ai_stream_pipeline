import { Handle, Position, NodeProps } from '@xyflow/react';
import { nodeColor, nodeLabel } from '../graph/flow';

// data.status: waiting(灰) | running(绿) | error(红闪)
export default function PipelineNode({ id, data, selected }: NodeProps) {
  const type = (data as any).nodeType as string;
  const status = ((data as any).status as string) || 'waiting';
  const color = nodeColor(type);
  return (
    <div className={`pnode st-${status} ${selected ? 'sel' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="hd" style={{ background: color }}>
        <span>{nodeLabel(type)}</span>
        <span className="st" title={status} />
      </div>
      <div className="body">{id}</div>
      <div className="ty">{type}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
