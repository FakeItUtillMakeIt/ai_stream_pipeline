// 与 ApiServer / buildFromJson 对齐的管道数据模型
export interface NodeDef {
  id: string;
  type: string;
  params: Record<string, unknown>;
  // 画布坐标（仅前端用，导出时保留以便复现布局；后端忽略未知字段）
  position?: { x: number; y: number };
}

export interface EdgeDef {
  from: string;
  to: string;
}

export interface PipelineGraph {
  nodes: NodeDef[];
  edges: EdgeDef[];
}

export interface PipelineDoc {
  id: string;
  graph: PipelineGraph;
}

// 嵌套字段按点路径读写：如 "detector_config.conf_threshold"
export function getPath(obj: Record<string, unknown>, path: string): unknown {
  return path.split('.').reduce<any>((o, k) => (o == null ? undefined : o[k]), obj);
}

export function setPath(obj: Record<string, unknown>, path: string, val: unknown): void {
  const keys = path.split('.');
  let cur: any = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    if (typeof cur[k] !== 'object' || cur[k] === null) cur[k] = {};
    cur = cur[k];
  }
  cur[keys[keys.length - 1]] = val;
}

export function deletePath(obj: Record<string, unknown>, path: string): void {
  const keys = path.split('.');
  let cur: any = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    if (cur[keys[i]] == null) return;
    cur = cur[keys[i]];
  }
  delete cur[keys[keys.length - 1]];
}
