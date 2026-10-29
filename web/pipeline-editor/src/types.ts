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

// ===== 模板系统 =====
// overrides：模板声明"暴露给实施"的字段；其余参数锁定。
export interface Overrides {
  source_url?: { node: string; path: string; label?: string };  // 拉流地址（默认 path=params.url）
  push?: { node: string; label?: string };                      // 是否推流（开关=保留/移除该 sink 节点）
  alerts?: { node: string; types: string[]; label?: string };   // 告警项（可勾选的规则类型子集）
}
export interface TemplateDef {
  name: string;
  version: string;
  description: string;
  doc: PipelineDoc;
  overrides: Overrides;
}

const SOURCE_TYPES = ['file_source', 'rtsp_source'];

// 从管道图自动探测可暴露字段（研发"发布为模板"时的默认勾选）
export function detectOverrides(doc: PipelineDoc): Overrides {
  const ov: Overrides = {};
  const src = doc.graph.nodes.find((n) => SOURCE_TYPES.includes(n.type));
  if (src) ov.source_url = { node: src.id, path: 'params.url', label: '拉流地址' };
  const rtmp = doc.graph.nodes.find((n) => n.type === 'rtmp_sink');
  if (rtmp) ov.push = { node: rtmp.id, label: '是否推流' };
  const alert = doc.graph.nodes.find((n) => n.type === 'alert');
  if (alert) {
    const rules = (getPath(alert.params, 'rules') as any[]) || [];
    const types = rules.map((r) => String(r.type)).filter(Boolean);
    ov.alerts = { node: alert.id, types, label: '告警项' };
  }
  return ov;
}
