import { PipelineDoc, getPath } from '../types';
import { SCHEMA } from '../schema/nodeSchema';

export interface Issue {
  level: 'error' | 'warn';
  message: string;
  nodeId?: string;
}

// 返回问题列表；有 error 则不可 Build
export function validate(doc: PipelineDoc): Issue[] {
  const issues: Issue[] = [];
  const nodes = doc.graph.nodes;
  const edges = doc.graph.edges;

  if (!doc.id || !doc.id.trim()) issues.push({ level: 'error', message: '管道 id 为空' });
  if (nodes.length === 0) issues.push({ level: 'error', message: '没有任何节点' });

  // 唯一 id
  const seen = new Set<string>();
  for (const n of nodes) {
    if (!n.id) issues.push({ level: 'error', message: '存在空节点 id' });
    else if (seen.has(n.id)) issues.push({ level: 'error', message: `节点 id 重复: ${n.id}`, nodeId: n.id });
    seen.add(n.id);
    if (!SCHEMA[n.type]) issues.push({ level: 'error', message: `未知节点类型: ${n.type}`, nodeId: n.id });
  }

  // 边引用有效 + 自环
  const ids = new Set(nodes.map((n) => n.id));
  const adj = new Map<string, string[]>();
  const indeg = new Map<string, number>();
  nodes.forEach((n) => { adj.set(n.id, []); indeg.set(n.id, 0); });
  const edgeKey = new Set<string>();
  for (const e of edges) {
    if (!ids.has(e.from)) issues.push({ level: 'error', message: `边的起点不存在: ${e.from}` });
    if (!ids.has(e.to)) issues.push({ level: 'error', message: `边的终点不存在: ${e.to}` });
    if (e.from === e.to) issues.push({ level: 'error', message: `自环: ${e.from}`, nodeId: e.from });
    const k = e.from + '->' + e.to;
    if (edgeKey.has(k)) issues.push({ level: 'warn', message: `重复边: ${k}` });
    edgeKey.add(k);
    if (adj.has(e.from) && ids.has(e.to)) {
      adj.get(e.from)!.push(e.to);
      indeg.set(e.to, (indeg.get(e.to) || 0) + 1);
    }
  }

  // 必填 params
  for (const n of nodes) {
    const def = SCHEMA[n.type];
    if (!def) continue;
    for (const [path, f] of Object.entries(def.fields)) {
      if (!f.required) continue;
      const v = getPath(n.params, path);
      const empty = v === undefined || v === null || v === '' ||
        (Array.isArray(v) && v.length === 0);
      if (empty) issues.push({ level: 'error', message: `${n.id}(${def.label}) 缺少必填: ${f.label || path}`, nodeId: n.id });
    }
  }

  // 环检测（Kahn 拓扑排序）
  const q: string[] = [];
  const deg = new Map(indeg);
  for (const [id, d] of deg) if (d === 0) q.push(id);
  let visited = 0;
  while (q.length) {
    const u = q.shift()!;
    visited++;
    for (const v of adj.get(u) || []) {
      const nd = (deg.get(v) || 0) - 1;
      deg.set(v, nd);
      if (nd === 0) q.push(v);
    }
  }
  if (visited < nodes.length) issues.push({ level: 'error', message: '存在环，不是 DAG' });

  // 孤立节点（非源/汇且无连接）
  for (const n of nodes) {
    const hasEdge = edges.some((e) => e.from === n.id || e.to === n.id);
    if (!hasEdge && nodes.length > 1) issues.push({ level: 'warn', message: `孤立节点: ${n.id}`, nodeId: n.id });
  }
  const hasSource = nodes.some((n) => SCHEMA[n.type]?.category === 'source');
  const hasSink = nodes.some((n) => SCHEMA[n.type]?.category === 'sink');
  if (!hasSource) issues.push({ level: 'warn', message: '没有源(source)节点' });
  if (!hasSink) issues.push({ level: 'warn', message: '没有输出(sink)节点' });

  return issues;
}

export function hasErrors(issues: Issue[]): boolean {
  return issues.some((i) => i.level === 'error');
}
