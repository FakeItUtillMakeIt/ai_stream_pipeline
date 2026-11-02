import { EdgeDef } from '../types';

// 分层(Sugiyama 风格)自动布局：按最长路径分列(左→右=数据流方向)，
// 列内用重心法多趟排序减少交叉，最后按列高垂直居中。
export function layoutDAG(
  nodeIds: string[],
  edges: EdgeDef[],
  opts?: { xGap?: number; yGap?: number; originX?: number; originY?: number },
): Record<string, { x: number; y: number }> {
  const xGap = opts?.xGap ?? 250;
  const yGap = opts?.yGap ?? 150;
  const ox = opts?.originX ?? 60;
  const oy = opts?.originY ?? 60;

  const succ = new Map<string, string[]>();
  const pred = new Map<string, string[]>();
  nodeIds.forEach((id) => { succ.set(id, []); pred.set(id, []); });
  for (const e of edges) {
    if (succ.has(e.from) && succ.has(e.to) && e.from !== e.to) {
      succ.get(e.from)!.push(e.to);
      pred.get(e.to)!.push(e.from);
    }
  }

  // 最长路径分列（Kahn 拓扑；有环则剩余节点兜底放最后一列）
  const indeg = new Map<string, number>(nodeIds.map((id) => [id, pred.get(id)!.length]));
  const rank = new Map<string, number>(nodeIds.map((id) => [id, 0]));
  const queue = nodeIds.filter((id) => indeg.get(id) === 0);
  const placed = new Set(queue);
  while (queue.length) {
    const n = queue.shift()!;
    for (const m of succ.get(n)!) {
      rank.set(m, Math.max(rank.get(m)!, rank.get(n)! + 1));
      indeg.set(m, indeg.get(m)! - 1);
      if (indeg.get(m) === 0 && !placed.has(m)) { placed.add(m); queue.push(m); }
    }
  }
  let maxRank = 0;
  rank.forEach((r) => { if (r > maxRank) maxRank = r; });
  for (const id of nodeIds) if (!placed.has(id)) rank.set(id, maxRank + 1); // 环内节点兜底

  // 分层
  const layers = new Map<number, string[]>();
  for (const id of nodeIds) {
    const r = rank.get(id)!;
    if (!layers.has(r)) layers.set(r, []);
    layers.get(r)!.push(id);
  }
  const ranks = [...layers.keys()].sort((a, b) => a - b);
  const posInLayer = new Map<string, number>();
  ranks.forEach((r) => layers.get(r)!.forEach((id, i) => posInLayer.set(id, i)));

  const bary = (id: string, adj: Map<string, string[]>) => {
    const ns = adj.get(id)!;
    if (!ns.length) return posInLayer.get(id)!;
    let s = 0; for (const n of ns) s += posInLayer.get(n) ?? 0;
    return s / ns.length;
  };
  // 重心排序（前向+后向各若干趟）
  for (let iter = 0; iter < 4; iter++) {
    for (const r of ranks) { const arr = layers.get(r)!; arr.sort((a, b) => bary(a, pred) - bary(b, pred)); arr.forEach((id, i) => posInLayer.set(id, i)); }
    for (let k = ranks.length - 1; k >= 0; k--) { const arr = layers.get(ranks[k])!; arr.sort((a, b) => bary(a, succ) - bary(b, succ)); arr.forEach((id, i) => posInLayer.set(id, i)); }
  }

  // 坐标：列 = rank，行 = 列内序号，按最高列垂直居中
  const maxRows = Math.max(1, ...ranks.map((r) => layers.get(r)!.length));
  const out: Record<string, { x: number; y: number }> = {};
  ranks.forEach((r, ci) => {
    const arr = layers.get(r)!;
    const offset = (maxRows - arr.length) / 2;
    arr.forEach((id, ri) => { out[id] = { x: ox + ci * xGap, y: oy + (ri + offset) * yGap }; });
  });
  return out;
}
