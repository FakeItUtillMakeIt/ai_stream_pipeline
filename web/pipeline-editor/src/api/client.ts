import { PipelineDoc } from '../types';

export interface ApiSettings {
  baseUrl: string;   // 同源默认 '' → 相对路径；dev 走 vite 代理
  token: string;     // AI_STREAM_API_TOKEN（可选）
}

function headers(s: ApiSettings): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  if (s.token) h['Authorization'] = 'Bearer ' + s.token;
  return h;
}

async function req(s: ApiSettings, method: string, path: string, body?: unknown) {
  const url = s.baseUrl + path;
  const res = await fetch(url, {
    method,
    headers: headers(s),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json: any = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* 非 JSON */ }
  return { ok: res.ok, status: res.status, json, text };
}

export interface PipelineInfo { id: string; running: boolean; state: string; play_url: string; evidence_dirs: string[]; }

export const api = {
  build: (s: ApiSettings, doc: PipelineDoc, replace = false) => req(s, 'POST', '/api/v1/pipeline/build', { ...doc, replace }),
  start: (s: ApiSettings, id: string) => req(s, 'POST', '/api/v1/pipeline/start', { id }),
  stop: (s: ApiSettings, id: string) => req(s, 'POST', '/api/v1/pipeline/stop', { id }),
  del: (s: ApiSettings, id: string) => req(s, 'DELETE', '/api/v1/pipeline/delete', { id }),
  list: (s: ApiSettings) => req(s, 'GET', '/api/v1/pipeline/list'),
  status: (s: ApiSettings, id: string) => req(s, 'POST', '/api/v1/pipeline/status', { id }),
  saveConfig: (s: ApiSettings, doc: PipelineDoc) => req(s, 'POST', '/api/v1/pipeline/save-config', doc),
  getConfig: (s: ApiSettings, id: string) => req(s, 'GET', '/api/v1/pipeline/get-config?id=' + encodeURIComponent(id)),
  metrics: (s: ApiSettings) => req(s, 'GET', '/api/v1/metrics'),
  evidenceList: (s: ApiSettings, pipeline: string) =>
    req(s, 'GET', '/api/v1/evidence/list?pipeline=' + encodeURIComponent(pipeline)),
  evidenceFileUrl: (s: ApiSettings, pipeline: string, dir: number, name: string) =>
    s.baseUrl + `/api/v1/evidence/file?pipeline=${encodeURIComponent(pipeline)}&dir=${dir}&name=${encodeURIComponent(name)}`,
};

// 拉取并规范化管道列表（含 play_url / evidence 目录）
export async function listPipelines(s: ApiSettings): Promise<PipelineInfo[]> {
  const r = await api.list(s);
  const arr = r.json?.pipelines || [];
  return (Array.isArray(arr) ? arr : []).map((it: any) => ({
    id: it.id,
    running: it.running === true || deriveState(it) === 'running',
    state: deriveState(it) || (it.running ? 'running' : 'stopped'),
    play_url: it.play_url || '',
    evidence_dirs: Array.isArray(it.evidence_dirs) ? it.evidence_dirs : [],
  }));
}

// 从 status/list 的返回项推导状态字符串。
// 同步模式只有 {running:bool}（无 state）；异步模式有 {state,running}。
export function deriveState(s: any): string {
  if (!s) return '';
  if (typeof s.state === 'string' && s.state) return s.state;
  if (typeof s.status === 'string' && s.status && s.status !== 'error') return s.status;
  if (typeof s.running === 'boolean') return s.running ? 'running' : 'stopped';
  return '';
}

// 轮询 status 直到 state 稳定或超时
export async function waitState(
  s: ApiSettings, id: string, want: string[], timeoutMs = 30000, onTick?: (st: any) => void,
): Promise<any> {
  const deadline = Date.now() + timeoutMs;
  let last: any = null;
  while (Date.now() < deadline) {
    const r = await api.status(s, id);
    last = r.json;
    onTick?.(last);
    const st = deriveState(r.json);
    if (st && want.includes(st)) return last;
    if (st === 'load_failed') return last;
    await new Promise((r) => setTimeout(r, 500));
  }
  return last;
}
