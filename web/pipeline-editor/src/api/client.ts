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

export const api = {
  build: (s: ApiSettings, doc: PipelineDoc) => req(s, 'POST', '/api/v1/pipeline/build', doc),
  start: (s: ApiSettings, id: string) => req(s, 'POST', '/api/v1/pipeline/start', { id }),
  stop: (s: ApiSettings, id: string) => req(s, 'POST', '/api/v1/pipeline/stop', { id }),
  del: (s: ApiSettings, id: string) => req(s, 'DELETE', '/api/v1/pipeline/delete', { id }),
  list: (s: ApiSettings) => req(s, 'GET', '/api/v1/pipeline/list'),
  status: (s: ApiSettings, id: string) => req(s, 'POST', '/api/v1/pipeline/status', { id }),
  saveConfig: (s: ApiSettings, doc: PipelineDoc) => req(s, 'POST', '/api/v1/pipeline/save-config', doc),
};

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
    const st = r.json?.state || r.json?.status;
    if (st && want.includes(st)) return last;
    if (st === 'load_failed') return last;
    await new Promise((r) => setTimeout(r, 500));
  }
  return last;
}
