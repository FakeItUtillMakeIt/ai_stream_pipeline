import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiSettings, api, listPipelines, PipelineInfo } from '../api/client';
import { MediaImg } from './Media';

interface PreGate { triggered?: boolean; alert_name?: string; snapshot?: string; }
interface PostGate { decision?: string; verdict?: boolean; confidence?: number; reason?: string; error?: string; pushed?: boolean; latency_ms?: number; model?: string; snapshot?: string; }
interface AlertRec {
  uid: string; pipeline: string; ts: number; type?: string; level?: string;
  description?: string; proc?: string; snapshot?: string; vlm?: unknown; alert?: Record<string, unknown>;
  pre_gate?: PreGate; post_gate?: PostGate;
}

interface AlertStats {
  new?: number; ack?: number; dismiss?: number; resolve?: number; total?: number;
  types?: Record<string, number>;
  decisions?: Record<string, number>;
  daily?: { date: string; count: number }[];
}

const PROC_LABEL: Record<string, string> = { new: '未处理', ack: '已确认', dismiss: '误报', resolve: '已解决' };
const TYPE_LABEL: Record<string, string> = { '人员入侵': '人员入侵', '打架': '打架', '儿童靠近边界': '儿童靠近边界', '攀爬': '攀爬', '人员徘徊': '人员徘徊', '跌倒': '跌倒' };
const ALERT_PAGE = 100;

function alertRouteParams(): URLSearchParams {
  const raw = (location.hash || '').replace(/^#\/?/, '');
  return new URLSearchParams(raw.split('?')[1] || '');
}

const csvCell = (value: unknown) => `"${String(value ?? '').replace(/"/g, '""')}"`;

export default function AlertsView({ settings, initialPipeline, onBack, backLabel = '← 返回' }: {
  settings: ApiSettings; initialPipeline?: string; onBack: () => void; backLabel?: string;
}) {
  const route = useRef(alertRouteParams()).current;
  const [pipes, setPipes] = useState<PipelineInfo[]>([]);
  const [pipeline, setPipeline] = useState(initialPipeline || route.get('pipeline') || '');
  const [type, setType] = useState(route.get('type') || '');
  const [proc, setProc] = useState(route.get('proc') || 'all');
  const [q, setQ] = useState(route.get('q') || '');
  const [decision, setDecision] = useState(route.get('decision') || 'all');   // all|PUSH|HOLD|DROP|PASSTHROUGH|none
  const [range, setRange] = useState(route.get('range') || 'all');   // all|today|7d|30d
  const [items, setItems] = useState<AlertRec[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<AlertStats>({});
  const [detail, setDetail] = useState<string | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [moreBusy, setMoreBusy] = useState(false);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [bulkBusy, setBulkBusy] = useState(false);
  const [lightbox, setLightbox] = useState<{ url: string; name: string } | null>(null);

  const fromMs = () => {
    const now = Date.now();
    if (range === 'today') { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime(); }
    if (range === '7d') return now - 7 * 864e5;
    if (range === '30d') return now - 30 * 864e5;
    return 0;
  };

  const loadAlerts = useCallback(async (offset: number, append: boolean) => {
    if (append) setMoreBusy(true); else setBusy(true);
    try {
      const p: Record<string, string | number> = { limit: ALERT_PAGE, offset, proc };
      if (pipeline) p.pipeline = pipeline;
      if (type) p.type = type;
      if (q) p.q = q;
      if (decision && decision !== 'all') p.decision = decision;
      const f = fromMs(); if (f) p.from = f;
      const r = await api.alerts(settings, p);
      if (r.ok) {
        const incoming: AlertRec[] = Array.isArray(r.json?.items) ? r.json.items : [];
        if (append) {
          setItems((prev) => {
            const seen = new Set(prev.map((x) => x.uid));
            const merged = prev.slice();
            for (const item of incoming) {
              if (!seen.has(item.uid)) { seen.add(item.uid); merged.push(item); }
            }
            return merged;
          });
        } else {
          setItems(incoming);
        }
        setTotal(Number(r.json?.total) || 0);
        setErr('');
      } else {
        setErr('查询失败: ' + JSON.stringify(r.json));
      }
      const s = await api.alertsStats(settings, pipeline);
      if (s.ok) setStats(s.json || {});
    } catch (e: any) {
      setErr('查询异常: ' + (e?.message || e));
    } finally {
      if (append) setMoreBusy(false); else setBusy(false);
    }
  }, [settings, pipeline, type, proc, decision, q, range]);

  const refresh = useCallback(() => loadAlerts(0, false), [loadAlerts]);
  const loadMore = () => loadAlerts(items.length, true);

  useEffect(() => { listPipelines(settings).then(setPipes).catch(() => {}); /* eslint-disable-next-line */ }, [settings.baseUrl, settings.token]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    // 已加载多页时暂停自动刷新，避免把用户翻出来的记录冲掉
    const t = setInterval(() => { if (items.length <= ALERT_PAGE) refresh(); }, 8000);
    return () => clearInterval(t);
  }, [refresh, items.length]);
  useEffect(() => { if (initialPipeline) setPipeline(initialPipeline); }, [initialPipeline]);

  // 筛选条件写入 URL，刷新/分享/前进后退可恢复
  useEffect(() => {
    const params = new URLSearchParams();
    if (pipeline) params.set('pipeline', pipeline);
    if (type) params.set('type', type);
    if (proc !== 'all') params.set('proc', proc);
    if (decision !== 'all') params.set('decision', decision);
    if (range !== 'all') params.set('range', range);
    if (q) params.set('q', q);
    const query = params.toString();
    const next = '#/alerts' + (query ? '?' + query : '');
    if (location.hash !== next) history.replaceState(null, '', next);
  }, [pipeline, type, proc, decision, range, q]);

  useEffect(() => {
    const onHash = () => {
      const params = alertRouteParams();
      setPipeline(params.get('pipeline') || '');
      setType(params.get('type') || '');
      setProc(params.get('proc') || 'all');
      setDecision(params.get('decision') || 'all');
      setRange(params.get('range') || 'all');
      setQ(params.get('q') || '');
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const doAck = async (uid: string, p: string) => { await api.alertsAck(settings, uid, p); refresh(); };

  // 筛选条件变化时清空选择；自动刷新/加载更多不会打断勾选
  useEffect(() => { setSelected({}); }, [pipeline, type, proc, decision, range, q]);

  useEffect(() => {
    if (!lightbox) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setLightbox(null); };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener('keydown', onKey);
    };
  }, [lightbox]);

  const selectedCount = items.filter((a) => selected[a.uid]).length;
  const allSelected = items.length > 0 && items.every((a) => selected[a.uid]);
  const toggleAll = () => setSelected(allSelected ? {} : Object.fromEntries(items.map((a) => [a.uid, true])));

  const doBulk = async (proc: string) => {
    const uids = items.filter((a) => selected[a.uid]).map((a) => a.uid);
    if (!uids.length) return;
    setBulkBusy(true);
    try {
      for (const uid of uids) await api.alertsAck(settings, uid, proc);
    } finally {
      setBulkBusy(false);
      setSelected({});
      refresh();
    }
  };

  const exportCsv = (onlySelected = false) => {
    const rows = onlySelected ? items.filter((a) => selected[a.uid]) : items;
    if (!rows.length) return;
    const header = ['uid', '时间', '管道', '类型', '级别', '处理', '描述', '判决', '置信度', '模型', '原因', '错误', '已推送', '快照'];
    const lines = [header.map(csvCell).join(',')];
    for (const a of rows) {
      const g = a.post_gate;
      lines.push([
        a.uid,
        a.ts ? new Date(a.ts).toISOString() : '',
        a.pipeline,
        a.type || '',
        a.level || '',
        a.proc || 'new',
        a.description || '',
        g?.decision || '',
        g?.confidence ?? '',
        g?.model || '',
        g?.reason || '',
        g?.error || '',
        g?.pushed ?? '',
        a.snapshot || a.pre_gate?.snapshot || '',
      ].map(csvCell).join(','));
    }
    const blob = new Blob(['\uFEFF' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const link = document.createElement('a');
    link.href = url;
    link.download = `alerts-${stamp}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="alerts">
      <div className="al-top">
        <button className="btn" onClick={onBack}>{backLabel}</button>
        <span className="al-title">告警记录</span>
        <label className="al-check-all" title="全选当前列表">
          <input type="checkbox" checked={allSelected} onChange={toggleAll} disabled={items.length === 0} /> 全选
        </label>
        <select value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
          <option value="">全部管道</option>
          {pipes.map((p) => <option key={p.id} value={p.id}>{p.id}</option>)}
        </select>
        <select value={proc} onChange={(e) => setProc(e.target.value)}>
          <option value="all">全部状态</option>
          <option value="new">未处理</option><option value="ack">已确认</option>
          <option value="dismiss">误报</option><option value="resolve">已解决</option>
        </select>
        <select value={decision} onChange={(e) => setDecision(e.target.value)}>
          <option value="all">全部判决</option>
          <option value="PUSH">闸门:PUSH</option><option value="HOLD">闸门:HOLD</option>
          <option value="DROP">闸门:DROP</option><option value="PASSTHROUGH">闸门:直通</option>
          <option value="none">无闸门</option>
        </select>
        <select value={range} onChange={(e) => setRange(e.target.value)}>
          <option value="all">全部时间</option><option value="today">今天</option>
          <option value="7d">近7天</option><option value="30d">近30天</option>
        </select>
        <input className="al-q" placeholder="搜索 描述/类型/管道" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="spacer" />
        <span className="al-stat">
          未处理 <b style={{ color: 'var(--err)' }}>{stats.new || 0}</b> ·
          已确认 <b style={{ color: 'var(--primary)' }}>{stats.ack || 0}</b> ·
          误报 <b style={{ color: 'var(--muted)' }}>{stats.dismiss || 0}</b> ·
          已解决 <b style={{ color: 'var(--ok)' }}>{stats.resolve || 0}</b> ·
          共 {stats.total || 0}
        </span>
        <button className="btn sm" onClick={() => exportCsv(false)} disabled={items.length === 0 || bulkBusy}>导出当前</button>
        <button className="btn sm" onClick={refresh} disabled={busy}>{busy ? '刷新中…' : '刷新'}</button>
      </div>
      {err && <div className="al-err">{err}</div>}
      <AlertOverview stats={stats} />

      <div className="al-body">
        {busy && items.length === 0 ? (
          <div className="al-empty">查询中…</div>
        ) : items.length === 0 ? (
          <div className="al-empty">当前筛选条件下无告警记录（有告警且管道含 evidence 节点即自动记录）</div>
        ) : items.map((a) => {
          const pr = a.proc || 'new';
          const snap = a.pre_gate?.snapshot || a.snapshot;
          const snapUrl = snap ? api.alertFileUrl(settings, snap) : '';
          const g = a.post_gate;
          const typeLabel = TYPE_LABEL[a.type || ''] || a.type || (a.alert?.alert_type_name as string) || '告警';
          return (
              <div key={a.uid} className={'al-row' + (pr === 'new' ? ' unhandled' : '')}>
                <input
                  className="al-rowchk"
                  type="checkbox"
                  checked={!!selected[a.uid]}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setSelected((s) => ({ ...s, [a.uid]: e.target.checked }))}
                  aria-label="选择告警"
                />
                {snap && <MediaImg className="al-thumb" url={snapUrl} token={settings.token}
                  onClick={(e) => { e.stopPropagation(); setLightbox({ url: snapUrl, name: snap }); }} />}
                <div className="al-main" onClick={() => setDetail(detail === a.uid ? null : a.uid)}>
                  <div className="al-h">
                    <span className={'al-proc ' + pr}>{PROC_LABEL[pr] || pr}</span>
                    <span className="al-type">{typeLabel}</span>
                    <span className={'al-level ' + (a.level || '')}>{a.level || ''}</span>
                    <span className="al-pipe">{a.pipeline}</span>
                  </div>
                  <div className="al-desc">{a.description || (a.alert?.description as string) || '—'}</div>
                  <div className="al-layers">
                    <span className="al-layer pre">① 触发{a.pre_gate ? '✓' : '（无证据层）'}{snap ? ' · 有快照' : ''}</span>
                    <span className={'al-layer post ' + (g?.decision || 'none')}>
                      ② 闸门{g?.decision ? '：' + g.decision : '：无'}
                      {g && typeof g.confidence === 'number' ? ` · conf ${g.confidence.toFixed(2)}` : ''}
                      {g?.error ? ` · ${g.error}` : ''}
                    </span>
                  </div>
                  <div className="al-time">{a.ts ? new Date(a.ts).toLocaleString() : '—'}</div>
                </div>
                <div className="al-acts" onClick={(e) => e.stopPropagation()}>
                  {pr === 'new' && <button className="btn sm" onClick={() => doAck(a.uid, 'ack')}>确认</button>}
                  {pr !== 'dismiss' && <button className="btn sm" onClick={() => doAck(a.uid, 'dismiss')}>误报</button>}
                  {pr !== 'resolve' && <button className="btn sm" onClick={() => doAck(a.uid, 'resolve')}>已解决</button>}
                  {pr !== 'new' && <button className="btn sm ghost" onClick={() => doAck(a.uid, 'new')}>重置</button>}
                </div>
                {detail === a.uid && (
                  <div className="al-detail">
                    <div className="al-detail-grid">
                      <div className="al-detail-card">
                        <div className="al-detail-title">触发信息</div>
                        <div className="al-detail-row"><span>类型</span><b>{typeLabel}</b></div>
                        <div className="al-detail-row"><span>时间</span><b>{a.ts ? new Date(a.ts).toLocaleString() : '—'}</b></div>
                        <div className="al-detail-row"><span>管道</span><b>{a.pipeline}</b></div>
                        <div className="al-detail-row"><span>处理</span><b>{PROC_LABEL[pr] || pr}</b></div>
                        <div className="al-detail-row"><span>描述</span><b>{a.description || (a.alert?.description as string) || '—'}</b></div>
                        <div className="al-detail-row"><span>证据层</span><b>{a.pre_gate?.triggered ? '已触发' : a.pre_gate ? '未触发' : '无'}</b></div>
                        {snapUrl && (
                          <MediaImg
                            className="al-detail-thumb"
                            url={snapUrl}
                            token={settings.token}
                            alt={snap || 'snapshot'}
                            onClick={() => setLightbox({ url: snapUrl, name: snap || 'snapshot' })}
                          />
                        )}
                      </div>
                      <div className="al-detail-card">
                        <div className="al-detail-title">VLM 闸门</div>
                        <div className="al-detail-row">
                          <span>判决</span>
                          <span className={'al-layer post ' + (g?.decision || 'none')}>{g?.decision || '无'}</span>
                        </div>
                        <div className="al-detail-row"><span>置信度</span><b>{typeof g?.confidence === 'number' ? g.confidence.toFixed(2) : '—'}</b></div>
                        <div className="al-detail-row"><span>模型</span><b>{g?.model || '—'}</b></div>
                        <div className="al-detail-row"><span>延迟</span><b>{typeof g?.latency_ms === 'number' ? `${g.latency_ms} ms` : '—'}</b></div>
                        <div className="al-detail-row"><span>推送</span><b>{g?.pushed ? '已推送' : g?.pushed === false ? '未推送' : '—'}</b></div>
                        <div className="al-detail-row"><span>原因</span><b>{g?.reason || '—'}</b></div>
                        {g?.error && <div className="al-detail-err">{g.error}</div>}
                      </div>
                    </div>
                    <details className="al-raw">
                      <summary>原始 JSON</summary>
                      <pre className="al-json">{JSON.stringify({ uid: a.uid, pre_gate: a.pre_gate, post_gate: a.post_gate, alert: a.alert }, null, 2)}</pre>
                    </details>
                  </div>
                )}
              </div>
            );
        })}
      </div>
      {selectedCount > 0 && (
        <div className="al-bulk">
          <span>已选 <b>{selectedCount}</b> 项</span>
          <button className="btn sm primary" onClick={() => doBulk('ack')} disabled={bulkBusy}>确认</button>
          <button className="btn sm" onClick={() => doBulk('dismiss')} disabled={bulkBusy}>误报</button>
          <button className="btn sm" onClick={() => doBulk('resolve')} disabled={bulkBusy}>已解决</button>
          <button className="btn sm ghost" onClick={() => doBulk('new')} disabled={bulkBusy}>重置</button>
          <button className="btn sm" onClick={() => exportCsv(true)} disabled={bulkBusy}>导出选中</button>
          <div className="spacer" />
          {bulkBusy && <span className="al-bulk-busy">处理中…</span>}
          <button className="btn sm ghost" onClick={() => setSelected({})}>取消选择</button>
        </div>
      )}
      <div className="al-foot">
        <span>显示 {items.length} / {total}</span>
        {items.length < total && (
          <button className="btn sm" onClick={loadMore} disabled={moreBusy}>
            {moreBusy ? '加载中…' : '加载更多'}
          </button>
        )}
        {items.length > ALERT_PAGE ? (
          <span className="al-foot-note">已加载多页，自动刷新已暂停；点「刷新」回到最新</span>
        ) : (
          <span className="al-foot-note">自动刷新 8s</span>
        )}
      </div>
      {lightbox && (
        <div className="al-lightbox" onClick={(e) => { if (e.target === e.currentTarget) setLightbox(null); }}>
          <div className="al-lightbox-inner" role="dialog" aria-modal="true" aria-label="告警快照">
            <button className="al-lightbox-close" onClick={() => setLightbox(null)} aria-label="关闭">×</button>
            <MediaImg className="al-lightbox-img" url={lightbox.url} token={settings.token} alt={lightbox.name} />
            <div className="al-lightbox-name">{lightbox.name}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function AlertOverview({ stats }: { stats: AlertStats }) {
  const daily = stats.daily || [];
  const types = Object.entries(stats.types || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  if (daily.length === 0 && types.length === 0) return null;
  const max = Math.max(1, ...daily.map((d) => d.count));
  return (
    <div className="al-overview">
      <div className="al-overview-block">
        <div className="al-overview-title">近 7 天告警</div>
        <div className="al-bars">
          {daily.map((d) => (
            <div className="al-bar" key={d.date} title={`${d.date} · ${d.count} 条`}>
              <span style={{ height: `${Math.max(4, (d.count / max) * 100)}%` }} />
              <em>{d.date.slice(5)}</em>
            </div>
          ))}
        </div>
      </div>
      <div className="al-overview-block al-overview-types">
        <div className="al-overview-title">类型分布</div>
        <div className="al-type-chips">
          {types.map(([type, count]) => (
            <span className="al-type-chip" key={type}>{TYPE_LABEL[type] || type}<b>{count}</b></span>
          ))}
          {types.length === 0 && <span className="al-overview-empty">暂无</span>}
        </div>
      </div>
    </div>
  );
}
