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

const PROC_LABEL: Record<string, string> = { new: '未处理', ack: '已确认', dismiss: '误报', resolve: '已解决' };
const TYPE_LABEL: Record<string, string> = { '人员入侵': '人员入侵', '打架': '打架', '儿童靠近边界': '儿童靠近边界', '攀爬': '攀爬', '人员徘徊': '人员徘徊', '跌倒': '跌倒' };
const ALERT_PAGE = 100;

function alertRouteParams(): URLSearchParams {
  const raw = (location.hash || '').replace(/^#\/?/, '');
  return new URLSearchParams(raw.split('?')[1] || '');
}

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
  const [stats, setStats] = useState<Record<string, number>>({});
  const [detail, setDetail] = useState<string | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [moreBusy, setMoreBusy] = useState(false);

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

  return (
    <div className="alerts">
      <div className="al-top">
        <button className="btn" onClick={onBack}>{backLabel}</button>
        <span className="al-title">告警记录</span>
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
        <button className="btn sm" onClick={refresh} disabled={busy}>{busy ? '刷新中…' : '刷新'}</button>
      </div>
      {err && <div className="al-err">{err}</div>}

      <div className="al-body">
        {busy && items.length === 0 ? (
          <div className="al-empty">查询中…</div>
        ) : items.length === 0 ? (
          <div className="al-empty">当前筛选条件下无告警记录（有告警且管道含 evidence 节点即自动记录）</div>
        ) : items.map((a) => {
          const pr = a.proc || 'new';
          const snap = a.pre_gate?.snapshot || a.snapshot;
          const g = a.post_gate;
          return (
              <div key={a.uid} className={'al-row' + (pr === 'new' ? ' unhandled' : '')}>
                {snap && <MediaImg className="al-thumb" url={api.alertFileUrl(settings, snap)} token={settings.token}
                  onClick={() => setDetail(detail === a.uid ? null : a.uid)} />}
                <div className="al-main" onClick={() => setDetail(detail === a.uid ? null : a.uid)}>
                  <div className="al-h">
                    <span className={'al-proc ' + pr}>{PROC_LABEL[pr] || pr}</span>
                    <span className="al-type">{TYPE_LABEL[a.type || ''] || a.type || (a.alert?.alert_type_name as string) || '告警'}</span>
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
                    <pre className="al-json">{JSON.stringify({ uid: a.uid, pre_gate: a.pre_gate, post_gate: a.post_gate, alert: a.alert }, null, 2)}</pre>
                  </div>
                )}
              </div>
            );
        })}
      </div>
      <div className="al-foot">
        <span>显示 {items.length} / {total}</span>
        {items.length < total && (
          <button className="btn sm" onClick={loadMore} disabled={moreBusy}>
            {moreBusy ? '加载中…' : '加载更多'}
          </button>
        )}
        {items.length > ALERT_PAGE && (
          <span className="al-foot-note">已加载多页，自动刷新已暂停；点「刷新」回到最新</span>
        )}
      </div>
    </div>
  );
}
