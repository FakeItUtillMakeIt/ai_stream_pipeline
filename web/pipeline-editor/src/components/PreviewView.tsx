import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { MediaImg, MediaVideo, LivePlayer } from './Media';
import type { LivePlayerStatus } from './Media';
import { ApiSettings, api, listPipelines, PipelineInfo } from '../api/client';

export type PreviewTab = 'preview' | 'evidence' | 'metrics' | 'status';
type Tab = PreviewTab;
type EvidenceKind = 'all' | 'image' | 'video';

interface EvFile {
  dir_index: number;
  name: string;
  kind: string;
  size: number;
}

interface EvidenceMeta {
  typeKey: string;
  typeLabel: string;
  ts: number | null;
  timeText: string;
}

interface EvItem extends EvFile, EvidenceMeta {
  key: string;
  mediaKind: 'image' | 'video';
}

interface MetricRow {
  pipeline: string;
  node: string;
  total: number;
  dropped: number;
  fps: number;
  latencyMin: number | null;
  latencyAvg: number | null;
  latencyMax: number | null;
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'preview', label: '实时预览' },
  { id: 'evidence', label: '证据' },
  { id: 'metrics', label: '指标' },
  { id: 'status', label: '状态' },
];

function previewRouteParams(): URLSearchParams {
  const raw = (location.hash || '').replace(/^#\/?/, '');
  return new URLSearchParams(raw.split('?')[1] || '');
}

function isPreviewTab(value: string | null): value is Tab {
  return !!value && TABS.some((tab) => tab.id === value);
}

const EVIDENCE_LABELS: Record<string, string> = {
  person_intrusion: '人员入侵',
  child_near_wall: '儿童靠近边界',
  child_near_boundary: '儿童靠近边界',
  fighting: '打架',
  fall_down: '跌倒',
  climb: '攀爬',
  loitering: '人员徘徊',
  helmet: '未戴安全帽',
  work_clothes: '未穿工作服',
  phone: '玩手机',
  smoking: '吸烟',
  safety_belt: '未系安全带',
  sleeping: '睡岗',
};

function evidenceLabel(key: string): string {
  if (!key) return '证据';
  return EVIDENCE_LABELS[key] || key.replace(/_/g, ' ');
}

function parseEvidenceName(name: string): EvidenceMeta {
  const base = name.replace(/\.[^.]+$/, '');
  const m = base.match(/^(.*)_(\d{8})_(\d{6})(?:_(\d{3}))?$/);
  if (!m) return { typeKey: '', typeLabel: '证据', ts: null, timeText: '' };
  const [, typeKey, d, t, ms] = m;
  const ts = new Date(
    `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}T${t.slice(0, 2)}:${t.slice(2, 4)}:${t.slice(4, 6)}`,
  ).getTime() + (ms ? Number(ms) : 0);
  if (!Number.isFinite(ts)) return { typeKey, typeLabel: evidenceLabel(typeKey), ts: null, timeText: '' };
  return { typeKey, typeLabel: evidenceLabel(typeKey), ts, timeText: new Date(ts).toLocaleString() };
}

function fmtSize(n: number) {
  if (!n && n !== 0) return '-';
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

function fmtTime(ts: number | null) {
  if (!ts) return '—';
  return new Date(ts).toLocaleTimeString();
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function numberValue(value: unknown): number {
  return numberOrNull(value) ?? 0;
}

function normalizeMetrics(data: any): { rows: MetricRow[]; system: any } {
  const rows: MetricRow[] = [];
  const push = (pipeline: string, n: any) => {
    const latency = n?.latency_ms || {};
    rows.push({
      pipeline: pipeline || n?.pipeline || n?.pipeline_id || '—',
      node: n?.node_name || n?.node || n?.name || '—',
      total: numberValue(n?.total_packets),
      dropped: numberValue(n?.dropped_packets),
      fps: numberValue(n?.fps),
      latencyMin: numberOrNull(n?.min_latency_ms ?? latency.min),
      latencyAvg: numberOrNull(n?.avg_latency_ms ?? latency.avg),
      latencyMax: numberOrNull(n?.max_latency_ms ?? latency.max),
    });
  };

  const pipelines = data?.pipelines;
  if (Array.isArray(pipelines)) {
    pipelines.forEach((n: any) => push(n?.pipeline || data?.pipeline_id || '—', n));
  } else if (pipelines && typeof pipelines === 'object') {
    Object.entries(pipelines).forEach(([pid, nodes]) => {
      if (Array.isArray(nodes)) nodes.forEach((n: any) => push(pid, n));
    });
  } else if (Array.isArray(data?.nodes)) {
    data.nodes.forEach((n: any) => push(data?.pipeline_id || '—', n));
  } else if (Array.isArray(data)) {
    data.forEach((n: any) => push(n?.pipeline || '—', n));
  }
  return { rows, system: data?.system || {} };
}

function latencyText(row: MetricRow): string {
  if (row.latencyAvg != null) return row.latencyAvg.toFixed(1);
  if (row.latencyMin != null || row.latencyMax != null) {
    const min = row.latencyMin ?? 0;
    const max = row.latencyMax ?? 0;
    if (min === 0 && max === 0) return '—';
    return `${min}~${max}`;
  }
  return '—';
}

function dropText(row: MetricRow): string {
  if (row.total > 0) {
    const rate = (row.dropped / row.total) * 100;
    return `${row.dropped} (${rate.toFixed(1)}%)`;
  }
  return String(row.dropped);
}

export default function PreviewView({ settings, initialId, initialTab, onBack, backLabel = '← 返回编辑' }: {
  settings: ApiSettings; initialId?: string; initialTab?: PreviewTab; onBack: () => void; backLabel?: string;
}) {
  const route = useRef(previewRouteParams()).current;
  const routeTab = route.get('tab');
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [sel, setSel] = useState(initialId || route.get('id') || '');
  const [tab, setTab] = useState<Tab>(
    initialId
      ? (initialTab || 'preview')
      : (isPreviewTab(routeTab) ? routeTab : (initialTab || 'preview')),
  );
  const [pipeLoading, setPipeLoading] = useState(false);
  const [pipeErr, setPipeErr] = useState('');
  const [pipeUpdated, setPipeUpdated] = useState<number | null>(null);

  const [files, setFiles] = useState<EvFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [filesErr, setFilesErr] = useState('');
  const [filesUpdated, setFilesUpdated] = useState<number | null>(null);
  const [evQuery, setEvQuery] = useState('');
  const [evKind, setEvKind] = useState<EvidenceKind>('all');
  const [evSelected, setEvSelected] = useState<EvItem | null>(null);
  const [downloading, setDownloading] = useState('');
  const evSeq = useRef(0);

  const [metrics, setMetrics] = useState<any>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsErr, setMetricsErr] = useState('');
  const [metricsUpdated, setMetricsUpdated] = useState<number | null>(null);
  const metricSeq = useRef(0);

  const [status, setStatus] = useState<any>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusErr, setStatusErr] = useState('');
  const [statusUpdated, setStatusUpdated] = useState<number | null>(null);
  const statusSeq = useRef(0);

  const [previewNonce, setPreviewNonce] = useState(0);
  const [previewState, setPreviewState] = useState<LivePlayerStatus>('idle');
  const [copyHint, setCopyHint] = useState('');

  const refreshPipelines = useCallback(async (silent = false) => {
    if (!silent) setPipeLoading(true);
    try {
      const ps = await listPipelines(settings);
      setPipelines(ps);
      setPipeErr('');
      setPipeUpdated(Date.now());
      setSel((prev) => (prev && ps.some((p) => p.id === prev) ? prev : (ps[0]?.id || '')));
    } catch (e: any) {
      if (!silent) setPipeErr('加载管道列表失败：' + (e?.message || e));
    } finally {
      if (!silent) setPipeLoading(false);
    }
  }, [settings]);

  useEffect(() => { refreshPipelines(); }, [refreshPipelines]);
  useEffect(() => {
    const timer = window.setInterval(() => refreshPipelines(true), 5000);
    return () => window.clearInterval(timer);
  }, [refreshPipelines]);
  useEffect(() => { if (initialId) setSel(initialId); }, [initialId]);
  useEffect(() => { if (initialTab) setTab(initialTab); }, [initialTab]);

  // 预览页 tab/管道写入 URL，刷新或分享后能回到同一位置
  useEffect(() => {
    const params = new URLSearchParams();
    if (sel) params.set('id', sel);
    params.set('tab', tab);
    const query = params.toString();
    const next = '#/preview' + (query ? '?' + query : '');
    if (location.hash !== next) history.replaceState(null, '', next);
  }, [sel, tab]);

  // 浏览器前进/后退或手改 hash 时同步当前 tab/管道
  useEffect(() => {
    const onHash = () => {
      const params = previewRouteParams();
      const id = params.get('id');
      const nextTab = params.get('tab');
      if (id) setSel(id);
      if (isPreviewTab(nextTab)) setTab(nextTab);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const loadEvidence = useCallback(async () => {
    const seq = ++evSeq.current;
    if (!sel) {
      setFiles([]);
      setFilesErr('');
      setFilesLoading(false);
      setFilesUpdated(Date.now());
      return;
    }
    setFilesLoading(true);
    setFilesErr('');
    try {
      const r = await api.evidenceList(settings, sel);
      if (seq !== evSeq.current) return;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setFiles(Array.isArray(r.json?.files) ? r.json.files : []);
      setFilesUpdated(Date.now());
    } catch (e: any) {
      if (seq === evSeq.current) setFilesErr('证据加载失败：' + (e?.message || e));
    } finally {
      if (seq === evSeq.current) setFilesLoading(false);
    }
  }, [sel, settings]);

  const loadMetrics = useCallback(async () => {
    const seq = ++metricSeq.current;
    setMetricsLoading(true);
    setMetricsErr('');
    try {
      const r = await api.metrics(settings);
      if (seq !== metricSeq.current) return;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setMetrics(r.json);
      setMetricsUpdated(Date.now());
    } catch (e: any) {
      if (seq === metricSeq.current) setMetricsErr('指标加载失败：' + (e?.message || e));
    } finally {
      if (seq === metricSeq.current) setMetricsLoading(false);
    }
  }, [settings]);

  const loadStatus = useCallback(async () => {
    const seq = ++statusSeq.current;
    if (!sel) {
      setStatus(null);
      setStatusErr('');
      setStatusLoading(false);
      setStatusUpdated(Date.now());
      return;
    }
    setStatusLoading(true);
    setStatusErr('');
    try {
      const r = await api.status(settings, sel);
      if (seq !== statusSeq.current) return;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setStatus(r.json);
      setStatusUpdated(Date.now());
    } catch (e: any) {
      if (seq === statusSeq.current) setStatusErr('状态加载失败：' + (e?.message || e));
    } finally {
      if (seq === statusSeq.current) setStatusLoading(false);
    }
  }, [sel, settings]);

  useEffect(() => {
    evSeq.current += 1;
    setFiles([]);
    setFilesErr('');
    setFilesUpdated(null);
    setEvQuery('');
    setEvKind('all');
    setEvSelected(null);
  }, [sel]);

  useEffect(() => {
    statusSeq.current += 1;
    setStatus(null);
    setStatusErr('');
    setStatusUpdated(null);
  }, [sel]);

  useEffect(() => {
    if (tab !== 'evidence') return;
    loadEvidence();
    const timer = window.setInterval(loadEvidence, 10000);
    return () => window.clearInterval(timer);
  }, [tab, loadEvidence]);

  useEffect(() => {
    if (tab !== 'metrics') return;
    loadMetrics();
    const timer = window.setInterval(loadMetrics, 5000);
    return () => window.clearInterval(timer);
  }, [tab, loadMetrics]);

  useEffect(() => {
    if (tab !== 'status') return;
    loadStatus();
    const timer = window.setInterval(loadStatus, 5000);
    return () => window.clearInterval(timer);
  }, [tab, loadStatus]);

  useEffect(() => {
    if (!evSelected) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setEvSelected(null); };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener('keydown', onKey);
    };
  }, [evSelected]);

  const cur = useMemo(() => pipelines.find((p) => p.id === sel), [pipelines, sel]);
  const isRunning = !!cur?.running;
  const streamUrl = cur?.play_url
    ? (cur.play_url.startsWith('http')
      ? cur.play_url
      : window.location.origin + (cur.play_url.startsWith('/') ? cur.play_url : '/' + cur.play_url))
    : '';

  useEffect(() => {
    if (!isRunning) setPreviewState('idle');
  }, [isRunning]);

  const { rows: metricRows, system } = useMemo(() => normalizeMetrics(metrics), [metrics]);
  const metricGroups = useMemo(() => {
    const map = new Map<string, MetricRow[]>();
    for (const row of metricRows) {
      if (!map.has(row.pipeline)) map.set(row.pipeline, []);
      map.get(row.pipeline)!.push(row);
    }
    return [...map.entries()].sort(([a], [b]) => {
      const selectedDelta = (a === sel ? -1 : 0) - (b === sel ? -1 : 0);
      return selectedDelta || a.localeCompare(b);
    });
  }, [metricRows, sel]);

  const totalDrops = metricRows.reduce((sum, row) => sum + row.dropped, 0);
  const fpsRows = metricRows.filter((row) => row.fps > 0);
  const avgFps = fpsRows.length
    ? fpsRows.reduce((sum, row) => sum + row.fps, 0) / fpsRows.length
    : null;

  const evidenceItems = useMemo<EvItem[]>(() => {
    const query = evQuery.trim().toLowerCase();
    return files.map((file) => {
      const meta = parseEvidenceName(file.name);
      const mediaKind: 'image' | 'video' = file.kind === 'video' || /\.(mp4|avi|mkv|mov|webm)$/i.test(file.name)
        ? 'video'
        : 'image';
      return { ...file, ...meta, mediaKind, key: `${file.dir_index}/${file.name}` };
    }).filter((item) => {
      if (evKind !== 'all' && item.mediaKind !== evKind) return false;
      if (!query) return true;
      return [item.name, item.typeLabel, item.timeText].some((text) => text.toLowerCase().includes(query));
    }).sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0) || b.name.localeCompare(a.name));
  }, [files, evKind, evQuery]);

  const evidenceGroups = useMemo(() => {
    const map = new Map<string, { label: string; items: EvItem[] }>();
    for (const item of evidenceItems) {
      const key = item.ts ? new Date(item.ts).toISOString().slice(0, 10) : 'unknown';
      const label = item.ts
        ? new Date(item.ts).toLocaleDateString(undefined, { year: 'numeric', month: '2-digit', day: '2-digit' })
        : '未知时间';
      if (!map.has(key)) map.set(key, { label, items: [] });
      map.get(key)!.items.push(item);
    }
    return [...map.entries()]
      .map(([key, value]) => ({ key, ...value }))
      .sort((a, b) => {
        if (a.key === 'unknown') return 1;
        if (b.key === 'unknown') return -1;
        return b.key.localeCompare(a.key);
      });
  }, [evidenceItems]);

  const copyStream = async () => {
    if (!streamUrl) return;
    try {
      await navigator.clipboard.writeText(streamUrl);
      setCopyHint('已复制');
    } catch {
      setCopyHint('复制失败');
    }
    window.setTimeout(() => setCopyHint(''), 2000);
  };

  const downloadEvidence = async (item: EvItem) => {
    setDownloading(item.key);
    try {
      const url = api.evidenceFileUrl(settings, sel, item.dir_index, item.name);
      const res = await fetch(url, {
        headers: settings.token ? { Authorization: `Bearer ${settings.token}` } : {},
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = item.name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (e: any) {
      setFilesErr('下载失败：' + (e?.message || e));
    } finally {
      setDownloading('');
    }
  };

  const extraStatus = Object.entries(status || {}).filter(
    ([key]) => !['id', 'running', 'state', 'status'].includes(key),
  );

  return (
    <div className="preview">
      <div className="pv-top">
        <button className="btn" onClick={onBack}>{backLabel}</button>
        <span className="pv-title">预览</span>
        <select value={sel} onChange={(e) => setSel(e.target.value)} aria-label="选择管道">
          {pipelines.length === 0 && <option value="">（无管道）</option>}
          {pipelines.map((p) => (
            <option key={p.id} value={p.id}>{p.running ? '● ' : ''}{p.id}（{p.state}）</option>
          ))}
        </select>
        <button className="btn tiny" onClick={() => refreshPipelines(false)} disabled={pipeLoading}>
          {pipeLoading ? '刷新中…' : '刷新'}
        </button>
        {cur && (
          <span className={'pv-state-badge ' + (cur.running ? 'running' : 'stopped')}>
            {cur.running ? '运行中' : '已停止'}
          </span>
        )}
        <div className="spacer" />
        <span className="pv-updated">列表更新 {fmtTime(pipeUpdated)}</span>
      </div>

      <div className="pv-tabbar" role="tablist" aria-label="预览页分区">
        {TABS.map((item) => (
          <button
            key={item.id}
            role="tab"
            aria-selected={tab === item.id}
            aria-controls="pv-panel"
            className={'pv-tab' + (tab === item.id ? ' active' : '')}
            onClick={() => setTab(item.id)}
          >
            {item.label}
            {item.id === 'evidence' && files.length > 0 && <span className="pv-tab-count">{files.length}</span>}
          </button>
        ))}
      </div>

      {pipeErr && <div className="pv-err">{pipeErr}</div>}

      <div className="pv-body">
        {tab === 'preview' && (
          <div className="pv-pane" id="pv-panel" role="tabpanel">
            {!cur ? (
              <StateBlock title="暂无管道" desc="请先在编辑器中构建并启动一个管道。" icon="◌" />
            ) : !cur.running ? (
              <StateBlock
                title="管道未运行"
                desc="启动管道后，实时预览会自动重新加载；当前不会显示播放器，避免误判为黑屏。"
                icon="■"
              />
            ) : !cur.play_url ? (
              <StateBlock title="未配置播放地址" desc="在 sink 节点的“播放地址”里填入 HLS/WHEP 地址后重新构建。" icon="◇" />
            ) : (
              <>
                <div className="pv-preview-head">
                  <div className="pv-preview-title">
                    <span className={'pv-state-badge ' + (cur.running ? 'running' : 'stopped')}>运行中</span>
                    <strong>{cur.id}</strong>
                    <span className={'pv-live-status ' + previewState}>
                      {previewState === 'playing' ? '直播中' : previewState === 'error' ? '流异常' : previewState === 'connecting' ? '连接中' : ''}
                    </span>
                  </div>
                  <div className="pv-preview-actions">
                    {copyHint && <span className="pv-copy-hint">{copyHint}</span>}
                    <button className="btn tiny" onClick={() => setPreviewNonce((n) => n + 1)}>重连</button>
                    <button className="btn tiny" onClick={copyStream}>复制地址</button>
                    <a className="btn tiny" href={streamUrl} target="_blank" rel="noreferrer">打开流</a>
                  </div>
                </div>
                <div className="pv-live-wrap">
                  <LivePlayer
                    key={`${cur.id}:${cur.play_url}:${cur.running}:${previewNonce}`}
                    url={cur.play_url}
                    onStatus={setPreviewState}
                  />
                  {previewState === 'connecting' && (
                    <div className="pv-live-overlay"><span className="pv-spinner" />正在连接视频流…</div>
                  )}
                  {previewState === 'playing' && <span className="pv-live-badge">● LIVE</span>}
                </div>
                <div className="pv-preview-foot">
                  <span>播放地址</span>
                  <code>{cur.play_url}</code>
                  {previewState === 'error' && <span className="pv-foot-err">流异常，可尝试“重连”或检查播放地址。</span>}
                </div>
              </>
            )}
          </div>
        )}

        {tab === 'evidence' && (
          <div className="pv-pane" id="pv-panel" role="tabpanel">
            <div className="ev-toolbar">
              <input
                className="ev-search"
                value={evQuery}
                onChange={(e) => setEvQuery(e.target.value)}
                placeholder="搜索类型 / 文件名 / 时间"
                aria-label="搜索证据"
              />
              <div className="ev-filters">
                {(['all', 'image', 'video'] as const).map((kind) => (
                  <button
                    key={kind}
                    className={'ev-filter' + (evKind === kind ? ' on' : '')}
                    onClick={() => setEvKind(kind)}
                  >
                    {kind === 'all' ? '全部' : kind === 'image' ? '图片' : '视频'}
                  </button>
                ))}
              </div>
              <div className="spacer" />
              <span className="ev-count">
                {evidenceItems.length} 条 · 更新 {fmtTime(filesUpdated)}
              </span>
              <button className="btn tiny" onClick={loadEvidence} disabled={filesLoading}>
                {filesLoading ? '刷新中…' : '刷新'}
              </button>
            </div>

            {filesErr && files.length > 0 && <div className="pv-panel-err">{filesErr}</div>}

            {filesLoading && files.length === 0 ? (
              <EvidenceSkeleton />
            ) : filesErr && files.length === 0 ? (
              <StateBlock
                title="证据加载失败"
                desc={filesErr}
                icon="!"
                action={<button className="btn sm" onClick={loadEvidence}>重试</button>}
              />
            ) : files.length === 0 && filesUpdated === null ? (
              <EvidenceSkeleton />
            ) : files.length === 0 ? (
              <StateBlock
                title="暂无证据"
                desc="有告警且管道包含 evidence 节点时，会自动生成图片/视频证据。"
                icon="□"
                action={<button className="btn sm" onClick={loadEvidence}>重新扫描</button>}
              />
            ) : evidenceItems.length === 0 ? (
              <StateBlock title="没有匹配的证据" desc="试试清空搜索或切换图片/视频筛选。" icon="⌕" />
            ) : (
              <div className="ev-days">
                {evidenceGroups.map((group) => (
                  <section className="ev-day" key={group.key}>
                    <div className="ev-day-title">
                      <span>{group.label}</span>
                      <span className="ev-day-count">{group.items.length} 条</span>
                    </div>
                    <div className="ev-grid">
                      {group.items.map((item) => {
                        const url = api.evidenceFileUrl(settings, sel, item.dir_index, item.name);
                        return (
                          <article className="ev-card" key={item.key}>
                            <button
                              className="ev-card-media"
                              onClick={() => setEvSelected(item)}
                              aria-label={`查看 ${item.typeLabel} ${item.timeText || item.name}`}
                            >
                              {item.mediaKind === 'image'
                                ? <EvThumb url={url} token={settings.token} alt={item.name} />
                                : <div className="ev-video-ph"><span>▶</span><em>视频</em></div>}
                              <span className="ev-kind">{item.mediaKind === 'image' ? '图片' : '视频'}</span>
                            </button>
                            <div className="ev-card-body">
                              <div className="ev-card-title" title={item.name}>{item.typeLabel}</div>
                              <div className="ev-card-time">{item.timeText || item.name}</div>
                              <div className="ev-card-meta">
                                <span>{fmtSize(item.size)}</span>
                                <span>{item.name.split('.').pop()?.toUpperCase() || '—'}</span>
                              </div>
                              <div className="ev-card-actions">
                                <button className="btn tiny" onClick={() => setEvSelected(item)}>查看</button>
                                <button
                                  className="btn tiny ghost"
                                  disabled={downloading === item.key}
                                  onClick={() => downloadEvidence(item)}
                                >
                                  {downloading === item.key ? '下载中…' : '下载'}
                                </button>
                              </div>
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  </section>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === 'metrics' && (
          <div className="pv-pane" id="pv-panel" role="tabpanel">
            <div className="pv-panel-head">
              <div>
                <h3>运行指标</h3>
                <span className="pv-sub">每 5 秒自动刷新 · 最后更新 {fmtTime(metricsUpdated)}</span>
              </div>
              <button className="btn tiny" onClick={loadMetrics} disabled={metricsLoading}>
                {metricsLoading ? '刷新中…' : '刷新'}
              </button>
            </div>

            {metricsErr && metrics && <div className="pv-panel-err">{metricsErr}</div>}

            {metricsErr && !metrics ? (
              <StateBlock
                title="指标加载失败"
                desc={metricsErr}
                icon="!"
                action={<button className="btn sm" onClick={loadMetrics}>重试</button>}
              />
            ) : !metrics && (metricsLoading || metricsUpdated === null) ? (
              <MetricsSkeleton />
            ) : (
              <>
                <div className="pv-kpis">
                  <Kpi label="活跃管道" value={`${system.active_pipelines ?? '—'}${system.total_pipelines != null ? ` / ${system.total_pipelines}` : ''}`} sub="active / total" />
                  <Kpi label="节点数" value={String(metricRows.length)} sub="有指标上报的节点" />
                  <Kpi label="总丢帧" value={String(totalDrops)} sub="累计 dropped_packets" tone={totalDrops > 0 ? 'warn' : 'ok'} />
                  <Kpi label="平均 FPS" value={avgFps != null ? avgFps.toFixed(1) : '—'} sub="仅统计 FPS > 0 的节点" />
                  <Kpi label="CPU" value={system.cpu_percent != null ? `${system.cpu_percent}%` : '—'} sub="system.cpu_percent" />
                  <Kpi label="GPU 显存" value={system.gpu_memory_mb != null ? (system.gpu_memory_mb >= 1024 ? `${(system.gpu_memory_mb / 1024).toFixed(1)} GB` : `${system.gpu_memory_mb} MB`) : '—'} sub="system.gpu_memory_mb" />
                </div>

                {metricGroups.length === 0 ? (
                  <StateBlock
                    title="暂无指标"
                    desc="管道未运行或指标尚未上报。启动管道后，这里会自动出现节点吞吐、丢帧和延迟。"
                    icon="▤"
                    action={<button className="btn sm" onClick={loadMetrics}>重新加载</button>}
                  />
                ) : (
                  <div className="pv-metric-groups">
                    {metricGroups.map(([pid, items]) => {
                      const pipeline = pipelines.find((p) => p.id === pid);
                      const known = !!pipeline;
                      const drops = items.reduce((sum, row) => sum + row.dropped, 0);
                      return (
                        <section className={'pv-metric-group' + (pid === sel ? ' selected' : '')} key={pid}>
                          <div className="pv-metric-head">
                            <div className="pv-metric-title">
                              <span className={'pv-status-dot ' + (pipeline?.running ? 'running' : 'stopped')} />
                              <strong>{pid}</strong>
                              <span className={'pv-state-badge ' + (pipeline?.running ? 'running' : 'stopped')}>
                                {known ? (pipeline?.running ? '运行中' : '已停止') : '未知'}
                              </span>
                            </div>
                            <span className="pv-metric-sub">{items.length} 节点 · 丢帧 {drops}</span>
                          </div>
                          <div className="pv-table-wrap">
                            <table className="pv-table" aria-label={`${pid} 节点指标`}>
                              <thead>
                                <tr>
                                  <th>节点</th>
                                  <th>FPS</th>
                                  <th>包数</th>
                                  <th>丢帧</th>
                                  <th>延迟(ms)</th>
                                </tr>
                              </thead>
                              <tbody>
                                {items.map((row, index) => (
                                  <tr key={`${row.pipeline}-${row.node}-${index}`}>
                                    <td className="pv-node-name">{row.node}</td>
                                    <td>{row.fps.toFixed(1)}</td>
                                    <td>{row.total}</td>
                                    <td className={row.dropped > 0 ? 'bad' : ''}>{dropText(row)}</td>
                                    <td>{latencyText(row)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </section>
                      );
                    })}
                  </div>
                )}

                <details className="pv-raw">
                  <summary>原始指标 JSON</summary>
                  <pre className="pv-json">{JSON.stringify(metrics, null, 2)}</pre>
                </details>
              </>
            )}
          </div>
        )}

        {tab === 'status' && (
          <div className="pv-pane" id="pv-panel" role="tabpanel">
            <div className="pv-panel-head">
              <div>
                <h3>运行状态</h3>
                <span className="pv-sub">每 5 秒自动刷新 · 最后更新 {fmtTime(statusUpdated)}</span>
              </div>
              <button className="btn tiny" onClick={loadStatus} disabled={statusLoading}>
                {statusLoading ? '刷新中…' : '刷新'}
              </button>
            </div>

            {statusErr && status && <div className="pv-panel-err">{statusErr}</div>}

            {statusErr && !status ? (
              <StateBlock
                title="状态加载失败"
                desc={statusErr}
                icon="!"
                action={<button className="btn sm" onClick={loadStatus}>重试</button>}
              />
            ) : !status && (statusLoading || statusUpdated === null) ? (
              <StatusSkeleton />
            ) : (
              <>
                <div className="pv-status-grid">
                  <StatusCard label="运行状态">
                    <div className="pv-status-value">
                      <span className={'pv-status-dot ' + (cur?.running ? 'running' : 'stopped')} />
                      {cur?.running ? '运行中' : '已停止'}
                    </div>
                    <div className="pv-status-sub">{cur?.state || '—'}</div>
                  </StatusCard>
                  <StatusCard label="管道">
                    <div className="pv-status-value mono">{sel || '—'}</div>
                    <div className="pv-status-sub">当前选择</div>
                  </StatusCard>
                  <StatusCard label="播放地址">
                    <code className="pv-status-code">{cur?.play_url || '—'}</code>
                    <div className="pv-status-sub">sink.play_url</div>
                  </StatusCard>
                  <StatusCard label="证据目录">
                    {cur?.evidence_dirs?.length
                      ? <ul className="pv-dirs">{cur.evidence_dirs.map((dir) => <li key={dir}>{dir}</li>)}</ul>
                      : <div className="pv-status-value muted">—</div>}
                  </StatusCard>
                  {extraStatus.length > 0 && (
                    <StatusCard label="其他状态" wide>
                      <div className="pv-kv-list">
                        {extraStatus.map(([key, value]) => (
                          <div className="pv-kv" key={key}>
                            <span>{key}</span>
                            <code>{typeof value === 'string' ? value : JSON.stringify(value)}</code>
                          </div>
                        ))}
                      </div>
                    </StatusCard>
                  )}
                </div>

                <details className="pv-raw">
                  <summary>原始状态 JSON</summary>
                  <pre className="pv-json">{JSON.stringify({ pipeline: sel, ...(status || {}) }, null, 2)}</pre>
                </details>
              </>
            )}
          </div>
        )}
      </div>

      {evSelected && (
        <EvidenceModal
          item={evSelected}
          settings={settings}
          pipeline={sel}
          downloading={downloading === evSelected.key}
          onClose={() => setEvSelected(null)}
          onDownload={() => downloadEvidence(evSelected)}
        />
      )}
    </div>
  );
}

function StateBlock({ title, desc, icon = '·', action }: {
  title: string; desc?: string; icon?: string; action?: ReactNode;
}) {
  return (
    <div className="pv-state">
      <div className="pv-state-icon">{icon}</div>
      <div className="pv-state-title">{title}</div>
      {desc && <div className="pv-state-desc">{desc}</div>}
      {action && <div className="pv-state-action">{action}</div>}
    </div>
  );
}

function Kpi({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: 'ok' | 'warn';
}) {
  return (
    <div className={'pv-kpi' + (tone ? ' ' + tone : '')}>
      <div className="pv-kpi-label">{label}</div>
      <div className="pv-kpi-value">{value}</div>
      {sub && <div className="pv-kpi-sub">{sub}</div>}
    </div>
  );
}

function StatusCard({ label, children, wide = false }: {
  label: string; children: ReactNode; wide?: boolean;
}) {
  return (
    <div className={'pv-status-card' + (wide ? ' wide' : '')}>
      <div className="pv-status-label">{label}</div>
      {children}
    </div>
  );
}

function EvThumb({ url, token, alt }: { url: string; token: string; alt: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [show, setShow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === 'undefined') {
      setShow(true);
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setShow(true);
        observer.disconnect();
      }
    }, { rootMargin: '240px' });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={ref} className="ev-thumb-wrap">
      {show ? <MediaImg url={url} token={token} alt={alt} className="ev-thumb-img" /> : <div className="ev-thumb-ph" />}
    </div>
  );
}

function EvidenceModal({ item, settings, pipeline, downloading, onClose, onDownload }: {
  item: EvItem; settings: ApiSettings; pipeline: string; downloading: boolean;
  onClose: () => void; onDownload: () => void;
}) {
  const url = api.evidenceFileUrl(settings, pipeline, item.dir_index, item.name);
  return (
    <div className="ev-modal-backdrop" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="ev-modal" role="dialog" aria-modal="true" aria-label={`${item.typeLabel} 证据`}>
        <div className="ev-modal-head">
          <div>
            <div className="ev-modal-title">{item.typeLabel}</div>
            <div className="ev-modal-sub">{item.timeText || item.name} · {fmtSize(item.size)}</div>
          </div>
          <button className="ev-modal-close" onClick={onClose} aria-label="关闭">×</button>
        </div>
        <div className="ev-modal-media">
          {item.mediaKind === 'video'
            ? <MediaVideo className="ev-modal-video" url={url} token={settings.token} />
            : <MediaImg className="ev-modal-img" url={url} token={settings.token} alt={item.name} />}
        </div>
        <div className="ev-modal-foot">
          <span className="ev-modal-name" title={item.name}>{item.name}</span>
          <button className="btn sm" onClick={onDownload} disabled={downloading}>
            {downloading ? '下载中…' : '下载'}
          </button>
          <button className="btn sm ghost" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  );
}

function EvidenceSkeleton() {
  return (
    <div className="ev-grid">
      {Array.from({ length: 6 }).map((_, index) => (
        <div className="ev-card ev-skeleton" key={index}>
          <div className="ev-sk-media" />
          <div className="ev-sk-body">
            <div className="ev-sk-line" />
            <div className="ev-sk-line short" />
            <div className="ev-sk-line tiny" />
          </div>
        </div>
      ))}
    </div>
  );
}

function MetricsSkeleton() {
  return (
    <>
      <div className="pv-kpis">
        {Array.from({ length: 6 }).map((_, index) => <div className="pv-kpi ev-skeleton" key={index}><div className="ev-sk-line" /><div className="ev-sk-line short" /></div>)}
      </div>
      <div className="pv-skeleton-table" />
    </>
  );
}

function StatusSkeleton() {
  return (
    <div className="pv-status-grid">
      {Array.from({ length: 4 }).map((_, index) => <div className="pv-status-card ev-skeleton" key={index}><div className="ev-sk-line" /><div className="ev-sk-line short" /></div>)}
    </div>
  );
}
