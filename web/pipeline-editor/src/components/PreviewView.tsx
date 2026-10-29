import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import { MediaImg, MediaVideo } from './Media';
import { ApiSettings, api, listPipelines, PipelineInfo } from '../api/client';

type Tab = 'preview' | 'evidence' | 'metrics' | 'status';

interface EvFile { dir_index: number; name: string; kind: string; size: number; }

// 实时预览播放器：HLS(.m3u8) 用 hls.js，原生 mp4/webm 用 <video>，其余提示
function LivePlayer({ url }: { url: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const v = ref.current; setError('');
    if (!v || !url) return;
    let hls: Hls | null = null;
    const isHls = url.includes('.m3u8');
    if (isHls && Hls.isSupported()) {
      hls = new Hls({ lowLatencyMode: true, enableWorker: true });
      hls.loadSource(url);
      hls.attachMedia(v);
      hls.on(Hls.Events.MANIFEST_PARSED, () => { v.play().catch(() => { /* 自动播放可能被拦 */ }); });
      hls.on(Hls.Events.ERROR, (_e, data) => {
        if (!data.fatal) return;
        let msg = `HLS ${data.type}/${data.details}`;
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR) msg += '（网络/CORS：确认媒体服务器已开放跨源，或用 ffplay 验证地址可达）';
        else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) { msg += '（媒体错误：尝试恢复）'; hls?.recoverMediaError(); }
        else if (data.type === Hls.ErrorTypes.OTHER_ERROR) msg += '（无法加载清单，检查 URL/是否 404）';
        setError(msg);
      });
    } else if (/\.(mp4|webm|ogg)$/i.test(url) || (isHls && v.canPlayType('application/vnd.apple.mpegurl'))) {
      v.src = url; v.play().catch(() => {});
    } else {
      setError(`无法播放该地址：预览目前支持 HLS(.m3u8) 与 mp4/webm。若这是 WebRTC(WHEP) 地址，请改用同一路的 .m3u8。`);
    }
    return () => { hls?.destroy(); };
  }, [url]);
  if (!url) return <div className="pv-empty">该管道未配置 play_url（在 sink 节点填"播放地址"）</div>;
  return (
    <div className="pv-player">
      {error && <div className="pv-play-err">{error}</div>}
      <video ref={ref} controls autoPlay muted playsInline className="pv-video" />
    </div>
  );
}

export default function PreviewView({ settings, initialId, onBack, backLabel = '← 返回编辑' }: {
  settings: ApiSettings; initialId?: string; onBack: () => void; backLabel?: string;
}) {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [sel, setSel] = useState(initialId || '');
  const [tab, setTab] = useState<Tab>('preview');
  const [files, setFiles] = useState<EvFile[]>([]);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [metrics, setMetrics] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);
  const [err, setErr] = useState('');

  const refreshPipelines = async () => {
    try {
      const ps = await listPipelines(settings);
      setPipelines(ps);
      if (!sel && ps.length) setSel(ps[0].id);
    } catch (e: any) { setErr('加载管道列表失败: ' + e.message); }
  };

  useEffect(() => { refreshPipelines(); /* eslint-disable-next-line */ }, [settings.baseUrl, settings.token]);
  // 定时轮询：外部 build/start/stop/delete 自动反映到下拉与运行态
  const refreshRef = useRef(refreshPipelines); refreshRef.current = refreshPipelines;
  useEffect(() => { const t = setInterval(() => refreshRef.current(), 4000); return () => clearInterval(t); }, []);
  // 管道/标签变化时拉对应数据
  useEffect(() => {
    if (!sel) return;
    setErr(''); setOpen({});
    if (tab === 'evidence') api.evidenceList(settings, sel).then((r) => setFiles(r.json?.files || []));
    if (tab === 'metrics') api.metrics(settings).then((r) => setMetrics(r.json));
    if (tab === 'status') api.status(settings, sel).then((r) => setStatus(r.json));
  }, [sel, tab, settings]);

  const cur = pipelines.find((p) => p.id === sel);

  return (
    <div className="preview">
      <div className="pv-top">
        <button className="btn" onClick={onBack}>{backLabel}</button>
        <select value={sel} onChange={(e) => setSel(e.target.value)}>
          {pipelines.length === 0 && <option value="">（无运行管道）</option>}
          {pipelines.map((p) => (
            <option key={p.id} value={p.id}>{p.running ? '● ' : ''}{p.id}（{p.state}）</option>
          ))}
        </select>
        <button className="btn tiny" onClick={refreshPipelines}>刷新</button>
        <div className="spacer" />
        <span className="pv-tabs">
          {(['preview', 'evidence', 'metrics', 'status'] as Tab[]).map((t) => (
            <button key={t} className={'btn' + (tab === t ? ' primary' : '')} onClick={() => setTab(t)}>
              {{ preview: '实时预览', evidence: '证据', metrics: '指标', status: '状态' }[t]}
            </button>
          ))}
        </span>
      </div>

      {err && <div className="pv-err">{err}</div>}

      <div className="pv-body">
        {tab === 'preview' && (
          <div className="pv-pane">
            <LivePlayer url={cur?.play_url || ''} />
            {cur && !cur.play_url && <div className="pv-empty">该管道未配置播放地址</div>}
          </div>
        )}

        {tab === 'evidence' && (
          <div className="pv-pane">
            <div className="ev-toolbar">
              <button className="btn tiny" onClick={() => api.evidenceList(settings, sel).then((r) => setFiles(r.json?.files || []))}>重新扫描</button>
              <span className="ev-hint">点条目展开查看/播放（懒加载）</span>
            </div>
            {files.length === 0 && <div className="pv-empty">暂无证据文件（确认 evidence 节点 output_dir 与服务器工作目录一致）</div>}
            <div className="ev-cols">
              <div className="ev-col">
                <EvGroup title="图片" kind="image" files={files.filter((f) => f.kind === 'image')} open={open} setOpen={setOpen} settings={settings} sel={sel} />
              </div>
              <div className="ev-col">
                <EvGroup title="视频" kind="video" files={files.filter((f) => f.kind === 'video')} open={open} setOpen={setOpen} settings={settings} sel={sel} />
              </div>
            </div>
          </div>
        )}

        {tab === 'metrics' && (
          <div className="pv-pane">
            <MetricsTable data={metrics} />
          </div>
        )}

        {tab === 'status' && (
          <div className="pv-pane">
            <pre className="pv-json">{JSON.stringify({ id: sel, ...(status || {}) }, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

function fmtSize(n: number) {
  if (!n && n !== 0) return '-';
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

function EvGroup({ title, kind, files, open, setOpen, settings, sel }: {
  title: string; kind: 'video' | 'image'; files: EvFile[];
  open: Record<string, boolean>; setOpen: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  settings: ApiSettings; sel: string;
}) {
  return (
    <div className="ev-group">
      <div className="ev-group-title">{title}（{files.length}）</div>
      {files.length === 0 ? <div className="ev-none">无</div> : (
        <ul className="ev-list">
          {files.map((f) => {
            const key = `${f.dir_index}/${f.name}`;
            const u = api.evidenceFileUrl(settings, sel, f.dir_index, f.name);
            const isOpen = !!open[key];
            return (
              <li key={key} className="ev-item">
                <div className="ev-row" onClick={() => setOpen((o) => ({ ...o, [key]: !o[key] }))}>
                  <span className="ev-caret">{isOpen ? '▾' : '▸'}</span>
                  <span className="ev-name">{f.name}</span>
                  <span className="ev-size">{fmtSize(f.size)}</span>
                  <span className="ev-act">{isOpen ? '收起' : (kind === 'video' ? '播放' : '查看')}</span>
                </div>
                {isOpen && (kind === 'video'
                  ? <MediaVideo className="ev-media" url={u} token={settings.token} />
                  : <MediaImg className="ev-media" url={u} token={settings.token} alt={f.name} />)}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function MetricsTable({ data }: { data: any }) {
  if (!data) return <div className="pv-empty">无指标数据</div>;
  const rows: any[] = data.nodes || data.pipelines || (Array.isArray(data) ? data : []);
  if (!rows.length) return <pre className="pv-json">{JSON.stringify(data, null, 2)}</pre>;
  return (
    <table className="pv-table">
      <thead><tr><th>管道</th><th>节点</th><th>包数</th><th>丢包</th><th>均延迟(ms)</th><th>错误</th></tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td>{r.pipeline || r.id || '-'}</td>
            <td>{r.node || r.name || '-'}</td>
            <td>{r.total_packets ?? '-'}</td>
            <td>{r.dropped_packets ?? '-'}</td>
            <td>{r.avg_latency_ms ?? '-'}</td>
            <td>{r.error_count ?? '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
