import { useEffect, useRef, useState } from 'react';
import { ApiSettings, api } from '../api/client';

const LEVELS = ['info', 'warn', 'error'] as const;
type Level = typeof LEVELS[number];
interface LogLine { ts: string; level: string; src: string; msg: string; }
interface LogData { file?: string; dir?: string; matched?: number; pipeline?: string; lines?: LogLine[]; }

// 研发日志查看弹窗：按实例（子串）+ 等级过滤；自动刷新 3s。
export default function LogViewer({ settings, pipeline, onClose }: {
  settings: ApiSettings;
  pipeline: string;
  onClose: () => void;
}) {
  const [enabled, setEnabled] = useState<Record<Level, boolean>>({ info: true, warn: true, error: true });
  const [onlyThis, setOnlyThis] = useState(true);
  const [auto, setAuto] = useState(true);
  const [data, setData] = useState<LogData | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const stickBottom = useRef(true);

  const activeLevels = LEVELS.filter((l) => enabled[l]);
  const levelKey = activeLevels.join(',');

  const fetchLogs = async () => {
    if (activeLevels.length === 0) {
      setData({ lines: [], matched: 0 });
      setErr('');
      return;
    }
    setBusy(true);
    const r = await api.logs(settings, {
      pipeline: onlyThis ? pipeline : '',
      levels: levelKey,
      limit: 500,
    });
    setBusy(false);
    if (!r.ok) {
      setErr((r.json && (r.json as any).error) || `HTTP ${r.status}`);
      return;
    }
    setErr('');
    setData((r.json || {}) as LogData);
  };

  useEffect(() => { fetchLogs(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ },
    [settings.baseUrl, settings.token, pipeline, onlyThis, levelKey]);

  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => { fetchLogs(); }, 3000);
    return () => clearInterval(t);
    /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, [auto, settings.baseUrl, settings.token, pipeline, onlyThis, levelKey]);

  const lines = data?.lines || [];
  useEffect(() => {
    // 仅在用户停留在底部时自动跟随，避免打断向上翻查
    if (auto && stickBottom.current && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [data, auto]);

  const toggle = (l: Level) => setEnabled((v) => ({ ...v, [l]: !v[l] }));

  return (
    <div className="log-mask" onClick={onClose}>
      <div className="log-modal" onClick={(e) => e.stopPropagation()}>
        <div className="log-head">
          <span className="log-title">日志 · {pipeline}</span>
          <div className="spacer" />
          <span className="log-file" title={data?.dir}>{data?.file || '—'} · 命中 {data?.matched ?? 0}</span>
          <button className="btn sm ghost" onClick={onClose}>✕</button>
        </div>
        <div className="log-bar">
          {LEVELS.map((l) => (
            <button key={l} className={'log-lv ' + l + (enabled[l] ? ' on' : '')}
              onClick={() => toggle(l)} title={`过滤 ${l} 日志`}>
              {l}
            </button>
          ))}
          <label className="log-chk" title="按实例 ID / 节点名子串过滤日志行">
            <input type="checkbox" checked={onlyThis} onChange={(e) => setOnlyThis(e.target.checked)} /> 仅本实例
          </label>
          <label className="log-chk">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> 自动刷新 3s
          </label>
          <button className="btn sm" onClick={fetchLogs} disabled={busy}>{busy ? '刷新中…' : '刷新'}</button>
          <span className="spacer" />
          <span className="log-hint">节点日志不一定含实例名；看不到时可取消「仅本实例」</span>
        </div>
        {err && <div className="err log-err">{err}</div>}
        <div className="log-list" ref={listRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            stickBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
          }}>
          {lines.length === 0 && <div className="log-empty">暂无匹配日志</div>}
          {lines.map((ln, i) => (
            <div key={i} className={'log-line ' + ln.level}>
              <span className="log-ts">{ln.ts}</span>
              <span className={'log-badge ' + ln.level}>{ln.level}</span>
              <span className="log-src" title={ln.src}>{ln.src}</span>
              <span className="log-msg">{ln.msg}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
