import { useEffect, useRef, useState } from 'react';
import { ApiSettings, api } from '../api/client';

const LEVELS = ['info', 'warn', 'error'] as const;
type Level = typeof LEVELS[number];
interface LogLine { ts: string; level: string; src: string; msg: string; }
interface LogData { file?: string; dir?: string; matched?: number; pipeline?: string; lines?: LogLine[]; }

// 研发日志查看抽屉：按实例（子串）+ 等级过滤；自动增量刷新 3s（无新行不重绘）。
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
  const lastTs = useRef('');   // 增量拉取的游标（ts 字符串可直接比较）

  const activeLevels = LEVELS.filter((l) => enabled[l]);
  const levelKey = activeLevels.join(',');
  const lineKey = (ln: LogLine) => `${ln.ts}|${ln.level}|${ln.src}|${ln.msg}`;

  // mode=replace：过滤条件变化/手动刷新，整表替换
  // mode=append：自动刷新，只取 ts>=lastTs 的新行；无新行时完全不触发渲染，避免闪烁
  const fetchLogs = async (mode: 'replace' | 'append' = 'replace') => {
    if (activeLevels.length === 0) {
      if (mode === 'replace') {
        lastTs.current = '';
        setData({ lines: [], matched: 0 });
        setErr('');
      }
      return;
    }
    if (mode === 'replace') setBusy(true);
    const params: Record<string, string | number> = {
      pipeline: onlyThis ? pipeline : '',
      levels: levelKey,
      limit: 500,
    };
    if (mode === 'append' && lastTs.current) params.since = lastTs.current;
    const r = await api.logs(settings, params);
    if (mode === 'replace') setBusy(false);
    if (!r.ok) {
      const msg = (r.json && (r.json as any).error) || `HTTP ${r.status}`;
      setErr((prev) => (prev === msg ? prev : msg));
      return;
    }
    const d = (r.json || {}) as LogData;
    const incoming = d.lines || [];
    if (mode === 'append' && lastTs.current) {
      if (incoming.length === 0) { setErr(''); return; }
      setData((prev) => {
        const prevLines = prev?.lines || [];
        const seen = new Set(prevLines.map(lineKey));
        const merged = prevLines.slice();
        let added = false;
        for (const ln of incoming) {
          const k = lineKey(ln);
          if (seen.has(k)) continue;
          seen.add(k);
          merged.push(ln);
          added = true;
        }
        if (!added) return prev;   // 全为重复行 → 不更新，避免无谓重绘
        return { ...(prev || {}), ...d, lines: merged.slice(-1000) };
      });
    } else {
      setData(d);
    }
    const tail = incoming.length ? incoming[incoming.length - 1].ts : '';
    if (tail) lastTs.current = tail;
    setErr('');
  };

  useEffect(() => {
    lastTs.current = '';
    fetchLogs('replace');
    /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, [settings.baseUrl, settings.token, pipeline, onlyThis, levelKey]);

  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => { fetchLogs('append'); }, 3000);
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

  // Esc 关闭（抽屉不阻塞页面其余操作）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <aside className="log-drawer" role="dialog" aria-label={`日志 ${pipeline}`}>
        <div className="log-head">
          <span className="log-title">日志 · {pipeline}</span>
          <div className="spacer" />
          <span className="log-file" title={data?.dir}>{data?.file || '—'} · {lines.length} 行</span>
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
          <button className="btn sm" onClick={() => fetchLogs('replace')} disabled={busy}>{busy ? '刷新中…' : '刷新'}</button>
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
    </aside>
  );
}
