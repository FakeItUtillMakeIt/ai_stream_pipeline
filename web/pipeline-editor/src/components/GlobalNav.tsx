import { useEffect, useState } from 'react';
import { api, ApiSettings } from '../api/client';

export type NavView = 'editor' | 'impl' | 'preview' | 'alerts';

interface HealthState {
  api: boolean;
  hls: boolean;
  rtmp: boolean;
  cpu: number;
  gpu: number;
}

function gpuText(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

export default function GlobalNav({ settings, view, role, onNavigate }: {
  settings: ApiSettings;
  view: NavView;
  role: 'dev' | 'impl' | 'anon' | 'unknown';
  onNavigate: (view: NavView) => void;
}) {
  const [health, setHealth] = useState<HealthState>({ api: false, hls: false, rtmp: false, cpu: 0, gpu: 0 });

  useEffect(() => {
    let alive = true;
    const load = async () => {
      const [h, m] = await Promise.all([api.health(settings), api.metrics(settings)]);
      if (!alive) return;
      const checks = h.ok ? (h.json?.checks || {}) : {};
      const system = m.ok ? (m.json?.system || {}) : {};
      setHealth({
        api: h.ok,
        hls: !!checks.mediamtx_hls,
        rtmp: !!checks.mediamtx_rtmp,
        cpu: Number(system.process_cpu_percent ?? system.cpu_percent) || 0,
        gpu: Number(system.gpu_memory_mb) || 0,
      });
    };
    load();
    const timer = window.setInterval(load, 5000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [settings.baseUrl, settings.token]);

  const pill = (label: string, ok: boolean, title: string) => (
    <span className={'gnav-pill ' + (ok ? 'ok' : 'bad')} title={title}>
      <i />{label}
    </span>
  );

  return (
    <div className="gnav">
      <div className="gnav-brand">AI Stream</div>
      <div className="gnav-tabs">
        {role === 'dev' && (
          <button className={view === 'editor' ? 'on' : ''} onClick={() => onNavigate('editor')}>编辑器</button>
        )}
        <button className={view === 'impl' ? 'on' : ''} onClick={() => onNavigate('impl')}>实施台</button>
        <button className={view === 'preview' ? 'on' : ''} onClick={() => onNavigate('preview')}>预览</button>
        <button className={view === 'alerts' ? 'on' : ''} onClick={() => onNavigate('alerts')}>告警</button>
      </div>
      <div className="spacer" />
      <div className="gnav-health">
        {pill('API', health.api, '后端 /health')}
        {pill('HLS', health.hls, 'mediamtx HLS 127.0.0.1:8888')}
        {pill('RTMP', health.rtmp, 'mediamtx RTMP 127.0.0.1:1935')}
        <span className="gnav-metric">CPU {(health.cpu / 100).toFixed(1)} 核</span>
        <span className="gnav-metric">GPU {health.gpu >= 0 ? gpuText(health.gpu) : '—'}</span>
      </div>
      <span className="gnav-role">{role === 'dev' ? '研发' : role === 'impl' ? '实施' : role === 'anon' ? '未授权' : '连接中…'}</span>
    </div>
  );
}
