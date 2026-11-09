import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

// 实时预览播放器：HLS(.m3u8) 用 hls.js，原生 mp4/webm 用 <video>，其余提示
// 网络/清单致命错误时自动重试（流刚启动时清单可能还没生成），避免一挂载就"无信号"卡死
export function LivePlayer({ url, compact = false }: { url: string; compact?: boolean }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const v = ref.current; setError('');
    if (!v || !url) return;
    let hls: Hls | null = null;
    let retryTimer: number | undefined;
    let disposed = false;
    const isHls = url.includes('.m3u8');
    if (isHls && Hls.isSupported()) {
      hls = new Hls({ lowLatencyMode: true, enableWorker: true });
      hls.loadSource(url);
      hls.attachMedia(v);
      hls.on(Hls.Events.MANIFEST_PARSED, () => { v.play().catch(() => { /* 自动播放可能被拦 */ }); });
      hls.on(Hls.Events.ERROR, (_e, data) => {
        if (!data.fatal) return;
        // 网络/清单错误：流可能刚启动未就绪 → 自动重试（最多 8 次，间隔 2.5s）
        if ((data.type === Hls.ErrorTypes.NETWORK_ERROR || data.type === Hls.ErrorTypes.OTHER_ERROR) && attempt < 8) {
          hls?.destroy();
          retryTimer = window.setTimeout(() => { if (!disposed) setAttempt((a) => a + 1); }, 2500);
          return;
        }
        let msg = `HLS ${data.type}/${data.details}`;
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR) msg += '（网络/CORS：确认媒体服务器已开放跨源，或用 ffplay 验证地址可达）';
        else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) { msg += '（媒体错误：尝试恢复）'; hls?.recoverMediaError(); }
        else if (data.type === Hls.ErrorTypes.OTHER_ERROR) msg += '（无法加载清单，检查 URL/是否 404）';
        setError(msg);
      });
    } else if (/\.(mp4|webm|ogg)$/i.test(url) || (isHls && v.canPlayType('application/vnd.apple.mpegurl'))) {
      v.src = url; v.play().catch(() => {});
    } else {
      setError('无法播放该地址：预览目前支持 HLS(.m3u8) 与 mp4/webm。');
    }
    return () => { disposed = true; if (retryTimer) clearTimeout(retryTimer); hls?.destroy(); };
  }, [url, attempt]);
  if (!url) return <div className="pv-empty">该管道未配置 play_url（在 sink 节点填"播放地址"）</div>;
  if (compact) {
    return (
      <div className="pcv-live">
        {error && <div className="pcv-err">无信号</div>}
        <video ref={ref} autoPlay muted playsInline className="pcv-video" />
      </div>
    );
  }
  return (
    <div className="pv-player">
      {error && <div className="pv-play-err">{error}</div>}
      <video ref={ref} controls autoPlay muted playsInline className="pv-video" />
    </div>
  );
}

// 用 fetch + Authorization 拉取受保护媒体为 Blob URL（绕开 <img>/<video> 无法带头、
// 以及本版本 httplib 对开区间 Range 生成畸形 Content-Range 的问题）。
export function useAuthBlobUrl(url: string, token: string): string {
  const [obj, setObj] = useState('');
  useEffect(() => {
    let alive = true; let created = '';
    setObj('');
    if (!url) return;
    fetch(url, { headers: token ? { Authorization: 'Bearer ' + token } : {} })
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error('HTTP ' + r.status))))
      .then((b) => { if (!alive) return; created = URL.createObjectURL(b); setObj(created); })
      .catch(() => { /* 失败则保持空 */ });
    return () => { alive = false; if (created) URL.revokeObjectURL(created); };
  }, [url, token]);
  return obj;
}

export function MediaImg({ url, token, className, alt, onClick }: {
  url: string; token: string; className?: string; alt?: string; onClick?: () => void;
}) {
  const u = useAuthBlobUrl(url, token);
  if (!u) return <div className={className} style={{ background: '#000' }} onClick={onClick} />;
  return <img className={className} src={u} alt={alt || ''} onClick={onClick} />;
}

export function MediaVideo({ url, token, className }: {
  url: string; token: string; className?: string;
}) {
  const u = useAuthBlobUrl(url, token);
  if (!u) return <div className={className} style={{ background: '#000', minHeight: 60 }} />;
  return <video className={className} src={u} controls autoPlay muted playsInline />;
}
