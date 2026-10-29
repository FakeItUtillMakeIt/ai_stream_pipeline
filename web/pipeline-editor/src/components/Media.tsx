import { useEffect, useState } from 'react';

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
