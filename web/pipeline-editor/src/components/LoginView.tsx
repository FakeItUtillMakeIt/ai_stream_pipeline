import { useState } from 'react';
import { ApiSettings, api } from '../api/client';

export default function LoginView({ settings, setSettings }: {
  settings: ApiSettings; setSettings: (s: ApiSettings) => void;
}) {
  const [baseUrl, setBaseUrl] = useState(settings.baseUrl);
  const [token, setToken] = useState(settings.token);
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setErr(''); setLoading(true);
    try {
      const r = await api.whoami({ baseUrl, token });
      if (!r.ok) { setErr(r.status === 401 ? 'Token 无效或未授权' : `无法连接服务（${r.status || '网络'}）`); return; }
      const role = r.json?.role;
      if (role !== 'dev' && role !== 'impl') { setErr('Token 无效'); return; }
      setSettings({ baseUrl, token });   // App 据 whoami 角色进入研发/实施界面
    } catch (e: any) {
      setErr('登录失败: ' + (e?.message || e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-wrap">
      <div className="login-card">
        <div className="login-brand">AI 视频分析平台</div>
        <div className="login-desc">输入服务地址与 Token 登录，系统按账号权限自动进入研发或实施界面。</div>
        <div className="login-field">
          <label>服务地址 (API base)</label>
          <input type="text" placeholder="同源可留空；跨源填 http://<ip>:8080" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
        <div className="login-field">
          <label>Token</label>
          <input type="password" placeholder="Bearer Token" value={token} onChange={(e) => setToken(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') submit(); }} />
        </div>
        {err && <div className="login-err">{err}</div>}
        <button className="btn primary login-btn" onClick={submit} disabled={loading}>{loading ? '登录中…' : '登 录'}</button>
        <div className="login-foot">若服务端未启用鉴权（无 token），会自动以研发身份进入。</div>
      </div>
    </div>
  );
}
