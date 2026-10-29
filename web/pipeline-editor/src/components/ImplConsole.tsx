import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiSettings, api, listPipelines, PipelineInfo, deriveState, waitState } from '../api/client';
import { PipelineDoc, NodeDef, getPath, setPath } from '../types';

const SOURCE_TYPES = ['file_source', 'rtsp_source'];
const THRESH_KEYS = ['relation_threshold', 'conf_threshold', 'confidence_threshold', 'score_threshold'];

interface CheckItem { key: string; label: string; level: 'ok' | 'warn' | 'err' | 'idle'; detail: string; }

interface Inst {
  id: string;
  info: PipelineInfo;
  doc?: PipelineDoc;        // 从服务器 get-config 载入
  baseDoc?: PipelineDoc;    // 原始（用于推流开关还原）
  sourceUrl: string;
  rules: { type: string; params: Record<string, unknown> }[];
  push: boolean;            // 是否含 rtmp_sink
  loading: boolean;
  sc?: CheckItem[];         // 后端 /selfcheck 结果
  scBusy?: boolean;
}

// 深拷贝
const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

// 从 doc 派生实施视图字段
function derive(doc: PipelineDoc) {
  const src = doc.graph.nodes.find((n) => SOURCE_TYPES.includes(n.type));
  const alert = doc.graph.nodes.find((n) => n.type === 'alert');
  const rules = (getPath(alert?.params || {}, 'rules') as any[]) || [];
  return {
    sourceUrl: String(src?.params?.url ?? ''),
    rules: rules.map((r) => ({ type: r.type, params: r.params || {} })),
    push: doc.graph.nodes.some((n) => n.type === 'rtmp_sink'),
  };
}

// 编辑：设置源地址
function withSource(doc: PipelineDoc, url: string): PipelineDoc {
  const d = clone(doc);
  const src = d.graph.nodes.find((n) => SOURCE_TYPES.includes(n.type));
  if (src) setPath(src.params, 'url', url);
  return d;
}
// 编辑：推流开关（关=移除 rtmp_sink 及其边；开=用含 rtmp_sink 的 baseDoc）
function withPush(doc: PipelineDoc, base: PipelineDoc, on: boolean): PipelineDoc {
  if (on) {
    const has = base.graph.nodes.some((n) => n.type === 'rtmp_sink');
    return has ? clone(base) : clone(doc);
  }
  const d = clone(doc);
  const sinkIds = new Set(d.graph.nodes.filter((n) => n.type === 'rtmp_sink').map((n) => n.id));
  d.graph.nodes = d.graph.nodes.filter((n) => !sinkIds.has(n.id));
  d.graph.edges = d.graph.edges.filter((e) => !sinkIds.has(e.from) && !sinkIds.has(e.to));
  return d;
}
// 编辑：规则集合（保留原有其它参数，仅增删规则 + 改阈值）
function withRules(doc: PipelineDoc, rules: { type: string; params: Record<string, unknown> }[]): PipelineDoc {
  const d = clone(doc);
  const alert = d.graph.nodes.find((n) => n.type === 'alert');
  if (alert) setPath(alert.params, 'rules', rules);
  return d;
}
// 用新 id 克隆：替换 id + 各 URL 里出现的旧 id（推流/播放地址随实例名走）
function reId(doc: PipelineDoc, newId: string): PipelineDoc {
  const d = clone(doc); d.id = newId;
  const rep = (s: string) => s.split(doc.id).join(newId);
  for (const n of d.graph.nodes) {
    for (const k of ['url', 'output_url', 'play_url']) {
      if (typeof (n.params as any)[k] === 'string') (n.params as any)[k] = rep((n.params as any)[k]);
    }
  }
  return d;
}

export default function ImplConsole({ settings, onOpenPreview }: {
  settings: ApiSettings; onOpenPreview: (id: string) => void;
}) {
  const [insts, setInsts] = useState<Inst[]>([]);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState<Record<string, string>>({});
  const [edit, setEdit] = useState<null | { mode: 'edit' | 'new'; base: Inst; draft: PipelineDoc; name: string; url: string; push: boolean; rules: Inst['rules'] }>(null);
  const [openDiag, setOpenDiag] = useState<Record<string, boolean>>({});
  const [pick, setPick] = useState<Inst[] | null>(null);
  const timer = useRef<number | null>(null);

  const fetchSc = async (id: string) => {
    setInsts((prev) => prev.map((p) => (p.id === id ? { ...p, scBusy: true } : p)));
    const r = await api.selfcheck(settings, id);
    const sc: CheckItem[] = r.ok && Array.isArray(r.json?.checks) ? r.json.checks : [];
    setInsts((prev) => prev.map((p) => (p.id === id ? { ...p, sc, scBusy: false } : p)));
  };

  const applyList = async (probe = false) => {
    try {
      const ps = await listPipelines(settings);
      setErr('');
      setInsts((prev) => {
        const byId = new Map(prev.map((p) => [p.id, p]));
        return ps.map((info) => {
          const old = byId.get(info.id);
          return old ? { ...old, info } : { id: info.id, info, sourceUrl: '', rules: [], push: false, loading: true } as Inst;
        });
      });
      // 拉每路配置（服务器管道数量通常不大）
      for (const info of ps) {
        const c = await api.getConfig(settings, info.id);
        if (c.ok && c.json?.graph) {
          const doc: PipelineDoc = { id: info.id, graph: c.json.graph };
          const dv = derive(doc);
          setInsts((prev) => prev.map((p) => (p.id === info.id ? { ...p, doc, baseDoc: clone(doc), ...dv, loading: false } : p)));
        } else {
          setInsts((prev) => prev.map((p) => (p.id === info.id ? { ...p, loading: false } : p)));
        }
      }
      if (probe) for (const info of ps) fetchSc(info.id);   // 探测并发出（各自异步更新）
    } catch (e: any) { setErr('加载失败: ' + e.message); }
  };

  useEffect(() => { applyList(true); /* eslint-disable-next-line */ }, [settings.baseUrl, settings.token]);
  useEffect(() => { timer.current = window.setInterval(() => applyList(false), 6000); return () => { if (timer.current) clearInterval(timer.current); }; /* eslint-disable-next-line */ }, [settings.baseUrl, settings.token]);

  const setB = (id: string, v: string) => setBusy((b) => ({ ...b, [id]: v }));

  const doStart = async (id: string) => { setB(id, 'start'); const r = await api.start(settings, id); setB(id, ''); if (r.ok) applyList(true); };
  const doStop = async (id: string) => { setB(id, 'stop'); await api.stop(settings, id); setB(id, ''); applyList(true); };
  const doDel = async (id: string) => { if (!window.confirm(`删除 ${id}？`)) return; setB(id, 'del'); await api.del(settings, id); setB(id, ''); applyList(true); };
  const doBuild = async (i: Inst, start = true) => {
    if (!i.doc) return; setB(i.id, 'build');
    const r = await api.build(settings, i.doc, true);
    if (r.ok && start) { await waitState(settings, i.id, ['stopped', 'running', 'load_failed'], 30000); await api.start(settings, i.id); }
    setB(i.id, ''); applyList(true);
  };

  const openEdit = (i: Inst) => setEdit({ mode: 'edit', base: i, draft: clone(i.doc || { id: i.id, graph: { nodes: [], edges: [] } }), name: i.id, url: i.sourceUrl, push: i.push, rules: clone(i.rules) });
  const startNew = (base: Inst) => {
    let n = 2; while (insts.some((i) => i.id === base.id + '_' + n)) n++;
    setEdit({ mode: 'new', base, draft: clone(base.doc!), name: base.id + '_' + n, url: base.sourceUrl, push: base.push, rules: clone(base.rules) });
  };
  const openNew = () => {
    const bases = insts.filter((i) => i.doc);
    if (bases.length === 0) { setErr('没有可作为模板的现有管道（先在研发模式建一条并 Build）'); return; }
    if (bases.length === 1) { startNew(bases[0]); return; }
    setPick(bases);   // 多个 → 让用户选
  };

  const saveEdit = async () => {
    if (!edit) return;
    let doc = clone(edit.draft);
    doc = withSource(doc, edit.url);
    doc = withRules(doc, edit.rules);
    doc = withPush(doc, edit.base.baseDoc || doc, edit.push);
    if (edit.mode === 'new') doc = reId(doc, edit.name.trim());
    setBusy((b) => ({ ...b, [edit.name]: 'build' }));
    const r = await api.build(settings, doc, true);
    if (r.ok) { await api.saveConfig(settings, doc); await api.start(settings, doc.id); }
    setBusy((b) => { const n = { ...b }; delete n[edit.name]; return n; });
    if (r.ok) { setEdit(null); applyList(true); } else setErr('保存/重建失败: ' + JSON.stringify(r.json));
  };

  // 编辑抽屉里对规则的操作
  const toggleRule = (type: string) => {
    if (!edit) return;
    setEdit((e) => {
      if (!e) return e;
      const has = e.rules.find((r) => r.type === type);
      if (has) return { ...e, rules: e.rules.filter((r) => r.type !== type) };
      const orig = (e.base.baseDoc ? derive(e.base.baseDoc).rules : []).find((r) => r.type === type);
      return { ...e, rules: [...e.rules, orig ? clone(orig) : { type, params: { name: type } }] };
    });
  };
  const setThreshold = (type: string, val: number) => {
    setEdit((e) => {
      if (!e) return e;
      return { ...e, rules: e.rules.map((r) => { if (r.type !== type) return r; const p = { ...r.params }; const k = THRESH_KEYS.find((x) => typeof p[x] === 'number'); if (k) p[k] = val; return { ...r, params: p }; }) };
    });
  };

  const allRuleTypes = useMemo(() => {
    const s = new Set<string>();
    insts.forEach((i) => (i.baseDoc ? derive(i.baseDoc).rules : []).forEach((r) => s.add(r.type)));
    edit?.rules.forEach((r) => s.add(r.type));
    return Array.from(s);
  }, [insts, edit]);

  const RULE_LABEL: Record<string, string> = { person_intrusion: '人员入侵', person_loitering: '人员徘徊', child_near_boundary: '儿童靠近边界', fighting: '打架', climbing: '攀爬', falling: '跌倒' };

  return (
    <div className="impl">
      <div className="impl-top">
        <span className="impl-brand">监控实施台</span>
        <span className="impl-role">实施模式</span>
        <div className="spacer" />
        <span className="impl-sub">自动同步 6s</span>
        <button className="btn sm" onClick={() => applyList(true)}>🔄 服务器同步</button>
        <button className="btn sm" onClick={openNew}>＋ 新增一路</button>
      </div>
      {err && <div className="impl-err">{err}</div>}
      <div className="impl-sum">
        共 <b>{insts.length}</b> 路 ·
        运行中 <b style={{ color: 'var(--ok)' }}>{insts.filter((i) => i.info.state === 'running').length}</b> ·
        异常 <b style={{ color: 'var(--err)' }}>{insts.filter((i) => i.info.state === 'load_failed').length}</b> ·
        待部署 <b style={{ color: 'var(--muted)' }}>{insts.filter((i) => i.info.state !== 'running' && i.info.state !== 'load_failed').length}</b>
      </div>

      <div className="impl-grid">
        {insts.length === 0 && <div className="impl-empty">服务器暂无管道。可在研发模式 Build，或点「＋新增一路」。</div>}
        {insts.map((i) => {
          const sc = i.sc || [];
          const b = busy[i.id];
          return (
            <div key={i.id} className={'impl-card' + (i.info.state === 'load_failed' ? ' err' : sc.some((x) => x.level === 'err') ? ' err' : sc.some((x) => x.level === 'warn') ? ' warn' : '')}>
              <div className="impl-ch">
                <div><div className="impl-name">{i.id}</div><div className="impl-tpl">child_intrusion_fighting_fusion</div></div>
                <span className={'impl-st ' + i.info.state}><span className="impl-dot" />{b ? b + '…' : stateText(i.info.state)}</span>
              </div>
              <div className="impl-cb">
                <div className="impl-kv"><span className="impl-k">拉流</span><span className="impl-url">{i.loading ? '载入中…' : (i.sourceUrl || '—')}</span></div>
                <div className="impl-kv"><span className="impl-k">告警</span><div className="impl-chips">
                  {i.rules.length ? i.rules.map((r) => <span key={r.type} className="impl-chip on">{RULE_LABEL[r.type] || r.type}</span>) : <span className="impl-chip">—</span>}
                </div></div>
                <div className="impl-kv"><span className="impl-k">推流</span><span className={'impl-sw' + (i.push ? ' on' : '')}>{i.push ? '开' : '关'}</span></div>
                <div className="impl-kv"><span className="impl-k">自检</span>
                  <div className="impl-dots">{i.scBusy ? <span className="impl-sub">探测中…</span> : sc.map((x, k) => <span key={k} className={'impl-m ' + x.level} title={x.label + '：' + x.detail} />)}</div>
                  <span className="impl-diag" onClick={() => setOpenDiag((o) => ({ ...o, [i.id]: !o[i.id] }))}>{openDiag[i.id] ? '详情 ▴' : '详情 ▾'}</span>
                </div>
              </div>
              {openDiag[i.id] && (
                <div className="impl-diagbox">
                  {sc.length === 0 && <div className="impl-diagitem">点「重新自检」探测该行健康度</div>}
                  {sc.map((x, k) => <div key={k} className="impl-diagitem"><span className={'impl-dot ' + x.level} />{x.label}：{x.detail}</div>)}
                  <div className="impl-diagitem"><span className="impl-copyc" onClick={() => fetchSc(i.id)}>↻ 重新自检</span></div>
                </div>
              )}
              <div className="impl-cf">
                {i.info.state === 'running'
                  ? <button className="btn sm" onClick={() => doStop(i.id)}>停止</button>
                  : <button className="btn sm primary" onClick={() => i.doc ? doStart(i.id) : doBuild(i)}>
                      {i.info.state === 'stopped' ? 'Start' : 'Build'}
                    </button>}
                <button className="btn sm" onClick={() => openEdit(i)} disabled={!i.doc}>配置</button>
                {i.push && <button className="btn sm" onClick={() => onOpenPreview(i.id)}>预览</button>}
                <button className="btn sm" onClick={() => onOpenPreview(i.id)}>证据</button>
                {i.info.state !== 'running' && <button className="btn sm danger" onClick={() => doDel(i.id)}>删除</button>}
              </div>
            </div>
          );
        })}
        <div className="impl-add" onClick={openNew}><div className="impl-plus">＋</div><div>新增一路（基于现有模板）</div></div>
      </div>

      {pick && (
        <>
          <div className="impl-mask" onClick={() => setPick(null)} />
          <div className="impl-picker">
            <div className="impl-dh"><span className="impl-name">选择模板 / 基础管道</span><div className="spacer" /><button className="btn sm ghost" onClick={() => setPick(null)}>✕</button></div>
            <div className="impl-pkbody">
              {pick.map((b) => (
                <div key={b.id} className="impl-pkitem" onClick={() => { startNew(b); setPick(null); }}>
                  <div className="impl-pkname">{b.id}</div>
                  <div className="impl-pkmeta">
                    <span>源：{b.sourceUrl || '—'}</span>
                    <span>告警 {b.rules.length} 项</span>
                    <span>{b.push ? '推流开' : '推流关'}</span>
                  </div>
                  <div className="impl-pkchips">{b.rules.map((r) => <span key={r.type} className="impl-chip on">{RULE_LABEL[r.type] || r.type}</span>)}</div>
                </div>
              ))}
            </div>
            <div className="impl-hint" style={{ padding: '0 16px 12px' }}>选择一路作为模板克隆，再改拉流地址/告警/推流即可新增。</div>
          </div>
        </>
      )}

      {edit && (
        <>
          <div className="impl-mask" onClick={() => setEdit(null)} />
          <div className="impl-drawer">
            <div className="impl-dh">
              <span className="impl-name">{edit.mode === 'new' ? '新增一路' : '配置 · ' + edit.name}</span>
              <div className="spacer" /><button className="btn sm ghost" onClick={() => setEdit(null)}>✕</button>
            </div>
            <div className="impl-db">
              {edit.mode === 'new' && (
                <div className="impl-field"><label>实例名称(管道 id)</label>
                  <input type="text" value={edit.name} onChange={(e) => setEdit({ ...edit, name: e.target.value })} />
                  <div className="impl-hint">将基于「{edit.base.id}」克隆，推流/播放地址里的实例名自动替换</div></div>
              )}
              <div className="impl-field"><label>拉流地址</label>
                <input type="text" value={edit.url} onChange={(e) => setEdit({ ...edit, url: e.target.value })} />
                <div className="impl-hint">支持 rtsp:// / rtmp:// / file:///</div></div>

              <div className="impl-field"><label>告警项（模板开放项）</label>
                {allRuleTypes.map((t) => {
                  const on = !!edit.rules.find((r) => r.type === t);
                  const rule = edit.rules.find((r) => r.type === t);
                  const tk = rule ? THRESH_KEYS.find((x) => typeof rule.params[x] === 'number') : undefined;
                  return (
                    <div key={t} className={'impl-rule' + (on ? '' : ' locked')}>
                      <span className="impl-rn">{RULE_LABEL[t] || t}<div className="impl-rt">{t}{on ? '' : '（关）'}</div></span>
                      {on && tk && <input type="range" min={0} max={1} step={0.05} value={Number(rule!.params[tk]) || 0} onChange={(e) => setThreshold(t, Number(e.target.value))} />}
                      <span className={'impl-chip' + (on ? ' on' : '')} style={{ cursor: 'pointer' }} onClick={() => toggleRule(t)}>{on ? '开' : '关'}</span>
                    </div>
                  );
                })}
                <div className="impl-hint">灵敏度=阈值滑条；规则细节/区域/类别由模板锁定，如需增改请回研发模式。</div>
              </div>

              <div className="impl-field"><label>是否推流</label>
                <span className={'impl-sw big' + (edit.push ? ' on' : '')} onClick={() => setEdit({ ...edit, push: !edit.push })}>{edit.push ? '开（连接 RTMP，可实时预览）' : '关'}</span>
                <div className="impl-hint">关闭则移除 RTMP 节点、不生成播放地址；仅保留证据。</div></div>

              <div className="impl-locknote">🔒 节点拓扑/模型/融合/上报由模板锁定，实施页不可修改。</div>
            </div>
            <div className="impl-df">
              <button className="btn" onClick={() => setEdit(null)}>取消</button>
              <div className="spacer" />
              <button className="btn primary" onClick={saveEdit} disabled={!edit.name.trim()}>保存并重建</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function stateText(s: string) {
  return s === 'running' ? '运行中' : s === 'load_failed' ? '异常' : s === 'stopped' ? '已停止' : s === 'loading' ? '载入中' : '待部署';
}
