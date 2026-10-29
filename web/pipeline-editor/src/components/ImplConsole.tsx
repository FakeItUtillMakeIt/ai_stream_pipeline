import { useEffect, useRef, useState } from 'react';
import { ApiSettings, api, listPipelines, PipelineInfo, waitState } from '../api/client';
import { PipelineDoc, getPath, setPath, Overrides, TemplateDef, detectOverrides } from '../types';

const SOURCE_TYPES = ['file_source', 'rtsp_source'];
const THRESH_KEYS = ['relation_threshold', 'conf_threshold', 'confidence_threshold', 'score_threshold'];

interface CheckItem { key: string; label: string; level: 'ok' | 'warn' | 'err' | 'idle'; detail: string; }

interface Inst {
  id: string;
  info: PipelineInfo;
  doc?: PipelineDoc;
  baseDoc?: PipelineDoc;
  sourceUrl: string;
  rules: { type: string; params: Record<string, unknown> }[];
  push: boolean;
  loading: boolean;
  sc?: CheckItem[];
  scBusy?: boolean;
}

interface EditState {
  mode: 'edit' | 'new';
  draft: PipelineDoc;        // 工作图（new=模板doc；edit=已载入管道doc）
  origId: string;            // draft 原 id（new 时用于替换 url 里的实例名）
  ov: Overrides;             // 可改字段契约
  name: string;
  url: string;
  push: boolean;
  rules: { type: string; params: Record<string, unknown> }[];
  allRuleTypes: string[];    // 允许勾选的规则类型
}

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

// 从 doc 派生实施视图字段
function derive(doc: PipelineDoc) {
  const src = doc.graph.nodes.find((n) => SOURCE_TYPES.includes(n.type));
  const alert = doc.graph.nodes.find((n) => n.type === 'alert');
  const rules = (getPath(alert?.params || {}, 'rules') as any[]) || [];
  return {
    sourceUrl: String(src?.params?.url ?? ''),
    rules: rules.map((r) => ({ type: String(r.type), params: r.params || {} })),
    push: doc.graph.nodes.some((n) => n.type === 'rtmp_sink'),
  };
}

// 按 overrides 把编辑值注入回 doc
function applyOverrides(base: PipelineDoc, ov: Overrides, e: { url: string; push: boolean; rules: { type: string; params: Record<string, unknown> }[] }): PipelineDoc {
  const d = clone(base);
  const find = (id: string) => d.graph.nodes.find((n) => n.id === id);
  if (ov.source_url) { const n = find(ov.source_url.node); if (n) setPath(n.params, ov.source_url.path, e.url); }
  if (ov.alerts) { const n = find(ov.alerts.node); if (n) setPath(n.params, 'rules', e.rules); }
  if (ov.push && !e.push) {
    const sink = find(ov.push.node);
    if (sink) {
      const rm = sink.id;
      d.graph.nodes = d.graph.nodes.filter((x) => x.id !== rm);
      d.graph.edges = d.graph.edges.filter((x) => x.from !== rm && x.to !== rm);
    }
  }
  return d;
}

// 用新 id 克隆：替换 id + 各 URL 里出现的旧 id
function reId(doc: PipelineDoc, newId: string, oldId: string): PipelineDoc {
  const d = clone(doc); d.id = newId;
  if (!oldId) return d;
  const rep = (s: string) => s.split(oldId).join(newId);
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
  const [edit, setEdit] = useState<null | EditState>(null);
  const [openDiag, setOpenDiag] = useState<Record<string, boolean>>({});
  const [tpls, setTpls] = useState<{ name: string; version: string; description: string }[]>([]);
  const [pickTpl, setPickTpl] = useState<{ name: string; version: string; description: string }[] | null>(null);
  const timer = useRef<number | null>(null);

  const loadTpls = async (): Promise<{ name: string; version: string; description: string }[]> => {
    const r = await api.templates(settings);
    const t = r.ok && Array.isArray(r.json?.templates) ? r.json.templates : [];
    setTpls(t); return t;
  };
  useEffect(() => { loadTpls(); /* eslint-disable-next-line */ }, [settings.baseUrl, settings.token]);

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

  const openEdit = (i: Inst) => {
    const doc = i.doc || { id: i.id, graph: { nodes: [], edges: [] } };
    const ov = detectOverrides(doc);
    const allRuleTypes = ov.alerts?.types || i.rules.map((r) => r.type);
    setEdit({ mode: 'edit', draft: clone(doc), origId: i.id, ov, name: i.id, url: i.sourceUrl, push: i.push, rules: clone(i.rules), allRuleTypes });
  };
  const startNewFromTemplate = (tpl: TemplateDef) => {
    const doc = tpl.doc; const dv = derive(doc); const ov = tpl.overrides || {};
    const allRuleTypes = ov.alerts?.types || dv.rules.map((r) => r.type);
    let name = tpl.name; let n = 1; while (insts.some((i) => i.id === name)) { n++; name = tpl.name + '_' + n; }
    setEdit({ mode: 'new', draft: clone(doc), origId: doc.id, ov, name, url: dv.sourceUrl, push: dv.push, rules: clone(dv.rules), allRuleTypes });
  };
  const pickTemplate = async (name: string) => {
    const r = await api.templateGet(settings, name);
    if (!r.ok || !r.json?.doc?.graph) { setErr('读取模板失败: ' + name); return; }
    startNewFromTemplate(r.json as TemplateDef);
  };
  const openNew = async () => {
    const t = await loadTpls();
    if (t.length === 0) { setErr('模板库为空，请先在研发模式「发布为模板」'); return; }
    if (t.length === 1) { pickTemplate(t[0].name); return; }
    setPickTpl(t);
  };

  const saveEdit = async () => {
    if (!edit) return;
    let doc = applyOverrides(edit.draft, edit.ov, edit);
    if (edit.mode === 'new') doc = reId(doc, edit.name.trim(), edit.origId);
    const r = await api.build(settings, doc, edit.mode === 'edit');
    if (r.ok) { await api.saveConfig(settings, doc); await api.start(settings, doc.id); setEdit(null); applyList(true); }
    else setErr('保存/重建失败: ' + JSON.stringify(r.json));
  };

  const toggleRule = (type: string) => {
    setEdit((e) => {
      if (!e) return e;
      if (e.rules.find((r) => r.type === type)) return { ...e, rules: e.rules.filter((r) => r.type !== type) };
      const orig = derive(e.draft).rules.find((r) => r.type === type);
      return { ...e, rules: [...e.rules, orig ? clone(orig) : { type, params: { name: type } }] };
    });
  };
  const setThreshold = (type: string, val: number) => {
    setEdit((e) => {
      if (!e) return e;
      return { ...e, rules: e.rules.map((r) => { if (r.type !== type) return r; const p = { ...r.params }; const k = THRESH_KEYS.find((x) => typeof p[x] === 'number'); if (k) p[k] = val; return { ...r, params: p }; }) };
    });
  };

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

      {pickTpl && (
        <>
          <div className="impl-mask" onClick={() => setPickTpl(null)} />
          <div className="impl-picker">
            <div className="impl-dh"><span className="impl-name">选择模板</span><div className="spacer" /><button className="btn sm ghost" onClick={() => setPickTpl(null)}>✕</button></div>
            <div className="impl-pkbody">
              {pickTpl.map((t) => (
                <div key={t.name} className="impl-pkitem" onClick={() => { pickTemplate(t.name); setPickTpl(null); }}>
                  <div className="impl-pkname">{t.name} <span className="impl-pkver">v{t.version || '1'}</span></div>
                  {t.description && <div className="impl-pkmeta"><span>{t.description}</span></div>}
                </div>
              ))}
            </div>
            <div className="impl-hint" style={{ padding: '0 16px 12px' }}>选择模板实例化一路，再填拉流地址/告警/推流即可新增。</div>
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
                  <div className="impl-hint">基于模板「{edit.origId}」实例化，推流/播放地址里的实例名自动替换</div></div>
              )}
              {edit.ov.source_url && (
                <div className="impl-field"><label>{edit.ov.source_url.label || '拉流地址'}</label>
                  <input type="text" value={edit.url} onChange={(e) => setEdit({ ...edit, url: e.target.value })} />
                  <div className="impl-hint">支持 rtsp:// / rtmp:// / file:///</div></div>
              )}

              {edit.ov.alerts && (
                <div className="impl-field"><label>{edit.ov.alerts.label || '告警项'}</label>
                  {edit.allRuleTypes.map((t) => {
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
              )}

              {edit.ov.push && (
                <div className="impl-field"><label>{edit.ov.push.label || '是否推流'}</label>
                  <span className={'impl-sw big' + (edit.push ? ' on' : '')} onClick={() => setEdit({ ...edit, push: !edit.push })}>{edit.push ? '开（连接 RTMP，可实时预览）' : '关'}</span>
                  <div className="impl-hint">关闭则移除 RTMP 节点、不生成播放地址；仅保留证据。</div></div>
              )}

              <div className="impl-locknote">🔒 未在本模板「可改字段」中开放的参数（节点拓扑/模型/融合/上报）均锁定，实施页不可修改。</div>
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
