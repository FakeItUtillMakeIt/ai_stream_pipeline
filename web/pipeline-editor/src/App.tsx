import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow, Background, Controls, addEdge,
  useNodesState, useEdgesState, Connection, Node, Edge,
  MarkerType, ReactFlowProvider, useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import Palette from './components/Palette';
import PropertyPanel from './components/PropertyPanel';
import PipelineNode from './components/PipelineNode';
import PreviewView from './components/PreviewView';
import ImplConsole from './components/ImplConsole';
import AlertsView from './components/AlertsView';
import LoginView from './components/LoginView';
import { defaultParams } from './schema/nodeSchema';
import { toFlow, fromFlow } from './graph/flow';
import { validate, hasErrors, Issue } from './graph/validation';
import { PipelineDoc, NodeDef, detectOverrides, Overrides, TemplateDef } from './types';
import { api, ApiSettings, waitState, deriveState } from './api/client';

const nodeTypes = { pipelineNode: PipelineNode };

type LogLevel = 'info' | 'warn' | 'error';
type Status = 'waiting' | 'running' | 'error';
interface LogEntry { ts: string; level: LogLevel; msg: string; }
interface SrvInfo { running: boolean; state: string; }

const sig = (d: unknown) => JSON.stringify(d);

// URL(hash) 入口：#/impl=实施台，#/preview=预览，#/alerts=告警记录，其余=研发编辑器
type ViewName = 'editor' | 'preview' | 'impl' | 'alerts';
function viewFromHash(): ViewName {
  const h = (location.hash || '').toLowerCase();
  if (h.includes('impl')) return 'impl';
  if (h.includes('preview')) return 'preview';
  if (h.includes('alert')) return 'alerts';
  return 'editor';
}

// 深替换：把所有等于 oldId 的字符串值/对象键改成 newId（节点改名后修正引用）
function replaceRefs(v: any, oldId: string, newId: string): any {
  if (Array.isArray(v)) return v.map((x) => replaceRefs(x, oldId, newId));
  if (v && typeof v === 'object') {
    const o: any = {};
    for (const [k, val] of Object.entries(v)) o[k === oldId ? newId : k] = replaceRefs(val, oldId, newId);
    return o;
  }
  return v === oldId ? newId : v;
}

function load<T>(key: string, fallback: T): T {
  try { const s = localStorage.getItem(key); return s ? JSON.parse(s) : fallback; } catch { return fallback; }
}

function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [paramsById, setParamsById] = useState<Record<string, NodeDef['params']>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);

  // 多管道 store（持久化到 localStorage）
  const [store, setStore] = useState<Record<string, PipelineDoc>>(() => load('editorStore', {}));
  const [savedSig, setSavedSig] = useState<Record<string, string>>(() => load('editorSavedSig', {}));
  const [activeId, setActiveId] = useState('');
  const [idDraft, setIdDraft] = useState('');
  const [serverState, setServerState] = useState<Record<string, SrvInfo>>({});

  const [log, setLog] = useState<LogEntry[]>([]);
  const [levels, setLevels] = useState<Record<LogLevel, boolean>>({ info: true, warn: true, error: true });
  const [autoScroll, setAutoScroll] = useState(true);
  const [logOpen, setLogOpen] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const [theme, setTheme] = useState<'light' | 'dark'>(() => load('editorTheme', 'dark'));
  const [status, setStatus] = useState<Status>('waiting');
  const [view, setView] = useState<ViewName>(viewFromHash);
  const [previewId, setPreviewId] = useState('');
  const [previewFrom, setPreviewFrom] = useState<'editor' | 'impl'>('editor');
  const [role, setRole] = useState<'dev' | 'impl' | 'anon' | 'unknown'>('unknown');
  const [authEnabled, setAuthEnabled] = useState(false);
  const [checked, setChecked] = useState(false);
  const [pub, setPub] = useState<null | { name: string; version: string; description: string; ov: Overrides; expose: { source: boolean; push: boolean; alerts: boolean } }>(null);
  const [overwrite, setOverwrite] = useState<boolean>(() => load('editorOverwrite', true));

  const [settings, setSettings] = useState<ApiSettings>(() => load('apiSettings', { baseUrl: '', token: '' }));
  const fileRef = useRef<HTMLInputElement>(null);
  const counter = useRef(1);
  const inited = useRef(false);
  const rf = useReactFlow();

  useEffect(() => { localStorage.setItem('apiSettings', JSON.stringify(settings)); }, [settings]);
  useEffect(() => { localStorage.setItem('editorTheme', JSON.stringify(theme)); }, [theme]);
  useEffect(() => { localStorage.setItem('editorStore', JSON.stringify(store)); }, [store]);
  useEffect(() => { localStorage.setItem('editorSavedSig', JSON.stringify(savedSig)); }, [savedSig]);
  useEffect(() => { localStorage.setItem('editorOverwrite', JSON.stringify(overwrite)); }, [overwrite]);
  // view ↔ URL 双向同步：#/editor #/impl #/preview 可直达/收藏/前进后退
  useEffect(() => { const want = '#/' + view; if (location.hash !== want) location.hash = want; }, [view]);
  useEffect(() => { const onHash = () => setView(viewFromHash()); window.addEventListener('hashchange', onHash); return () => window.removeEventListener('hashchange', onHash); }, []);
  // 角色探测：非 dev 锁在实施台（后端对 dev-only 接口另有 403 兜底）
  useEffect(() => {
    let alive = true;
    api.whoami(settings).then((r) => {
      if (!alive) return;
      setRole(r.ok && r.json?.role ? r.json.role : 'anon');
      setAuthEnabled(!!(r.ok && r.json?.auth_enabled));
      setChecked(true);
    }).catch(() => { if (alive) { setRole('dev'); setAuthEnabled(false); setChecked(true); } });   // 服务不可达：不锁（离线开发）
    return () => { alive = false; };
  }, [settings.baseUrl, settings.token]);
  useEffect(() => { if ((role === 'impl' || role === 'anon') && view === 'editor') setView('impl'); }, [role, view]);
  useEffect(() => { setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, status } }))); }, [status, setNodes]);
  useEffect(() => { if (autoScroll && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log, autoScroll, logOpen]);

  const appendLog = (msg: string, level: LogLevel = 'info') =>
    setLog((l) => [...l, { ts: new Date().toLocaleTimeString(), level, msg }].slice(-500));

  const genId = (type: string) => {
    let id: string;
    do { id = `${type}_${counter.current++}`; } while (nodes.some((n) => n.id === id));
    return id;
  };
  const genPipelineId = () => { let n = 1; while (store[`pipeline_${n}`]) n++; return `pipeline_${n}`; };

  // ===== 当前编辑态 ↔ store =====
  const buildDoc = (id: string): PipelineDoc => fromFlow(id, nodes, edges, paramsById);
  const currentDoc = (): PipelineDoc => buildDoc(activeId);

  const loadDoc = (doc: PipelineDoc) => {
    const f = toFlow(doc);
    const info = serverState[doc.id];
    const st: Status = info?.running ? 'running' : info?.state === 'load_failed' ? 'error' : 'waiting';
    setNodes(f.nodes.map((n) => ({ ...n, data: { ...n.data, status: st } })));
    setEdges(f.edges.map((e) => ({ ...e, markerEnd: { type: MarkerType.ArrowClosed } })));
    const p: Record<string, NodeDef['params']> = {};
    doc.graph.nodes.forEach((n) => { p[n.id] = n.params || {}; });
    setParamsById(p);
    setActiveId(doc.id);
    setIdDraft(doc.id);
    setSelectedId(null);
    setIssues([]);
    setStatus(st);
    counter.current = Math.max(counter.current, doc.graph.nodes.length + 1);
  };

  const commitActive = () => { if (!activeId) return; const doc = buildDoc(activeId); setStore((s) => ({ ...s, [activeId]: doc })); };

  const switchTo = (id: string) => { if (id === activeId) return; commitActive(); const doc = store[id]; if (doc) loadDoc(doc); };

  const newPipeline = () => {
    commitActive();
    const key = genPipelineId();
    const doc: PipelineDoc = { id: key, graph: { nodes: [], edges: [] } };
    setStore((s) => ({ ...s, [key]: doc }));
    loadDoc(doc);
    appendLog(`新建管道 ${key}`);
  };

  const renameActive = (raw: string) => {
    const next = raw.trim();
    if (!activeId) {
      if (!next) return;
      if (store[next]) { appendLog(`管道 ${next} 已存在`, 'warn'); setIdDraft(''); return; }
      setStore((s) => ({ ...s, [next]: buildDoc(next) })); setActiveId(next); return;
    }
    if (!next || next === activeId) { setIdDraft(activeId); return; }
    if (store[next]) { appendLog(`管道 ${next} 已存在，未改名`, 'warn'); setIdDraft(activeId); return; }
    const doc = buildDoc(next);
    setStore((s) => { const n = { ...s }; delete n[activeId]; n[next] = doc; return n; });
    if (savedSig[activeId]) setSavedSig((s) => { const n = { ...s }; n[next] = n[activeId]; delete n[activeId]; return n; });
    setActiveId(next);
    appendLog(`管道重命名 ${activeId} → ${next}`);
  };

  // 节点改名：更新节点/边/params，并深替换其它节点参数里对该 id 的引用
  const renameNode = (oldId: string, rawNew: string) => {
    const newId = rawNew.trim();
    if (!newId || newId === oldId) return;
    if (nodes.some((n) => n.id === newId)) { appendLog(`节点 id ${newId} 已存在`, 'warn'); return; }
    setNodes((ns) => ns.map((n) => (n.id === oldId ? { ...n, id: newId, data: { ...n.data, label: newId } } : n)));
    setEdges((es) => es.map((e) => {
      const source = e.source === oldId ? newId : e.source;
      const target = e.target === oldId ? newId : e.target;
      return { ...e, source, target, id: `${source}->${target}` };
    }));
    setParamsById((p) => {
      const next: Record<string, NodeDef['params']> = {};
      for (const [k, v] of Object.entries(p)) {
        const nk = k === oldId ? newId : k;
        next[nk] = replaceRefs(v, oldId, newId);
      }
      return next;
    });
    setSelectedId((s) => (s === oldId ? newId : s));
    appendLog(`节点改名 ${oldId} → ${newId}`);
  };

  // ===== 画布编辑 =====
  const addNodeAt = (type: string, position: { x: number; y: number }) => {
    if (!activeId) { const id = genPipelineId(); setStore((s) => ({ ...s, [id]: buildDoc(id) })); setActiveId(id); setIdDraft(id); }
    const id = genId(type);
    const n: Node = { id, type: 'pipelineNode', position, data: { label: id, nodeType: type, status } };
    setNodes((ns) => ns.concat(n));
    setParamsById((p) => ({ ...p, [id]: defaultParams(type) }));
    setSelectedId(id);
  };
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('application/reactflow');
    if (!type) return;
    const position = rf.screenToFlowPosition({ x: e.clientX, y: e.clientY });
    addNodeAt(type, { x: Math.round(position.x), y: Math.round(position.y) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rf, status, nodes.length, activeId]);
  const onDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }, []);
  const onConnect = useCallback((c: Connection) => {
    if (!c.source || !c.target || c.source === c.target) return;
    setEdges((es) => addEdge({ ...c, id: `${c.source}->${c.target}`, markerEnd: { type: MarkerType.ArrowClosed }, style: { strokeWidth: 2 } }, es));
  }, [setEdges]);
  const onNodesDelete = useCallback((del: Node[]) => {
    const ids = new Set(del.map((d) => d.id));
    setParamsById((p) => { const n = { ...p }; ids.forEach((i) => delete n[i]); return n; });
    setEdges((es) => es.filter((e) => !ids.has(e.source) && !ids.has(e.target)));
    setSelectedId((s) => (s && ids.has(s) ? null : s));
  }, [setEdges]);

  const doValidate = () => {
    const iss = validate(currentDoc()); setIssues(iss);
    const e = iss.filter((i) => i.level === 'error').length, w = iss.filter((i) => i.level === 'warn').length;
    appendLog(`校验: ${e} 错误 / ${w} 警告`, e ? 'error' : w ? 'warn' : 'info');
    return iss;
  };
  const exportJson = () => {
    const doc = currentDoc();
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `${activeId || 'pipeline'}.json`; a.click();
    URL.revokeObjectURL(a.href); appendLog('已导出 JSON');
  };
  const importJson = (file: File) => {
    file.text().then((t) => {
      try {
        const doc: PipelineDoc = JSON.parse(t);
        if (!doc.id) doc.id = file.name.replace(/\.json$/i, '');
        if (store[doc.id] && !window.confirm(`管道 ${doc.id} 已存在，覆盖？`)) return;
        commitActive();
        setStore((s) => ({ ...s, [doc.id]: doc }));
        loadDoc(doc);
        appendLog(`导入 ${doc.id}：${doc.graph.nodes.length} 节点 / ${doc.graph.edges.length} 边`);
      } catch (e: any) { appendLog('导入失败: ' + e.message, 'error'); }
    });
  };

  // ===== 服务器交互 =====
  const guard = (): boolean => { if (!activeId) { appendLog('未选择管道', 'error'); return false; } const iss = validate(currentDoc()); setIssues(iss); return !hasErrors(iss); };
  const markSaved = (id: string) => { const doc = id === activeId ? currentDoc() : store[id]; if (doc) setSavedSig((s) => ({ ...s, [id]: sig(doc) })); };

  const refreshStatus = async () => {
    const r = await api.list(settings);
    if (r.ok && r.json?.pipelines) {
      const map: Record<string, SrvInfo> = {};
      for (const it of r.json.pipelines) {
        const id = it.id; if (!id) continue;
        const state = deriveState(it) || 'stopped';
        map[id] = { running: it.running === true || state === 'running', state };
      }
      setServerState(map); appendLog('已刷新服务器管道状态');
    } else appendLog('List → ' + JSON.stringify(r.json), 'error');
  };

  // 同步服务器：刷新运行态 + 把"服务器有、本地无"的管道配置导入编辑器
  const syncFromServer = async (silent = false) => {
    const r = await api.list(settings);
    if (!r.ok || !r.json?.pipelines) { if (!silent) appendLog('同步失败: ' + JSON.stringify(r.json), 'error'); return; }
    const arr = r.json.pipelines as any[];
    const map: Record<string, SrvInfo> = {};
    for (const it of arr) {
      const id = it.id; if (!id) continue;
      const state = deriveState(it) || 'stopped';
      map[id] = { running: it.running === true || state === 'running', state };
    }
    setServerState(map);
    let imported = 0;
    for (const it of arr) {
      const id = it.id; if (!id || store[id]) continue;
      const c = await api.getConfig(settings, id);
      if (c.ok && c.json?.graph) { setStore((s) => (s[id] ? s : { ...s, [id]: { id, graph: c.json.graph } })); imported++; }
    }
    if (!silent) appendLog(`同步服务器：${arr.length} 个管道${imported ? `，导入 ${imported} 个新管道` : ''}`);
  };

  // 定时静默同步：外部(Apipost/平台)的 build/start/stop/delete 自动反映到编辑器
  const syncRef = useRef<() => void>(() => {});
  syncRef.current = () => { void syncFromServer(true); };
  useEffect(() => { const t = setInterval(() => syncRef.current(), 6000); return () => clearInterval(t); }, []);

  const doBuild = async () => {
    if (!guard()) return;
    commitActive(); setStatus('waiting');
    let r = await api.build(settings, currentDoc(), overwrite);
    if (r.status === 409) {
      appendLog('管道已存在（未开覆盖重建）。勾选"覆盖重建"再 Build，或先 Delete。', 'warn');
      setStatus('error'); return;
    }
    appendLog(`Build ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (!r.ok) { setStatus('error'); return; }
    const final = await waitState(settings, activeId, ['stopped', 'running', 'load_failed'], 30000);
    const st = deriveState(final);
    if (st === 'load_failed') { setStatus('error'); appendLog('加载失败', 'error'); }
    else { markSaved(activeId); appendLog('Build 完成（已保存配置）'); }
    refreshStatus();
  };
  const doStart = async () => {
    if (!activeId) return;
    const r = await api.start(settings, activeId);
    appendLog(`Start ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    setStatus(r.ok ? 'running' : 'error'); refreshStatus();
    if (r.ok) { setPreviewFrom('editor'); setView('preview'); }
  };
  const doStop = async () => {
    if (!activeId) return;
    const r = await api.stop(settings, activeId);
    appendLog(`Stop ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (r.ok) setStatus('waiting'); refreshStatus();
  };
  const doDelete = async () => {
    if (!activeId) return;
    if (!window.confirm(`删除管道 ${activeId}？`)) return;
    const r = await api.del(settings, activeId);
    appendLog(`Delete ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (r.ok) {
      setStatus('waiting');
      const gone = activeId;
      setStore((s) => { const n = { ...s }; delete n[gone]; return n; });
      setSavedSig((s) => { const n = { ...s }; delete n[gone]; return n; });
      const rest = Object.keys(store).filter((k) => k !== gone);
      if (rest[0]) loadDoc(store[rest[0]]); else { setActiveId(''); setIdDraft(''); setNodes([]); setEdges([]); setParamsById({}); }
      refreshStatus();
    }
  };
  const doSave = async () => {
    if (!guard()) return;
    commitActive();
    const r = await api.saveConfig(settings, currentDoc());
    appendLog(`保存到服务器 ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (r.ok) markSaved(activeId);
  };

  const openPublish = () => {
    if (!activeId) { appendLog('未选择管道，无法发布模板', 'error'); return; }
    const doc = currentDoc();
    const ov = detectOverrides(doc);
    if (!ov.source_url && !ov.push && !ov.alerts) { appendLog('该管道未检测到可暴露字段（源/推流/告警），仍可发布但实施无法改配置', 'warn'); }
    setPub({ name: activeId, version: '1', description: '', ov, expose: { source: !!ov.source_url, push: !!ov.push, alerts: !!ov.alerts } });
  };
  const doPublish = async () => {
    if (!pub) return;
    const name = pub.name.trim();
    if (!name) { appendLog('模板名不能为空', 'error'); return; }
    const ov: Overrides = {};
    if (pub.expose.source && pub.ov.source_url) ov.source_url = pub.ov.source_url;
    if (pub.expose.push && pub.ov.push) ov.push = pub.ov.push;
    if (pub.expose.alerts && pub.ov.alerts) ov.alerts = pub.ov.alerts;
    const tpl: TemplateDef = { name, version: pub.version.trim() || '1', description: pub.description, doc: currentDoc(), overrides: ov };
    const r = await api.templateSave(settings, tpl);
    appendLog(`发布模板 ${name} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (r.ok) setPub(null);
  };

  const selectedNode: NodeDef | null = useMemo(() => {
    if (!selectedId) return null;
    const n = nodes.find((x) => x.id === selectedId);
    if (!n) return null;
    return { id: n.id, type: (n.data as any).nodeType, params: paramsById[n.id] || {}, position: n.position };
  }, [selectedId, nodes, paramsById]);

  const onParamChange = (nd: NodeDef) => {
    setParamsById((p) => ({ ...p, [nd.id]: nd.params }));
    setNodes((ns) => ns.map((n) => (n.id === nd.id ? { ...n, position: nd.position || n.position } : n)));
  };
  const onDeleteNode = (id: string) => {
    setNodes((ns) => ns.filter((n) => n.id !== id));
    setEdges((es) => es.filter((e) => e.source !== id && e.target !== id));
    setParamsById((p) => { const n = { ...p }; delete n[id]; return n; });
    setSelectedId(null);
  };

  const pipelineIds = Object.keys(store);
  const isDirty = (id: string) => { const doc = id === activeId ? currentDoc() : store[id]; return doc ? sig(doc) !== (savedSig[id] || '') : false; };
  const anyDirty = pipelineIds.some(isDirty) || (!!activeId && isDirty(activeId));

  // 关页/刷新前，若有未保存到服务器的改动则提示
  useEffect(() => {
    const h = (e: BeforeUnloadEvent) => { if (anyDirty) { e.preventDefault(); e.returnValue = ''; } };
    window.addEventListener('beforeunload', h);
    return () => window.removeEventListener('beforeunload', h);
  }, [anyDirty]);

  // 首次挂载：从 localStorage 恢复的 store 里载入第一条
  useEffect(() => {
    if (inited.current) return; inited.current = true;
    const ids = Object.keys(store);
    if (ids.length) loadDoc(store[ids[0]]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const shownLog = log.filter((l) => levels[l.level]);

  // 登录 gate：未校验完不渲染任何业务界面；服务启用鉴权且未登录 → 登录页
  if (!checked) return <div className={`app theme-${theme}`}><div className="login-splash">正在校验登录状态…</div></div>;
  if (authEnabled && role !== 'dev' && role !== 'impl')
    return <div className={`app theme-${theme}`}><LoginView settings={settings} setSettings={setSettings} /></div>;

  if (view === 'impl') {
    return (
      <div className={`app theme-${theme}`}>
        <ImplConsole settings={settings} setSettings={setSettings} role={role}
          onOpenPreview={(id) => { setPreviewId(id); setPreviewFrom('impl'); setView('preview'); }}
          onOpenAlerts={() => { setPreviewFrom('impl'); setView('alerts'); }}
          onBackToDev={() => { setView('editor'); refreshStatus(); }} />
      </div>
    );
  }

  if (view === 'alerts') {
    return (
      <div className={`app theme-${theme}`}>
        <AlertsView settings={settings} initialPipeline={previewId}
          backLabel={previewFrom === 'impl' ? '← 返回实施' : '← 返回编辑'}
          onBack={() => { setView(previewFrom); refreshStatus(); }} />
      </div>
    );
  }

  if (view === 'preview') {
    return (
      <div className={`app theme-${theme}`}>
        <PreviewView settings={settings} initialId={previewId || activeId}
          backLabel={previewFrom === 'impl' ? '← 返回实施' : '← 返回编辑'}
          onBack={() => { setView(previewFrom); setPreviewId(''); refreshStatus(); }} />
      </div>
    );
  }

  return (
    <div className={`app theme-${theme}`}>
      <div className="topbar">
        <div className="brand"><span className="logo" />Pipeline Editor</div>
        <div className="sep" />
        <select className="pid" value={activeId} onChange={(e) => switchTo(e.target.value)} title="切换已加载的管道">
          <option value="">（选择管道）</option>
          {pipelineIds.map((id) => (
            <option key={id} value={id}>{`${serverState[id]?.running ? '●' : ''}${isDirty(id) ? '✎' : ''} ${id}`}</option>
          ))}
        </select>
        <input className="pid" value={idDraft} placeholder="管道 id（改名）"
          onChange={(e) => setIdDraft(e.target.value)} onBlur={(e) => renameActive(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }} title="当前管道 id，可改名" />
        <button className="btn" onClick={newPipeline} title="新建管道">＋新建</button>
        <span className={'pill ' + (anyDirty ? 'dirty' : 'saved')} title={anyDirty ? '有改动未保存到服务器' : '已全部保存'}>{anyDirty ? '✎ 未保存' : '✓ 已保存'}</span>

        <div className="sep" />
        <div className="group">
          <span className="glabel">设计</span>
          <button className="btn" onClick={() => fileRef.current?.click()}><span className="ic">⭳</span>导入</button>
          <input ref={fileRef} type="file" accept=".json" hidden onChange={(e) => e.target.files?.[0] && importJson(e.target.files[0])} />
          <button className="btn" onClick={exportJson}><span className="ic">⭱</span>导出</button>
          <button className="btn" onClick={doValidate}><span className="ic">✓</span>校验</button>
          <button className="btn" onClick={doSave}><span className="ic">⤓</span>保存到服务器</button>
          <button className="btn ok" onClick={openPublish} title="把当前管道发布为模板，供实施台选用"><span className="ic">◆</span>发布为模板</button>
        </div>

        <div className="sep" />
        <div className="group">
          <span className="glabel">调试</span>
          <button className="btn primary" onClick={doBuild}><span className="ic">🛠</span>Build</button>
          <button className="btn ok" onClick={doStart}><span className="ic">▶</span>Start</button>
          <button className="btn warn" onClick={doStop}><span className="ic">⏸</span>Stop</button>
          <button className="btn danger" onClick={doDelete}><span className="ic">🗑</span>Delete</button>
        </div>

        <div className="spacer" />
        <label className={'switch' + (overwrite ? ' on' : '')} onClick={() => setOverwrite(!overwrite)} title="已存在同 id 时先停止销毁再重建"><span className="track" />覆盖重建</label>
        <button className="btn ghost icon" onClick={() => syncFromServer()} title="同步服务器（拉运行态并导入外部新建管道）">⟳</button>
        <div className="seg">
          <button className="on" onClick={() => setView('editor')}>编辑器</button>
          <button onClick={() => { commitActive(); setView('impl'); }}>实施台</button>
          <button onClick={() => { commitActive(); setPreviewFrom('editor'); setView('preview'); }}>预览</button>
        </div>
        <button className="btn ghost icon" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} title="主题">{theme === 'dark' ? '☀' : '🌙'}</button>
        {authEnabled && <button className="btn ghost icon" onClick={() => setSettings({ ...settings, token: '' })} title="退出登录">⏻</button>}
      </div>

      <div className="main">
        <Palette />
        <div className="canvas" onDrop={onDrop} onDragOver={onDragOver}>
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} onNodesDelete={onNodesDelete}
            onNodeClick={(_, n) => setSelectedId(n.id)}
            nodeTypes={nodeTypes} colorMode={theme}
            proOptions={{ hideAttribution: true }}
            deleteKeyCode={['Delete', 'Backspace']} fitView
          >
            <Background /><Controls />
          </ReactFlow>
          {issues.length > 0 && (
            <div className="issues">
              {issues.map((i, k) => (
                <div key={k} className={'issue ' + i.level} onClick={() => i.nodeId && setSelectedId(i.nodeId)}
                  style={i.nodeId ? { cursor: 'pointer' } : undefined}>• {i.message}</div>
              ))}
            </div>
          )}
        </div>

        <div className="right">
          <PropertyPanel node={selectedNode} allNodeIds={nodes.map((n) => n.id)} onChange={onParamChange} onDelete={onDeleteNode} onRename={renameNode} />
          <div className="logdrawer">
            <div className="log-head">
              <span className="lg-title">日志</span>
              <span className="log-filter">
                {(['info', 'warn', 'error'] as LogLevel[]).map((lv) => (
                  <button key={lv} className={levels[lv] ? 'on' : ''} onClick={() => setLevels((s) => ({ ...s, [lv]: !s[lv] }))}>{lv}</button>
                ))}
              </span>
              <span className="spacer" />
              <button className={'btn tiny' + (autoScroll ? ' active' : '')} onClick={() => setAutoScroll((v) => !v)}>滚动</button>
              <button className="btn tiny" onClick={() => setLog([])}>清空</button>
              <button className="btn tiny" onClick={() => setLogOpen((v) => !v)}>{logOpen ? '收起' : '展开'}</button>
            </div>
            <div ref={logRef} className={'log-body' + (logOpen ? '' : ' collapsed')} style={{ height: logOpen ? 170 : 0 }}>
              {shownLog.map((l, i) => (<div key={i} className={'logline ' + l.level}><span className="lt">{l.ts}</span>{l.msg}</div>))}
            </div>
          </div>
        </div>
      </div>

      {pub && (
        <>
          <div className="impl-mask" onClick={() => setPub(null)} />
          <div className="impl-picker">
            <div className="impl-dh"><span className="impl-name">发布为模板</span><div className="spacer" /><button className="btn sm ghost" onClick={() => setPub(null)}>✕</button></div>
            <div className="impl-pkbody">
              <div className="impl-field"><label>模板名称</label>
                <input type="text" value={pub.name} onChange={(e) => setPub({ ...pub, name: e.target.value })} /></div>
              <div className="impl-field"><label>版本</label>
                <input type="text" value={pub.version} onChange={(e) => setPub({ ...pub, version: e.target.value })} /></div>
              <div className="impl-field"><label>描述</label>
                <input type="text" value={pub.description} onChange={(e) => setPub({ ...pub, description: e.target.value })} /></div>
              <div className="impl-field"><label>暴露给实施的字段（其余锁定）</label>
                <label className="chk"><input type="checkbox" disabled={!pub.ov.source_url} checked={pub.expose.source} onChange={(e) => setPub({ ...pub, expose: { ...pub.expose, source: e.target.checked } })} />拉流地址{!pub.ov.source_url && '（无源节点）'}</label>
                <label className="chk"><input type="checkbox" disabled={!pub.ov.push} checked={pub.expose.push} onChange={(e) => setPub({ ...pub, expose: { ...pub.expose, push: e.target.checked } })} />是否推流{!pub.ov.push && '（无 rtmp_sink）'}</label>
                <label className="chk"><input type="checkbox" disabled={!pub.ov.alerts} checked={pub.expose.alerts} onChange={(e) => setPub({ ...pub, expose: { ...pub.expose, alerts: e.target.checked } })} />告警项{pub.ov.alerts ? `（${pub.ov.alerts.types.length} 类）` : '（无 alert）'}</label>
              </div>
              <div className="impl-locknote">实施台从模板实例化：只能改上面勾选的字段，节点/模型/融合等锁定。</div>
            </div>
            <div className="impl-df"><button className="btn" onClick={() => setPub(null)}>取消</button><div className="spacer" /><button className="btn primary" onClick={doPublish}>发布</button></div>
          </div>
        </>
      )}
    </div>
  );
}

export default function App() {
  return <ReactFlowProvider><Editor /></ReactFlowProvider>;
}
