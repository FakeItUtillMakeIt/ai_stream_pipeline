import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow, Background, Controls, MiniMap, addEdge,
  useNodesState, useEdgesState, Connection, Node, Edge,
  MarkerType, ReactFlowProvider, useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import Palette from './components/Palette';
import PropertyPanel from './components/PropertyPanel';
import PipelineNode from './components/PipelineNode';
import { defaultParams } from './schema/nodeSchema';
import { toFlow, fromFlow } from './graph/flow';
import { validate, hasErrors, Issue } from './graph/validation';
import { PipelineDoc, NodeDef } from './types';
import { api, ApiSettings, waitState } from './api/client';

const nodeTypes = { pipelineNode: PipelineNode };

type LogLevel = 'info' | 'warn' | 'error';
type Status = 'waiting' | 'running' | 'error';
interface LogEntry { ts: string; level: LogLevel; msg: string; }

function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [paramsById, setParamsById] = useState<Record<string, NodeDef['params']>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);

  // 多管道：store 按 id 存文档，activeId 为当前编辑的管道
  const [store, setStore] = useState<Record<string, PipelineDoc>>({});
  const [activeId, setActiveId] = useState('');
  const [serverState, setServerState] = useState<Record<string, string>>({});

  // 日志
  const [log, setLog] = useState<LogEntry[]>([]);
  const [levels, setLevels] = useState<Record<LogLevel, boolean>>({ info: true, warn: true, error: true });
  const [autoScroll, setAutoScroll] = useState(true);
  const [logOpen, setLogOpen] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  // 主题 + 当前节点视觉状态
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (localStorage.getItem('editorTheme') as 'light' | 'dark') || 'dark');
  const [status, setStatus] = useState<Status>('waiting');

  const [settings, setSettings] = useState<ApiSettings>(() => {
    try { return JSON.parse(localStorage.getItem('apiSettings') || ''); }
    catch { return { baseUrl: '', token: '' }; }
  });
  const fileRef = useRef<HTMLInputElement>(null);
  const counter = useRef(1);
  const rf = useReactFlow();

  useEffect(() => { localStorage.setItem('apiSettings', JSON.stringify(settings)); }, [settings]);
  useEffect(() => { localStorage.setItem('editorTheme', theme); }, [theme]);
  useEffect(() => { setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, status } }))); }, [status, setNodes]);
  useEffect(() => { if (autoScroll && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log, autoScroll, logOpen]);

  const appendLog = (msg: string, level: LogLevel = 'info') =>
    setLog((l) => [...l, { ts: new Date().toLocaleTimeString(), level, msg }].slice(-500));

  const genId = (type: string) => {
    let id: string;
    do { id = `${type}_${counter.current++}`; } while (nodes.some((n) => n.id === id));
    return id;
  };

  // ===== 当前编辑态 ↔ store =====
  const buildDoc = (id: string): PipelineDoc => fromFlow(id, nodes, edges, paramsById);

  const loadDoc = (doc: PipelineDoc) => {
    const f = toFlow(doc);
    const st: Status = serverState[doc.id] === 'running' ? 'running'
      : serverState[doc.id] === 'load_failed' ? 'error' : 'waiting';
    setNodes(f.nodes.map((n) => ({ ...n, data: { ...n.data, status: st } })));
    setEdges(f.edges.map((e) => ({ ...e, markerEnd: { type: MarkerType.ArrowClosed } })));
    const p: Record<string, NodeDef['params']> = {};
    doc.graph.nodes.forEach((n) => { p[n.id] = n.params || {}; });
    setParamsById(p);
    setActiveId(doc.id);
    setSelectedId(null);
    setIssues([]);
    setStatus(st);
    counter.current = Math.max(counter.current, doc.graph.nodes.length + 1);
  };

  const commitActive = () => {
    if (!activeId) return;
    const doc = buildDoc(activeId);
    setStore((s) => ({ ...s, [activeId]: doc }));
  };

  const switchTo = (id: string) => {
    if (id === activeId) return;
    commitActive();
    const doc = store[id];
    if (doc) loadDoc(doc);
  };

  const newPipeline = () => {
    const id = window.prompt('新管道 id', 'pipeline_' + (Object.keys(store).length + 1));
    if (!id || !id.trim()) return;
    const key = id.trim();
    if (store[key]) { appendLog(`管道 ${key} 已存在`, 'warn'); return; }
    commitActive();
    const doc: PipelineDoc = { id: key, graph: { nodes: [], edges: [] } };
    setStore((s) => ({ ...s, [key]: doc }));
    loadDoc(doc);
    appendLog(`新建管道 ${key}`);
  };

  // ===== 画布编辑 =====
  const addNodeAt = (type: string, position: { x: number; y: number }) => {
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
  }, [rf, status, nodes.length]);
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

  const currentDoc = (): PipelineDoc => buildDoc(activeId);

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
    URL.revokeObjectURL(a.href);
    appendLog('已导出 JSON');
  };

  const importJson = (file: File) => {
    file.text().then((t) => {
      try {
        const doc: PipelineDoc = JSON.parse(t);
        if (!doc.id) doc.id = file.name.replace(/\.json$/i, '');
        commitActive();
        setStore((s) => ({ ...s, [doc.id]: doc }));
        loadDoc(doc);
        appendLog(`导入 ${doc.id}：${doc.graph.nodes.length} 节点 / ${doc.graph.edges.length} 边`);
      } catch (e: any) { appendLog('导入失败: ' + e.message, 'error'); }
    });
  };

  // ===== 服务器交互 =====
  const guard = (): boolean => { if (!activeId) { appendLog('未选择管道', 'error'); return false; } const iss = validate(currentDoc()); setIssues(iss); return !hasErrors(iss); };

  const refreshStatus = async () => {
    const r = await api.list(settings);
    if (r.ok && r.json) {
      const map: Record<string, string> = {};
      const arr = Array.isArray(r.json) ? r.json : (r.json.pipelines || r.json.items || []);
      for (const it of arr) { const id = it.id || it.name; const st = it.state || it.status || (it.running ? 'running' : 'stopped'); if (id) map[id] = st; }
      setServerState(map);
      appendLog('已刷新服务器管道状态');
    } else {
      appendLog('List → ' + JSON.stringify(r.json), 'error');
    }
  };

  const doBuild = async () => {
    if (!guard()) return;
    commitActive();
    setStatus('waiting');
    const r = await api.build(settings, currentDoc());
    appendLog(`Build ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (!r.ok) { setStatus('error'); return; }
    await waitState(settings, activeId, ['stopped', 'running', 'load_failed'], 30000,
      (s) => { const st = s?.state || s?.status; if (st === 'load_failed') { setStatus('error'); appendLog('加载失败', 'error'); } });
    refreshStatus();
  };
  const doStart = async () => {
    if (!activeId) return;
    const r = await api.start(settings, activeId);
    appendLog(`Start ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    setStatus(r.ok ? 'running' : 'error'); refreshStatus();
  };
  const doStop = async () => {
    if (!activeId) return;
    const r = await api.stop(settings, activeId);
    appendLog(`Stop ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (r.ok) setStatus('waiting'); refreshStatus();
  };
  const doDelete = async () => {
    if (!activeId) return;
    const r = await api.del(settings, activeId);
    appendLog(`Delete ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
    if (r.ok) {
      setStatus('waiting');
      setStore((s) => { const n = { ...s }; delete n[activeId]; return n; });
      const rest = Object.keys(store).filter((k) => k !== activeId);
      if (rest[0]) loadDoc(store[rest[0]]); else loadDoc({ id: '', graph: { nodes: [], edges: [] } });
      refreshStatus();
    }
  };
  const doSave = async () => {
    if (!guard()) return;
    commitActive();
    const r = await api.saveConfig(settings, currentDoc());
    appendLog(`保存到服务器 ${activeId} → ${r.status} ${JSON.stringify(r.json)}`, r.ok ? 'info' : 'error');
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

  const shownLog = log.filter((l) => levels[l.level]);
  const pipelineIds = Object.keys(store);

  return (
    <div className={`app theme-${theme}`}>
      <div className="topbar">
        <span className="title">Pipeline Editor</span>
        <select className="pid" value={activeId} onChange={(e) => switchTo(e.target.value)} title="切换已加载的管道">
          <option value="">（选择管道）</option>
          {pipelineIds.map((id) => (
            <option key={id} value={id}>{serverState[id] === 'running' ? '● ' : ''}{id}</option>
          ))}
        </select>
        <button className="btn tiny" onClick={newPipeline} title="新建管道">＋新建</button>
        <input className="cfg" value={settings.baseUrl} placeholder="API base(同源留空)" onChange={(e) => setSettings({ ...settings, baseUrl: e.target.value })} />
        <input className="cfg" value={settings.token} placeholder="API Token" onChange={(e) => setSettings({ ...settings, token: e.target.value })} />
        <div className="spacer" />
        <button className="btn" onClick={() => fileRef.current?.click()}>导入</button>
        <input ref={fileRef} type="file" accept=".json" hidden onChange={(e) => e.target.files?.[0] && importJson(e.target.files[0])} />
        <button className="btn" onClick={exportJson}>导出JSON</button>
        <button className="btn" onClick={doValidate}>校验</button>
        <button className="btn" onClick={doSave}>保存到服务器</button>
        <button className="btn primary" onClick={doBuild}>Build</button>
        <button className="btn" onClick={doStart}>Start</button>
        <button className="btn" onClick={doStop}>Stop</button>
        <button className="btn danger" onClick={doDelete}>Delete</button>
        <button className="btn" onClick={refreshStatus} title="刷新服务器管道状态">刷新状态</button>
        <button className="btn" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}>{theme === 'dark' ? '☀' : '🌙'}</button>
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
            deleteKeyCode={['Delete', 'Backspace']} fitView
          >
            <Background /><Controls /><MiniMap pannable zoomable />
          </ReactFlow>
          {issues.length > 0 && (
            <div className="issues">
              {issues.map((i, k) => <div key={k} className={'issue ' + i.level}>• {i.message}</div>)}
            </div>
          )}
        </div>

        <div className="right">
          <PropertyPanel node={selectedNode} allNodeIds={nodes.map((n) => n.id)} onChange={onParamChange} onDelete={onDeleteNode} />
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
    </div>
  );
}

export default function App() {
  return <ReactFlowProvider><Editor /></ReactFlowProvider>;
}
