import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow, Background, Controls, MiniMap, addEdge,
  useNodesState, useEdgesState, Connection, Node, Edge,
  MarkerType, ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import Palette from './components/Palette';
import PropertyPanel from './components/PropertyPanel';
import PipelineNode from './components/PipelineNode';
import { SCHEMA, defaultParams } from './schema/nodeSchema';
import { toFlow, fromFlow } from './graph/flow';
import { validate, hasErrors, Issue } from './graph/validation';
import { PipelineDoc, NodeDef } from './types';
import { api, ApiSettings, waitState } from './api/client';

const nodeTypes = { pipelineNode: PipelineNode };

function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [paramsById, setParamsById] = useState<Record<string, NodeDef['params']>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pipelineId, setPipelineId] = useState('my_pipeline');
  const [issues, setIssues] = useState<Issue[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [settings, setSettings] = useState<ApiSettings>(() => {
    try { return JSON.parse(localStorage.getItem('apiSettings') || ''); }
    catch { return { baseUrl: '', token: '' }; }
  });
  const fileRef = useRef<HTMLInputElement>(null);
  const counter = useRef(1);

  useEffect(() => { localStorage.setItem('apiSettings', JSON.stringify(settings)); }, [settings]);

  const appendLog = (m: string) => setLog((l) => [`[${new Date().toLocaleTimeString()}] ${m}`, ...l].slice(0, 100));

  const genId = (type: string) => {
    let id: string;
    do { id = `${type}_${counter.current++}`; } while (nodes.some((n) => n.id === id));
    return id;
  };

  const addNode = (type: string) => {
    const id = genId(type);
    const n: Node = {
      id, type: 'pipelineNode',
      position: { x: 120 + (nodes.length % 5) * 220, y: 80 + Math.floor(nodes.length / 5) * 130 },
      data: { label: id, nodeType: type },
    };
    setNodes((ns) => ns.concat(n));
    setParamsById((p) => ({ ...p, [id]: defaultParams(type) }));
    setSelectedId(id);
  };

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

  const currentDoc = (): PipelineDoc => fromFlow(pipelineId, nodes, edges, paramsById);

  const doValidate = () => { const iss = validate(currentDoc()); setIssues(iss); appendLog(`校验: ${iss.filter(i=>i.level==='error').length} 错误 / ${iss.filter(i=>i.level==='warn').length} 警告`); return iss; };

  const exportJson = () => {
    const doc = currentDoc();
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `${pipelineId || 'pipeline'}.json`; a.click();
    URL.revokeObjectURL(a.href);
  };

  const importJson = (file: File) => {
    file.text().then((t) => {
      try {
        const doc: PipelineDoc = JSON.parse(t);
        const f = toFlow(doc);
        setNodes(f.nodes); setEdges(f.edges.map((e) => ({ ...e, markerEnd: { type: MarkerType.ArrowClosed } })));
        const p: Record<string, NodeDef['params']> = {};
        doc.graph.nodes.forEach((n) => { p[n.id] = n.params || {}; });
        setParamsById(p);
        setPipelineId(doc.id || 'my_pipeline');
        counter.current = doc.graph.nodes.length + 1;
        appendLog(`已导入 ${doc.graph.nodes.length} 节点 / ${doc.graph.edges.length} 边`);
      } catch (e: any) { appendLog('导入失败: ' + e.message); }
    });
  };

  // ===== API =====
  const guard = (): boolean => { const iss = validate(currentDoc()); setIssues(iss); return !hasErrors(iss); };

  const doBuild = async () => {
    if (!guard()) { appendLog('校验未通过，取消 Build'); return; }
    const r = await api.build(settings, currentDoc());
    appendLog(`Build → ${r.status} ${JSON.stringify(r.json)}`);
    const st = await waitState(settings, pipelineId, ['stopped', 'running', 'load_failed'], 30000, (s) => appendLog('status: ' + JSON.stringify(s)));
    void st;
  };
  const doStart = async () => { const r = await api.start(settings, pipelineId); appendLog(`Start → ${r.status} ${JSON.stringify(r.json)}`); };
  const doStop = async () => { const r = await api.stop(settings, pipelineId); appendLog(`Stop → ${r.status} ${JSON.stringify(r.json)}`); };
  const doDelete = async () => { const r = await api.del(settings, pipelineId); appendLog(`Delete → ${r.status} ${JSON.stringify(r.json)}`); };
  const doList = async () => { const r = await api.list(settings); appendLog('List → ' + JSON.stringify(r.json)); };
  const doSave = async () => {
    if (!guard()) { appendLog('校验未通过，取消 Save'); return; }
    const r = await api.saveConfig(settings, currentDoc());
    appendLog(`保存到服务器 → ${r.status} ${JSON.stringify(r.json)}`);
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

  return (
    <div className="app">
      <div className="topbar">
        <span className="title">Pipeline Editor</span>
        <input className="pid" value={pipelineId} onChange={(e) => setPipelineId(e.target.value)} placeholder="pipeline id" />
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
        <button className="btn" onClick={doList}>List</button>
      </div>

      <div className="main">
        <Palette onAdd={addNode} />
        <div className="canvas">
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} onNodesDelete={onNodesDelete}
            onNodeClick={(_, n) => setSelectedId(n.id)}
            nodeTypes={nodeTypes}
            deleteKeyCode={['Delete', 'Backspace']}
            fitView
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>
          {issues.length > 0 && (
            <div className="issues">
              {issues.map((i, k) => <div key={k} className={'issue ' + i.level}>• {i.message}</div>)}
            </div>
          )}
        </div>
        <div className="right">
          <PropertyPanel
            node={selectedNode}
            allNodeIds={nodes.map((n) => n.id)}
            onChange={onParamChange}
            onDelete={onDeleteNode}
          />
          <div className="log">
            <div className="log-title">日志</div>
            {log.map((l, i) => <div key={i} className="logline">{l}</div>)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return <ReactFlowProvider><Editor /></ReactFlowProvider>;
}

// 让 SCHEMA 在 tree-shaking 下保留（调试用）
void SCHEMA;
