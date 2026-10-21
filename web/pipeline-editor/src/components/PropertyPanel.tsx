import { useState } from 'react';
import { NodeDef, getPath, setPath, deletePath } from '../types';
import { SCHEMA } from '../schema/nodeSchema';
import FieldInput from './FieldInput';
import RuleEditor from './RuleEditor';

interface Props {
  node: NodeDef | null;
  allNodeIds: string[];
  onChange: (node: NodeDef) => void;
  onDelete: (id: string) => void;
  onRename: (oldId: string, newId: string) => void;
}

export default function PropertyPanel({ node, allNodeIds, onChange, onDelete, onRename }: Props) {
  const [rawMode, setRawMode] = useState(false);
  const [rawText, setRawText] = useState('');
  const [rawErr, setRawErr] = useState('');
  const [idDraft, setIdDraft] = useState(node?.id ?? '');
  const [editId, setEditId] = useState(false);

  if (!node) return <div className="prop"><div className="prop-empty">选中一个节点以编辑参数</div></div>;
  if (idDraft !== node.id && !editId) setIdDraft(node.id);

  const def = SCHEMA[node.type];

  const commitId = () => {
    setEditId(false);
    if (idDraft.trim() && idDraft.trim() !== node.id) onRename(node.id, idDraft.trim());
    else setIdDraft(node.id);
  };

  const update = (path: string, val: unknown) => {
    const params = JSON.parse(JSON.stringify(node.params));
    if (val === undefined || val === '' || (Array.isArray(val) && val.length === 0 && path !== 'rules'))
      deletePath(params, path);
    else
      setPath(params, path, val);
    onChange({ ...node, params });
  };

  const startRaw = () => { setRawText(JSON.stringify(node.params, null, 2)); setRawErr(''); setRawMode(true); };
  const applyRaw = () => {
    try { onChange({ ...node, params: JSON.parse(rawText) }); setRawMode(false); }
    catch (e: any) { setRawErr('JSON 解析失败: ' + e.message); }
  };

  return (
    <div className="prop">
      <div className="prop-head">
        <div style={{ minWidth: 0 }}>
          <div className="prop-type">{def?.label || node.type}</div>
          {editId ? (
            <input className="prop-id-edit" value={idDraft} autoFocus
              onChange={(e) => setIdDraft(e.target.value)} onBlur={commitId}
              onKeyDown={(e) => { if (e.key === 'Enter') commitId(); if (e.key === 'Escape') { setIdDraft(node.id); setEditId(false); } }} />
          ) : (
            <div className="prop-id" title="双击改名" onDoubleClick={() => setEditId(true)}>
              {node.id} <button className="btn tiny" onClick={() => setEditId(true)}>✎</button>
            </div>
          )}
        </div>
        <button className="btn danger" onClick={() => onDelete(node.id)}>删除节点</button>
      </div>

      {rawMode ? (
        <div className="prop-raw">
          <div className="hint">直接编辑 params（JSON）</div>
          <textarea value={rawText} rows={18} onChange={(e) => setRawText(e.target.value)} />
          {rawErr && <div className="err">{rawErr}</div>}
          <div className="row">
            <button className="btn primary" onClick={applyRaw}>应用</button>
            <button className="btn" onClick={() => setRawMode(false)}>取消</button>
          </div>
        </div>
      ) : (
        <>
          <div className="prop-fields">
            {Object.entries(def?.fields || {}).map(([path, f]) =>
              f.type === 'rulelist' ? (
                <div className="field" key={path}>
                  <label className="flabel">{f.label || path}{f.required && <span className="req">*</span>}</label>
                  <RuleEditor value={getPath(node.params, path)} onChange={(v) => update(path, v)} allNodeIds={allNodeIds} />
                </div>
              ) : (
                <FieldInput key={path} path={path} f={f} value={getPath(node.params, path)}
                  onChange={(v) => update(path, v)} allNodeIds={allNodeIds} />
              )
            )}
            <ExtraKeys params={node.params} schemaPaths={Object.keys(def?.fields || {})} />
          </div>
          <div className="row">
            <button className="btn" onClick={startRaw}>用原始 JSON 编辑</button>
          </div>
        </>
      )}
    </div>
  );
}

function ExtraKeys({ params, schemaPaths }: { params: Record<string, unknown>; schemaPaths: string[] }) {
  const covered = new Set(schemaPaths.map((p) => p.split('.')[0]));
  const extra = Object.keys(params).filter((k) => !covered.has(k));
  if (!extra.length) return null;
  return <div className="hint">另有未定义字段：{extra.join(', ')}（用"原始 JSON 编辑"修改）</div>;
}
