import { useState } from 'react';
import { NodeDef, getPath, setPath, deletePath } from '../types';
import { SCHEMA, FieldDef, nodeTypesInCategory } from '../schema/nodeSchema';

interface Props {
  node: NodeDef | null;
  allNodeIds: string[];
  onChange: (node: NodeDef) => void;
  onDelete: (id: string) => void;
}

export default function PropertyPanel({ node, allNodeIds, onChange, onDelete }: Props) {
  const [rawMode, setRawMode] = useState(false);
  const [rawText, setRawText] = useState('');
  const [rawErr, setRawErr] = useState('');

  if (!node) return <div className="prop"><div className="prop-empty">选中一个节点以编辑参数</div></div>;

  const def = SCHEMA[node.type];

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
    try {
      const parsed = JSON.parse(rawText);
      onChange({ ...node, params: parsed });
      setRawMode(false);
    } catch (e: any) { setRawErr('JSON 解析失败: ' + e.message); }
  };

  return (
    <div className="prop">
      <div className="prop-head">
        <div>
          <div className="prop-type">{def?.label || node.type}</div>
          <div className="prop-id">{node.id}</div>
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
            {Object.entries(def?.fields || {}).map(([path, f]) => (
              <Field key={path} path={path} f={f} params={node.params} onUpdate={update} allNodeIds={allNodeIds} />
            ))}
            {/* schema 未覆盖但 params 里存在的键，提示去原始 JSON 编辑 */}
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
  const topKeys = Object.keys(params);
  const covered = new Set(schemaPaths.map((p) => p.split('.')[0]));
  const extra = topKeys.filter((k) => !covered.has(k));
  if (!extra.length) return null;
  return <div className="hint">另有未定义字段：{extra.join(', ')}（请用"原始 JSON 编辑"修改）</div>;
}

function Field({ path, f, params, onUpdate, allNodeIds }: {
  path: string; f: FieldDef; params: Record<string, unknown>;
  onUpdate: (path: string, val: unknown) => void; allNodeIds: string[];
}) {
  const val = getPath(params, path);
  const label = (
    <label className="flabel">
      {f.label || path}{f.required && <span className="req">*</span>}
      {f.hint && <span className="fhint" title={f.hint}> ⓘ</span>}
    </label>
  );

  if (f.type === 'bool') {
    return <div className="field">{label}
      <input type="checkbox" checked={!!val} onChange={(e) => onUpdate(path, e.target.checked)} /></div>;
  }
  if (f.type === 'enum') {
    return <div className="field">{label}
      <select value={String(val ?? '')} onChange={(e) => onUpdate(path, e.target.value)}>
        {(f.values || []).map((v) => <option key={v} value={v}>{v || '(空)'}</option>)}
      </select></div>;
  }
  if (f.type === 'number' || f.type === 'int') {
    return <div className="field">{label}
      <input type="number" step={f.step || (f.type === 'int' ? 1 : 0.01)} min={f.min} max={f.max}
        value={val === undefined ? '' : String(val)}
        onChange={(e) => { const n = e.target.value === '' ? undefined : Number(e.target.value); onUpdate(path, n); }} /></div>;
  }
  if (f.type === 'stringList') {
    const arr: string[] = Array.isArray(val) ? val : [];
    return <div className="field">{label}
      <input type="text" value={arr.join(', ')} placeholder="逗号分隔"
        onChange={(e) => onUpdate(path, e.target.value.split(',').map((s) => s.trim()).filter(Boolean))} /></div>;
  }
  if (f.type === 'nodeRefList') {
    const arr: string[] = Array.isArray(val) ? val : [];
    return <div className="field">{label}
      <div className="chips">
        {allNodeIds.map((id) => (
          <label key={id} className="chip">
            <input type="checkbox" checked={arr.includes(id)}
              onChange={(e) => onUpdate(path, e.target.checked ? [...arr, id] : arr.filter((x) => x !== id))} />
            {id}
          </label>
        ))}
      </div></div>;
  }
  if (f.type === 'json') {
    return <JsonField label={label} value={val} onChange={(v) => onUpdate(path, v)} />;
  }
  // string / path
  return <div className="field">{label}
    <input type="text" value={val === undefined ? '' : String(val)} placeholder={f.hint}
      onChange={(e) => onUpdate(path, e.target.value)} /></div>;
}

function JsonField({ label, value, onChange }: { label: React.ReactNode; value: unknown; onChange: (v: unknown) => void }) {
  const [text, setText] = useState(() => JSON.stringify(value ?? {}, null, 2));
  const [err, setErr] = useState('');
  return <div className="field">{label}
    <textarea className="jsonbox" value={text} rows={6}
      onChange={(e) => { setText(e.target.value); try { onChange(JSON.parse(e.target.value || 'null')); setErr(''); } catch (x: any) { setErr(x.message); } }} />
    {err && <div className="err">JSON: {err}</div>}</div>;
}
