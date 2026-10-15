import { useState } from 'react';
import { FieldDef } from '../schema/nodeSchema';

// 通用单字段渲染器：供节点属性面板与规则子表单共用。
export default function FieldInput({ path, f, value, onChange, allNodeIds }: {
  path: string;
  f: FieldDef;
  value: unknown;
  onChange: (val: unknown) => void;
  allNodeIds?: string[];
}) {
  const label = (
    <label className="flabel">
      {f.label || path}{f.required && <span className="req">*</span>}
      {f.hint && <span className="fhint" title={f.hint}> ⓘ</span>}
    </label>
  );

  if (f.type === 'bool')
    return <div className="field">{label}<input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /></div>;

  if (f.type === 'enum')
    return <div className="field">{label}
      <select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
        {(f.values || []).map((v) => <option key={v} value={v}>{v || '(空)'}</option>)}
      </select></div>;

  if (f.type === 'number' || f.type === 'int')
    return <div className="field">{label}
      <input type="number" step={f.step || (f.type === 'int' ? 1 : 0.01)} min={f.min} max={f.max}
        value={value === undefined ? '' : String(value)}
        onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))} /></div>;

  if (f.type === 'stringList') {
    const arr: string[] = Array.isArray(value) ? value : [];
    return <div className="field">{label}
      <input type="text" value={arr.join(', ')} placeholder="逗号分隔"
        onChange={(e) => onChange(e.target.value.split(',').map((s) => s.trim()).filter(Boolean))} /></div>;
  }

  if (f.type === 'nodeRefList') {
    const arr: string[] = Array.isArray(value) ? value : [];
    return <div className="field">{label}
      <div className="chips">
        {(allNodeIds || []).map((id) => (
          <label key={id} className="chip">
            <input type="checkbox" checked={arr.includes(id)}
              onChange={(e) => onChange(e.target.checked ? [...arr, id] : arr.filter((x) => x !== id))} />{id}
          </label>
        ))}
      </div></div>;
  }

  if (f.type === 'json') return <JsonField label={label} value={value} onChange={onChange} />;

  return <div className="field">{label}
    <input type="text" value={value === undefined ? '' : String(value)} placeholder={f.hint}
      onChange={(e) => onChange(e.target.value)} /></div>;
}

function JsonField({ label, value, onChange }: { label: React.ReactNode; value: unknown; onChange: (v: unknown) => void }) {
  const [text, setText] = useState(() => JSON.stringify(value ?? null, null, 2));
  const [err, setErr] = useState('');
  return <div className="field">{label}
    <textarea className="jsonbox" value={text} rows={5}
      onChange={(e) => { setText(e.target.value); try { onChange(JSON.parse(e.target.value || 'null')); setErr(''); } catch (x: any) { setErr(x.message); } }} />
    {err && <div className="err">JSON: {err}</div>}</div>;
}
