import { useState } from 'react';
import { RULE_SCHEMA, ruleTypes, defaultRule } from '../schema/ruleSchema';
import { getPath, setPath, deletePath } from '../types';
import FieldInput from './FieldInput';

interface RuleItem { type: string; params: Record<string, unknown>; }

export default function RuleEditor({ value, onChange, allNodeIds }: {
  value: unknown;
  onChange: (rules: RuleItem[]) => void;
  allNodeIds?: string[];
}) {
  const rules: RuleItem[] = Array.isArray(value) ? (value as RuleItem[]) : [];
  const [raw, setRaw] = useState(false);
  const [rawText, setRawText] = useState('');
  const [rawErr, setRawErr] = useState('');
  const [open, setOpen] = useState<Record<number, boolean>>({});

  const setRules = (next: RuleItem[]) => onChange(next);

  const addRule = (type: string) => setRules([...rules, defaultRule(type)]);

  const updateType = (i: number, type: string) => {
    const next = rules.map((r, k) => (k === i ? { type, params: defaultRule(type).params } : r));
    setRules(next);
  };
  const updateField = (i: number, path: string, val: unknown) => {
    const next = rules.map((r, k) => {
      if (k !== i) return r;
      const params = JSON.parse(JSON.stringify(r.params || {}));
      if (val === undefined || val === '' || (Array.isArray(val) && val.length === 0)) deletePath(params, path);
      else setPath(params, path, val);
      return { ...r, params };
    });
    setRules(next);
  };
  const remove = (i: number) => setRules(rules.filter((_, k) => k !== i));
  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir; if (j < 0 || j >= rules.length) return;
    const next = [...rules]; [next[i], next[j]] = [next[j], next[i]]; setRules(next);
  };

  const startRaw = () => { setRawText(JSON.stringify(rules, null, 2)); setRawErr(''); setRaw(true); };
  const applyRaw = () => {
    try { const parsed = JSON.parse(rawText); if (!Array.isArray(parsed)) throw new Error('必须是数组'); setRules(parsed); setRaw(false); }
    catch (e: any) { setRawErr(e.message); }
  };

  if (raw) {
    return (
      <div className="rule-editor">
        <div className="hint">直接编辑 rules（JSON 数组）</div>
        <textarea className="jsonbox" value={rawText} rows={16} onChange={(e) => setRawText(e.target.value)} />
        {rawErr && <div className="err">{rawErr}</div>}
        <div className="row">
          <button className="btn primary" onClick={applyRaw}>应用</button>
          <button className="btn" onClick={() => setRaw(false)}>返回表单</button>
        </div>
      </div>
    );
  }

  return (
    <div className="rule-editor">
      <div className="rule-toolbar">
        <select value="" onChange={(e) => { if (e.target.value) addRule(e.target.value); }}>
          <option value="">＋ 添加规则…</option>
          {ruleTypes().map((t) => <option key={t} value={t}>{RULE_SCHEMA[t].label} ({t})</option>)}
        </select>
        <span className="spacer" />
        <button className="btn tiny" onClick={startRaw}>原始 JSON</button>
      </div>

      {rules.length === 0 && <div className="hint">暂无规则，从上方选择类型添加。</div>}

      {rules.map((r, i) => {
        const def = RULE_SCHEMA[r.type];
        const isOpen = !!open[i];
        return (
          <div key={i} className="rule-card">
            <div className="rule-card-head" onClick={() => setOpen((o) => ({ ...o, [i]: !o[i] }))}>
              <span className="caret">{isOpen ? '▾' : '▸'}</span>
              <span className="rule-card-title">
                <b>{def?.label || r.type}</b>
                <code className="rule-type">{r.type}</code>
                <span className="rule-name">{String(r.params?.name ?? '')}</span>
              </span>
              <span className="rule-actions">
                <button className="btn tiny" onClick={(e) => { e.stopPropagation(); move(i, -1); }}>↑</button>
                <button className="btn tiny" onClick={(e) => { e.stopPropagation(); move(i, 1); }}>↓</button>
                <button className="btn tiny danger" title="删除规则" onClick={(e) => { e.stopPropagation(); remove(i); }}>✕</button>
              </span>
            </div>
            {isOpen && (
              <div className="rule-card-body">
                <div className="field">
                  <label className="flabel">规则类型</label>
                  <select value={r.type} onChange={(e) => updateType(i, e.target.value)}>
                    {ruleTypes().map((t) => <option key={t} value={t}>{RULE_SCHEMA[t].label} ({t})</option>)}
                  </select>
                </div>
                {def && Object.entries(def.fields).map(([path, f]) => (
                  <FieldInput key={path} path={path} f={f}
                    value={getPath(r.params || {}, path)}
                    onChange={(v) => updateField(i, path, v)} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
