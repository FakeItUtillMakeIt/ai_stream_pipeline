import { useState } from 'react';
import { RULE_SCHEMA, ruleTypes, defaultRule } from '../schema/ruleSchema';
import {
  ALERT_CATEGORIES, RULE_META, ModelCapabilities,
  categoryLabel, ruleUnavailableReason, severityTone,
} from '../schema/alertCatalog';
import { getPath, setPath, deletePath } from '../types';
import FieldInput from './FieldInput';

interface RuleItem { type: string; params: Record<string, unknown>; }

export default function RuleEditor({ value, onChange, modelCaps }: {
  value: unknown;
  onChange: (rules: RuleItem[]) => void;
  allNodeIds?: string[];
  modelCaps?: ModelCapabilities;
}) {
  const rules: RuleItem[] = Array.isArray(value) ? (value as RuleItem[]) : [];
  const [raw, setRaw] = useState(false);
  const [rawText, setRawText] = useState('');
  const [rawErr, setRawErr] = useState('');
  const [open, setOpen] = useState<Record<number, boolean>>({});

  const setRules = (next: RuleItem[]) => onChange(next);
  const unavailable = (type: string) => (modelCaps ? ruleUnavailableReason(type, modelCaps) : null);

  const addRule = (type: string) => {
    const reason = unavailable(type);
    if (reason) return;
    setRules([...rules, defaultRule(type)]);
  };

  const updateType = (i: number, type: string) => {
    const reason = unavailable(type);
    if (reason) return;
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

  const typeOptions = (current?: string) => {
    const groups = ALERT_CATEGORIES
      .map((cat) => ({
        cat,
        types: ruleTypes()
          .filter((t) => (RULE_META[t]?.category ?? '') === cat.id)
          .sort((a, b) => a.localeCompare(b)),
      }))
      .filter((g) => g.types.length);
    const unknown = ruleTypes().filter((t) => !RULE_META[t]);
    return (
      <>
        {groups.map(({ cat, types }) => (
          <optgroup key={cat.id} label={cat.label}>
            {types.map((t) => {
              const reason = unavailable(t);
              return (
                <option key={t} value={t} disabled={!!reason && t !== current}>
                  {RULE_SCHEMA[t].label} ({t}){reason ? ` — ${reason}` : ''}
                </option>
              );
            })}
          </optgroup>
        ))}
        {unknown.length > 0 && (
          <optgroup label="未分类">
            {unknown.map((t) => <option key={t} value={t}>{RULE_SCHEMA[t].label} ({t})</option>)}
          </optgroup>
        )}
      </>
    );
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

  const renderRule = (r: RuleItem, i: number) => {
    const def = RULE_SCHEMA[r.type];
    const meta = RULE_META[r.type];
    const isOpen = !!open[i];
    const reason = unavailable(r.type);
    return (
      <div key={i} className={'rule-card' + (reason ? ' rule-card-unavailable' : '')}>
        <div className="rule-card-head" onClick={() => setOpen((o) => ({ ...o, [i]: !o[i] }))}>
          <span className="caret">{isOpen ? '▾' : '▸'}</span>
          <span className="rule-card-title">
            <b>{def?.label || r.type}</b>
            <code className="rule-type">{r.type}</code>
            {meta && (
              <>
                <span className="rule-cat">{categoryLabel(meta.category)}</span>
                <span className={'sev-badge ' + severityTone(meta.severity)}>{meta.severity}</span>
              </>
            )}
            <span className="rule-name">{String(r.params?.name ?? '')}</span>
            {reason && <span className="rule-warn" title={reason}>⚠ {reason}</span>}
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
                {typeOptions(r.type)}
              </select>
              {reason && <div className="rule-warn-block">当前模板不能启用：{reason}</div>}
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
  };

  const indexed = rules.map((r, i) => ({ r, i }));
  const knownGroups = ALERT_CATEGORIES
    .map((cat) => ({
      cat,
      items: indexed
        .filter(({ r }) => (RULE_META[r.type]?.category ?? '') === cat.id)
        .sort((a, b) => a.r.type.localeCompare(b.r.type)),
    }))
    .filter((g) => g.items.length);
  const unknownItems = indexed.filter(({ r }) => !RULE_META[r.type]);

  return (
    <div className="rule-editor">
      <div className="rule-toolbar">
        <select value="" onChange={(e) => { if (e.target.value) addRule(e.target.value); }}>
          <option value="">＋ 添加规则…</option>
          {typeOptions()}
        </select>
        <span className="spacer" />
        <button className="btn tiny" onClick={startRaw}>原始 JSON</button>
      </div>

      {rules.length === 0 && <div className="hint">暂无规则，从上方选择类型添加。</div>}

      {knownGroups.map(({ cat, items }) => {
        const enabled = items.filter(({ r }) => !unavailable(r.type)).length;
        return (
          <div className="rule-group" key={cat.id}>
            <div className="rule-group-head">
              <span>{cat.label}</span>
              <span className="rule-group-sub">{enabled}/{items.length} 可用</span>
            </div>
            {items.map(({ r, i }) => renderRule(r, i))}
          </div>
        );
      })}
      {unknownItems.length > 0 && (
        <div className="rule-group">
          <div className="rule-group-head"><span>未分类</span></div>
          {unknownItems.map(({ r, i }) => renderRule(r, i))}
        </div>
      )}
    </div>
  );
}
