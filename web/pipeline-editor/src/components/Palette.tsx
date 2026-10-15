import { useState } from 'react';
import { SCHEMA, CATEGORIES, CATEGORY_LABEL } from '../schema/nodeSchema';

export default function Palette() {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const toggle = (cat: string) => setCollapsed((c) => ({ ...c, [cat]: !c[cat] }));

  const onDragStart = (e: React.DragEvent, type: string) => {
    e.dataTransfer.setData('application/reactflow', type);
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="palette">
      <div className="palette-title">节点库 <span className="palette-tip">拖到画布</span></div>
      {CATEGORIES.map((cat) => {
        const types = Object.keys(SCHEMA).filter((t) => SCHEMA[t].category === cat);
        if (!types.length) return null;
        const isCollapsed = !!collapsed[cat];
        return (
          <div key={cat} className="palette-group">
            <div
              className={'palette-cat' + (isCollapsed ? ' collapsed' : '')}
              onClick={() => toggle(cat)}
            >
              <span className="caret">{isCollapsed ? '▸' : '▾'}</span>
              {CATEGORY_LABEL[cat] || cat}
              <span style={{ marginLeft: 'auto', color: 'var(--muted)' }}>{types.length}</span>
            </div>
            {!isCollapsed && (
              <div className="palette-items">
                {types.map((t) => (
                  <div
                    key={t}
                    className="palette-item"
                    draggable
                    onDragStart={(e) => onDragStart(e, t)}
                    style={{ borderLeftColor: SCHEMA[t].color }}
                    title={`${t}（拖到画布）`}
                  >
                    <span className="dot" style={{ background: SCHEMA[t].color }} />
                    {SCHEMA[t].label}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
