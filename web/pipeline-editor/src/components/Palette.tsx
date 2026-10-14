import { SCHEMA, CATEGORIES, CATEGORY_LABEL } from '../schema/nodeSchema';

export default function Palette({ onAdd }: { onAdd: (type: string) => void }) {
  return (
    <div className="palette">
      <div className="palette-title">节点库</div>
      {CATEGORIES.map((cat) => {
        const types = Object.keys(SCHEMA).filter((t) => SCHEMA[t].category === cat);
        if (!types.length) return null;
        return (
          <div key={cat} className="palette-group">
            <div className="palette-cat">{CATEGORY_LABEL[cat] || cat}</div>
            {types.map((t) => (
              <button
                key={t}
                className="palette-item"
                style={{ borderColor: SCHEMA[t].color }}
                onClick={() => onAdd(t)}
                title={t}
              >
                <span className="dot" style={{ background: SCHEMA[t].color }} />
                {SCHEMA[t].label}
              </button>
            ))}
          </div>
        );
      })}
    </div>
  );
}
