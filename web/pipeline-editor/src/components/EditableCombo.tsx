import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';

interface Props {
  value: string;
  options: string[];
  placeholder?: string;
  onChange: (value: string) => void;
}

/**
 * 可编辑下拉框：输入框 + 下拉箭头。
 * 展开时始终展示全部预设（不按当前值过滤），也允许直接输入任意值。
 */
export default function EditableCombo({ value, options, placeholder, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocMouseDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, [open]);

  const choose = (v: string) => {
    onChange(v);
    setOpen(false);
    setActive(-1);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(options.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter' && open && active >= 0) {
      e.preventDefault();
      choose(options[active]);
    } else if (e.key === 'Escape') {
      setOpen(false);
      setActive(-1);
    }
  };

  return (
    <div className="editable-combo" ref={rootRef}>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        autoComplete="off"
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onClick={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      <button
        type="button"
        className="editable-combo-toggle"
        tabIndex={-1}
        aria-label="选择常用地址"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => setOpen((o) => !o)}
      >
        ▾
      </button>
      {open && options.length > 0 && (
        <div className="editable-combo-list" role="listbox">
          {options.map((opt, i) => (
            <button
              type="button"
              key={opt}
              role="option"
              aria-selected={i === active}
              className={'editable-combo-item' + (i === active ? ' active' : '')}
              title={opt}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => choose(opt)}
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
