import { useState } from "react";

interface JsonTreeProps {
  data: unknown;
  name?: string | number;
  depth?: number;
  defaultOpen?: boolean;
}

export function JsonTree({ data, name, depth = 0, defaultOpen = true }: JsonTreeProps) {
  const [open, setOpen] = useState(defaultOpen || depth < 2);

  const isObj = data !== null && typeof data === "object" && !Array.isArray(data);
  const isArr = Array.isArray(data);
  const isPrim = !isObj && !isArr;

  if (isPrim) {
    return (
      <span>
        {name !== undefined && (
          <>
            <span className="json-key">"{name}"</span>
            <span className="json-punct">: </span>
          </>
        )}
        <JsonValue value={data} />
      </span>
    );
  }

  const entries: [string | number, unknown][] = isArr
    ? (data as unknown[]).map((v, i) => [i, v])
    : Object.entries(data as Record<string, unknown>);
  const openBracket = isArr ? "[" : "{";
  const closeBracket = isArr ? "]" : "}";

  return (
    <div style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px", lineHeight: 1.55 }}>
      <span>
        <button
          onClick={() => setOpen((o) => !o)}
          style={{
            position: "relative",
            color: "var(--fg-tertiary)",
            width: 14,
            textAlign: "center",
            userSelect: "none",
            fontFamily: "inherit",
          }}
        >
          {open ? "▾" : "▸"}
        </button>
        {name !== undefined && (
          <>
            <span className="json-key">"{name}"</span>
            <span className="json-punct">: </span>
          </>
        )}
        <span className="json-punct">{openBracket}</span>
        {!open && (
          <span style={{ color: "var(--fg-muted)", fontStyle: "italic", fontSize: 10 }}>
            {" "}{entries.length} {isArr ? "items" : "keys"}{" "}
          </span>
        )}
        {!open && <span className="json-punct">{closeBracket}</span>}
      </span>
      {open && (
        <div>
          {entries.map(([k, v], i) => (
            <div key={k} style={{ paddingLeft: 16, position: "relative" }}>
              <JsonTree data={v} name={k} depth={depth + 1} defaultOpen={depth < 1} />
              {i < entries.length - 1 && <span className="json-punct">,</span>}
            </div>
          ))}
          <div><span className="json-punct">{closeBracket}</span></div>
        </div>
      )}
    </div>
  );
}

function JsonValue({ value }: { value: unknown }) {
  if (value === null || value === undefined)
    return <span style={{ color: "var(--fg-muted)", fontStyle: "italic" }}>null</span>;
  if (typeof value === "boolean")
    return <span style={{ color: "var(--sig-violet)" }}>{String(value)}</span>;
  if (typeof value === "number")
    return <span style={{ color: "var(--sig-amber)" }}>{value}</span>;
  if (typeof value === "string") {
    const display = value.length > 200 ? value.slice(0, 200) + "…" : value;
    return <span style={{ color: "var(--sig-emerald)" }}>"{display}"</span>;
  }
  return <span style={{ color: "var(--fg-secondary)" }}>{String(value)}</span>;
}

// Inline key color helper used by other components
export const jsonStyles = {
  key: { color: "oklch(0.78 0.1 280)" } as React.CSSProperties,
  string: { color: "var(--sig-emerald)" } as React.CSSProperties,
  number: { color: "var(--sig-amber)" } as React.CSSProperties,
  bool: { color: "var(--sig-violet)" } as React.CSSProperties,
  null: { color: "var(--fg-muted)", fontStyle: "italic" } as React.CSSProperties,
  punct: { color: "var(--fg-muted)" } as React.CSSProperties,
};
