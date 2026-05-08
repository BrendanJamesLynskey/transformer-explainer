/**
 * Section funnel as a table with inline bars. Per row:
 *   page views → interacts → completes,
 * showing the count plus the % retained from the previous step.
 */
type Row = {
  sectionSlug: string;
  views: number;
  interacts: number;
  completes: number;
};

function pct(num: number, denom: number): string {
  if (denom === 0) return "—";
  return `${Math.round((num / denom) * 100)}%`;
}

export function SectionFunnelTable({ rows }: { rows: Row[] }): JSX.Element {
  if (rows.length === 0) {
    return (
      <p className="font-mono text-xs text-neutral-500">
        no section events yet
      </p>
    );
  }
  const maxViews = Math.max(...rows.map((r) => r.views), 1);

  return (
    <table className="w-full text-sm">
      <thead className="text-left">
        <tr className="border-b border-neutral-200 text-xs uppercase tracking-wide text-neutral-500 dark:border-neutral-800">
          <th className="py-2 pr-4">Section</th>
          <th className="py-2 pr-4">Views</th>
          <th className="py-2 pr-4">Interacted</th>
          <th className="py-2 pr-4">Completed</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr
            key={r.sectionSlug}
            className="border-b border-neutral-100 last:border-b-0 dark:border-neutral-900"
          >
            <td className="py-2 pr-4 font-mono text-xs">{r.sectionSlug}</td>
            <td className="py-2 pr-4">
              <Bar value={r.views} max={maxViews} />
              <span className="ml-2 tabular-nums">{r.views}</span>
            </td>
            <td className="py-2 pr-4 tabular-nums">
              {r.interacts}{" "}
              <span className="text-xs text-neutral-500">
                ({pct(r.interacts, r.views)})
              </span>
            </td>
            <td className="py-2 pr-4 tabular-nums">
              {r.completes}{" "}
              <span className="text-xs text-neutral-500">
                ({pct(r.completes, r.views)})
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Bar({ value, max }: { value: number; max: number }): JSX.Element {
  const w = max === 0 ? 0 : (value / max) * 100;
  return (
    <span className="inline-block h-2 w-24 rounded bg-neutral-100 align-middle dark:bg-neutral-800">
      <span
        className="block h-full rounded bg-accent"
        style={{ width: `${w}%` }}
      />
    </span>
  );
}
