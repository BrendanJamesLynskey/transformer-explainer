/**
 * Admin card: database health plus this server instance's fallback tally.
 *
 * Server Component. The health report is the same one `/api/health`
 * returns; the fallback rows come from `lib/db-fallback`, which counts every
 * time a page quietly served empty data because the database threw. A
 * healthy check with a non-empty tally means the database *was* failing
 * recently on this instance.
 */
import type { FallbackStat } from "@/lib/db-fallback";
import type { HealthReport } from "@/lib/health-shared";

export function HealthCard({
  report,
  fallbacks,
}: {
  report: HealthReport;
  fallbacks: FallbackStat[];
}): JSX.Element {
  const schema = report.schema;
  return (
    <div className="space-y-4">
      <p
        data-testid="health-status"
        className={
          report.healthy
            ? "font-mono text-sm text-emerald-700 dark:text-emerald-400"
            : "font-mono text-sm font-semibold text-red-700 dark:text-red-400"
        }
      >
        {report.healthy ? "● healthy" : "● UNHEALTHY"} — database {report.db}
        {report.dbError ? ` (${report.dbError})` : ""}
        {schema
          ? ` · schema ${schema.status} (${schema.applied}/${schema.expected} migrations, latest ${schema.latestExpected ?? "none"})`
          : ""}
      </p>

      <div>
        <h3 className="text-sm font-medium">
          Fallbacks served by this server instance
        </h3>
        {fallbacks.length === 0 ? (
          <p className="mt-1 font-mono text-xs text-neutral-500 dark:text-neutral-400">
            none since this instance started
          </p>
        ) : (
          <table
            data-testid="fallback-table"
            className="mt-2 w-full text-left text-sm"
          >
            <thead className="text-xs text-neutral-500 dark:text-neutral-400">
              <tr>
                <th className="py-1 font-normal">Key</th>
                <th className="py-1 text-right font-normal">Count</th>
                <th className="py-1 text-right font-normal">Last (UTC)</th>
              </tr>
            </thead>
            <tbody className="font-mono text-xs">
              {fallbacks.map((f) => (
                <tr
                  key={f.key}
                  className="border-t border-neutral-200 dark:border-neutral-800"
                >
                  <td className="py-1">{f.key}</td>
                  <td className="py-1 text-right tabular-nums">{f.count}</td>
                  <td className="py-1 text-right">
                    {f.lastAt.slice(0, 19).replace("T", " ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
