import { formatInt } from "../utils/format";

function Stat({ label, value }) {
  return (
    <div className="flex-1 border-l border-border px-5 py-4 first:border-l-0 first:pl-0">
      <p className="font-mono text-2xl text-ink tabular">{value}</p>
      <p className="mt-1 text-xs text-muted">{label}</p>
    </div>
  );
}

/**
 * All four numbers come straight from the /preprocessing response —
 * dataset.rows / dataset.columns, the Raw Dataset stage's missing-value
 * total, and the Final Dataset stage's feature count.
 */
export default function DatasetOverview({ rows, columns, missingValues, features }) {
  return (
    <div className="flex flex-wrap rounded border border-border bg-surface">
      <Stat label="Rows" value={formatInt(rows)} />
      <Stat label="Columns" value={formatInt(columns)} />
      <Stat label="Missing values (raw)" value={formatInt(missingValues)} />
      <Stat label="Final features" value={formatInt(features)} />
    </div>
  );
}
