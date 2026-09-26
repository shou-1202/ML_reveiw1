import { useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { formatCell } from "../utils/format";

/**
 * Generic preview-row table. Renders whatever rows/columns it's given —
 * it never knows or assumes anything about preprocessing semantics.
 */
export default function DataTable({ rows, pageSize = 10, emptyMessage = "No rows to display." }) {
  const [page, setPage] = useState(0);

  const columns = useMemo(() => {
    if (!rows || rows.length === 0) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  const pageCount = Math.max(1, Math.ceil((rows?.length || 0) / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const visibleRows = useMemo(
    () => (rows || []).slice(currentPage * pageSize, currentPage * pageSize + pageSize),
    [rows, currentPage, pageSize]
  );

  if (!rows || rows.length === 0) {
    return <p className="py-6 text-sm text-muted">{emptyMessage}</p>;
  }

  return (
    <div className="rounded border border-border">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border bg-raised">
              {columns.map((col) => (
                <th
                  key={col}
                  scope="col"
                  className="whitespace-nowrap px-3 py-2 text-left font-mono text-xs font-medium text-muted"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, i) => (
              <tr
                key={currentPage * pageSize + i}
                className="border-b border-border/60 last:border-b-0 hover:bg-raised/60"
              >
                {columns.map((col) => (
                  <td key={col} className="tabular whitespace-nowrap px-3 py-2 text-ink/90">
                    {formatCell(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {pageCount > 1 && (
        <div className="flex items-center justify-between border-t border-border px-3 py-2 text-xs text-muted">
          <span>
            Rows {currentPage * pageSize + 1}–{Math.min((currentPage + 1) * pageSize, rows.length)} of{" "}
            {rows.length}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={currentPage === 0}
              className="rounded p-1 hover:bg-raised disabled:opacity-30"
              aria-label="Previous page"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="font-mono">
              {currentPage + 1} / {pageCount}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
              disabled={currentPage === pageCount - 1}
              className="rounded p-1 hover:bg-raised disabled:opacity-30"
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
