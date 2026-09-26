/**
 * Format a raw cell value from a preview row for display.
 * Never invents a value — null/undefined/NaN are all rendered as a plain dash.
 */
export function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Number.isNaN(value)) return "—";
    if (Number.isInteger(value)) return value.toLocaleString("en-US");
    return value.toLocaleString("en-US", { maximumFractionDigits: 4 });
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

export function formatInt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString("en-US");
}

export function formatPercent(part, total, digits = 0) {
  if (!total) return "—";
  return `${((part / total) * 100).toFixed(digits)}%`;
}
