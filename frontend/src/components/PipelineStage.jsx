import { Check } from "lucide-react";

/**
 * Builds the one-line summary shown under each stage name. Every value read
 * here comes directly from that stage's own API payload — nothing computed
 * beyond simple arithmetic on numbers the backend already returned.
 */
function summarize(stage) {
  switch (stage.name) {
    case "Raw Dataset":
      return `${stage.rows.toLocaleString()} rows · ${stage.columns} columns`;
    case "Missing Value Handling":
      return `${stage.before_missing_values_total.toLocaleString()} → ${stage.after_missing_values_total} missing values`;
    case "Feature Removal":
      return `${stage.columns_removed_count} columns removed`;
    case "Log Target Transformation":
      return "Log_SalePrice = log1p(SalePrice)";
    case "Encoding + Scaling":
      return `${stage.columns} columns after transform`;
    case "Redundant Feature Removal":
      return `${stage.columns_removed.length} redundant columns removed`;
    case "Outlier Detection":
      return `${stage.outliers_detected} outliers detected`;
    case "Outlier Removal":
      return `${stage.rows_removed} rows removed`;
    case "Final Dataset":
      return `${stage.feature_columns_count} features · ${stage.rows.toLocaleString()} rows`;
    case "Train/Test Split":
      return `${stage.train_rows.toLocaleString()} train · ${stage.test_rows.toLocaleString()} test`;
    default:
      return null;
  }
}

export default function PipelineStage({ stage, index, isSelected, isLast, onSelect }) {
  return (
    <div className="relative flex gap-4 pb-8 last:pb-0">
      {!isLast && (
        <span
          aria-hidden="true"
          className="absolute left-[15px] top-8 h-[calc(100%-1rem)] w-px bg-border"
        />
      )}

      <span
        className={`z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border font-mono text-xs ${
          isSelected
            ? "border-amber bg-amber/10 text-amber"
            : "border-border bg-surface text-muted"
        }`}
      >
        <Check className="h-3.5 w-3.5" strokeWidth={2} />
      </span>

      <button
        onClick={() => onSelect(index)}
        className={`flex-1 rounded border px-4 py-3 text-left transition-colors ${
          isSelected
            ? "border-amber/50 bg-amber/5"
            : "border-border bg-surface hover:border-faint"
        }`}
      >
        <p className="font-mono text-xs text-muted">STEP {index + 1}</p>
        <p className="mt-0.5 font-medium text-ink">{stage.name}</p>
        {summarize(stage) && (
          <p className="tabular mt-1 text-sm text-muted">{summarize(stage)}</p>
        )}
      </button>
    </div>
  );
}
