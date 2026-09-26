import DataTable from "./DataTable";
import { formatCell, formatInt, formatPercent } from "../utils/format";

const WHY_COPY = {
  "Raw Dataset":
    "Every pipeline starts from the untouched CSV, so later stages can be compared back against this baseline.",
  "Missing Value Handling":
    "Most models can't accept NaN values directly. Filling them in a controlled way keeps every row usable.",
  "Feature Removal":
    "Columns with little predictive signal or heavy redundancy are dropped early, before they can distort scaling or encoding.",
  "Log Target Transformation":
    "House prices are right-skewed. Taking log1p(SalePrice) makes the target closer to normally distributed, which most regression models assume.",
  "Encoding + Scaling":
    "Regression models need numeric input. Scaling puts numeric features on comparable ranges; one-hot encoding turns categories into numeric columns.",
  "Redundant Feature Removal":
    "Features that are highly correlated with others add noise without adding information, so they're dropped after encoding reveals the correlation.",
  "Outlier Detection":
    "Extreme target values can dominate a regression's loss function. The IQR method flags rows whose Log_SalePrice sits far outside the typical range.",
  "Outlier Removal":
    "Removing the flagged rows keeps the model from over-fitting to a handful of extreme prices.",
  "Final Dataset":
    "This is the fully-processed feature matrix and target that the model will actually be trained and evaluated on.",
  "Train/Test Split":
    "Holding out a test set lets the model be evaluated on data it never saw during training.",
};

function Why({ stageName }) {
  const text = WHY_COPY[stageName];
  if (!text) return null;
  return (
    <details className="group rounded border border-border">
      <summary className="cursor-pointer list-none px-4 py-2.5 text-sm text-muted marker:content-none group-open:text-ink">
        Why is this step performed?
      </summary>
      <p className="border-t border-border px-4 py-3 text-sm leading-relaxed text-muted">{text}</p>
    </details>
  );
}

function WhatHappened({ children }) {
  return (
    <details open className="group rounded border border-border">
      <summary className="cursor-pointer list-none px-4 py-2.5 text-sm font-medium text-ink marker:content-none">
        What happened?
      </summary>
      <div className="border-t border-border px-4 py-3 text-sm text-muted">{children}</div>
    </details>
  );
}

function StatGrid({ items }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {items.map(([label, value]) => (
        <div key={label} className="rounded border border-border bg-raised px-3 py-2.5">
          <p className="tabular font-mono text-lg text-ink">{value}</p>
          <p className="mt-0.5 text-xs text-muted">{label}</p>
        </div>
      ))}
    </div>
  );
}

function ChipList({ items, emptyLabel = "None" }) {
  if (!items || items.length === 0) {
    return <p className="text-sm text-muted">{emptyLabel}</p>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <span
          key={item}
          className="rounded border border-border bg-raised px-2 py-0.5 font-mono text-xs text-ink/80"
        >
          {item}
        </span>
      ))}
    </div>
  );
}

function SectionLabel({ children }) {
  return <p className="mb-2 text-sm font-medium text-ink">{children}</p>;
}

function BarSplit({ trainRows, testRows }) {
  const total = trainRows + testRows;
  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 flex items-center justify-between text-xs text-muted">
          <span>Train</span>
          <span className="tabular">
            {formatInt(trainRows)} rows · {formatPercent(trainRows, total)}
          </span>
        </div>
        <div className="h-2.5 overflow-hidden rounded bg-raised">
          <div
            className="h-full bg-amber"
            style={{ width: `${(trainRows / total) * 100}%` }}
          />
        </div>
      </div>
      <div>
        <div className="mb-1 flex items-center justify-between text-xs text-muted">
          <span>Test</span>
          <span className="tabular">
            {formatInt(testRows)} rows · {formatPercent(testRows, total)}
          </span>
        </div>
        <div className="h-2.5 overflow-hidden rounded bg-raised">
          <div className="h-full bg-teal" style={{ width: `${(testRows / total) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

/**
 * Renders the right-hand detail panel for whichever stage is selected.
 * Every number, list and preview row below is read straight from that
 * stage's slice of the /preprocessing response — nothing is computed from
 * assumptions about what the backend "should" have done.
 */
export default function StageDetails({ stage, previousStage }) {
  if (!stage) return null;

  switch (stage.name) {
    case "Raw Dataset":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Rows", formatInt(stage.rows)],
              ["Columns", formatInt(stage.columns)],
              ["Total missing values", formatInt(stage.total_missing_values)],
            ]}
          />
          <WhatHappened>
            The dataset was loaded from <code className="font-mono text-ink/80">train.csv</code> with no
            modifications applied yet.
          </WhatHappened>
          <div>
            <SectionLabel>Column names</SectionLabel>
            <ChipList items={stage.column_names} />
          </div>
          <div>
            <SectionLabel>First 10 rows</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Missing Value Handling": {
      const affected = Object.entries(stage.before_missing_values_by_column || {});
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Missing before", formatInt(stage.before_missing_values_total)],
              ["Missing after", formatInt(stage.after_missing_values_total)],
              ["Columns affected", formatInt(affected.length)],
            ]}
          />
          <WhatHappened>
            <ul className="list-inside list-disc space-y-1">
              <li>{stage.strategy?.LotFrontage} for LotFrontage</li>
              <li>Categorical columns: {stage.strategy?.categorical_columns}</li>
              <li>Numeric columns: {stage.strategy?.numeric_columns}</li>
            </ul>
          </WhatHappened>
          <div>
            <SectionLabel>Affected columns (before)</SectionLabel>
            {affected.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {affected.map(([col, count]) => (
                  <span
                    key={col}
                    className="rounded border border-border bg-raised px-2 py-0.5 font-mono text-xs text-ink/80"
                  >
                    {col} <span className="text-amber">{count}</span>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted">No missing values in the raw data.</p>
            )}
          </div>
          <div>
            <SectionLabel>Preview after handling</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );
    }

    case "Feature Removal":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Columns before", formatInt(previousStage?.columns)],
              ["Columns removed", formatInt(stage.columns_removed_count)],
              ["Columns after", formatInt(stage.columns)],
            ]}
          />
          <WhatHappened>
            {stage.columns_removed_count} column(s) identified as low-signal or redundant during EDA were
            dropped from the dataset.
          </WhatHappened>
          <div>
            <SectionLabel>Columns removed</SectionLabel>
            <ChipList items={stage.columns_removed} />
          </div>
          <div>
            <SectionLabel>Preview after removal</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Log Target Transformation": {
      const beforeAfterRows = (stage.preview || []).map((row) => ({
        SalePrice: row.SalePrice,
        Log_SalePrice: row.Log_SalePrice,
      }));
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded border border-border p-3">
              <p className="mb-2 font-mono text-xs text-muted">SalePrice</p>
              <StatGrid
                items={[
                  ["Mean", formatCell(stage.sale_price_stats?.mean)],
                  ["Min", formatCell(stage.sale_price_stats?.min)],
                  ["Max", formatCell(stage.sale_price_stats?.max)],
                ]}
              />
            </div>
            <div className="rounded border border-border p-3">
              <p className="mb-2 font-mono text-xs text-muted">Log_SalePrice</p>
              <StatGrid
                items={[
                  ["Mean", formatCell(stage.log_sale_price_stats?.mean)],
                  ["Min", formatCell(stage.log_sale_price_stats?.min)],
                  ["Max", formatCell(stage.log_sale_price_stats?.max)],
                ]}
              />
            </div>
          </div>
          <WhatHappened>
            <code className="font-mono text-ink/80">{stage.transformation}</code> was added as a new
            column.
          </WhatHappened>
          <div>
            <SectionLabel>SalePrice vs Log_SalePrice</SectionLabel>
            <DataTable rows={beforeAfterRows} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );
    }

    case "Encoding + Scaling":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Numeric features scaled", formatInt(stage.numeric_columns_scaled_count)],
              ["Categorical features encoded", formatInt(stage.categorical_columns_encoded_count)],
              ["Resulting columns", formatInt(stage.columns)],
            ]}
          />
          <WhatHappened>
            <p>
              Scaling: <code className="font-mono text-ink/80">{stage.scaling_method}</code>
            </p>
            <p className="mt-1">
              Encoding: <code className="font-mono text-ink/80">{stage.encoding_method}</code>
            </p>
          </WhatHappened>
          <div>
            <SectionLabel>Numeric columns scaled</SectionLabel>
            <ChipList items={stage.numeric_columns_scaled} />
          </div>
          <div>
            <SectionLabel>Categorical columns encoded</SectionLabel>
            <ChipList items={stage.categorical_columns_encoded} />
          </div>
          <div>
            <SectionLabel>Sample transformed rows</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Redundant Feature Removal":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Columns before", formatInt(previousStage?.columns)],
              ["Columns removed", formatInt(stage.columns_removed.length)],
              ["Columns after", formatInt(stage.columns)],
            ]}
          />
          <WhatHappened>{stage.reason}</WhatHappened>
          <div>
            <SectionLabel>Columns removed</SectionLabel>
            <ChipList items={stage.columns_removed} />
          </div>
          <div>
            <SectionLabel>Preview after removal</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Outlier Detection":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Method", stage.method],
              ["Outliers detected", formatInt(stage.outliers_detected)],
              ["IQR", formatCell(stage.iqr)],
              ["Q1", formatCell(stage.q1)],
              ["Q3", formatCell(stage.q3)],
              ["Lower / upper bound", `${formatCell(stage.lower_bound)} / ${formatCell(stage.upper_bound)}`],
            ]}
          />
          <WhatHappened>
            Rows whose Log_SalePrice falls outside [{formatCell(stage.lower_bound)},{" "}
            {formatCell(stage.upper_bound)}] were flagged as outliers.
          </WhatHappened>
          <div>
            <SectionLabel>Detected outlier rows</SectionLabel>
            <DataTable rows={stage.preview} emptyMessage="No outliers detected." />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Outlier Removal":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Rows before", formatInt(stage.rows_before)],
              ["Rows removed", formatInt(stage.rows_removed)],
              ["Rows after", formatInt(stage.rows_after)],
            ]}
          />
          <WhatHappened>
            The {stage.rows_removed} row(s) flagged in the previous step were dropped from the dataset.
          </WhatHappened>
          <div>
            <SectionLabel>Preview after removal</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Final Dataset":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Rows", formatInt(stage.rows)],
              ["Feature columns", formatInt(stage.feature_columns_count)],
              ["Target column", stage.target_column],
            ]}
          />
          <WhatHappened>
            The fully-processed feature matrix (<code className="font-mono text-ink/80">x_data</code>) and
            target (<code className="font-mono text-ink/80">y_data</code>) are ready for modeling.
          </WhatHappened>
          <div>
            <SectionLabel>Feature columns</SectionLabel>
            <ChipList items={stage.feature_columns} />
          </div>
          <div>
            <SectionLabel>Preview</SectionLabel>
            <DataTable rows={stage.preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    case "Train/Test Split":
      return (
        <div className="space-y-4">
          <StatGrid
            items={[
              ["Random state", formatInt(stage.random_state)],
              ["Test size", stage.test_size],
              ["Feature columns", formatInt(stage.feature_columns_count)],
            ]}
          />
          <WhatHappened>
            The final dataset was split with{" "}
            <code className="font-mono text-ink/80">
              train_test_split(random_state={stage.random_state})
            </code>
            .
          </WhatHappened>
          <BarSplit trainRows={stage.train_rows} testRows={stage.test_rows} />
          <div>
            <SectionLabel>Train preview</SectionLabel>
            <DataTable rows={stage.train_preview} />
          </div>
          <div>
            <SectionLabel>Test preview</SectionLabel>
            <DataTable rows={stage.test_preview} />
          </div>
          <Why stageName={stage.name} />
        </div>
      );

    default:
      return <p className="text-sm text-muted">No details available for this stage.</p>;
  }
}
