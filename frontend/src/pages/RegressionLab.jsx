import { useState } from "react";
import { FlaskConical, Play, RotateCcw } from "lucide-react";
import {
  fetchLinearRegression,
  fetchLassoRegression,
  fetchRidgeRegression,
} from "../services/api";
import DataTable from "../components/DataTable";
import LoadingState from "../components/LoadingState";
import { formatCell, formatInt, formatPercent } from "../utils/format";

const MODELS = [
  {
    id: "linear",
    label: "Linear Regression",
    fetch: fetchLinearRegression,
    blurb: "Pipeline(PolynomialFeatures(degree=2) → LinearRegression())",
    loadingMessage: "Training model… Running 4-fold cross-validation… Calculating predictions…",
  },
  {
    id: "lasso",
    label: "Lasso Regression",
    fetch: fetchLassoRegression,
    blurb: "GridSearchCV over 20 alpha values, 4-fold CV, then Lasso(max_iter=10000)",
    loadingMessage: "Running GridSearchCV over 20 alpha values (4-fold CV)… this can take a few seconds…",
  },
  {
    id: "ridge",
    label: "Ridge Regression",
    fetch: fetchRidgeRegression,
    blurb: "GridSearchCV over 20 alpha values, 4-fold CV, then Ridge()",
    loadingMessage: "Running GridSearchCV over 20 alpha values (4-fold CV)… this can take a few seconds…",
  },
];

// --- small presentational helpers, styled to match StageDetails.jsx --------

function SectionLabel({ children }) {
  return <p className="mb-2 text-sm font-medium text-ink">{children}</p>;
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

const ACCENT_CLASSES = {
  amber: "text-amber",
  teal: "text-teal",
};

function MetricCard({ label, value, accent = "amber" }) {
  return (
    <div className="rounded border border-border bg-surface p-4">
      <p className={`tabular font-mono text-2xl ${ACCENT_CLASSES[accent]}`}>{value}</p>
      <p className="mt-1 text-xs text-muted">{label}</p>
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

function WhatHappened({ children }) {
  return (
    <details open className="group rounded border border-border">
      <summary className="cursor-pointer list-none px-4 py-2.5 text-sm font-medium text-ink marker:content-none">
        Model configuration
      </summary>
      <div className="border-t border-border px-4 py-3 text-sm text-muted">{children}</div>
    </details>
  );
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
          <div className="h-full bg-amber" style={{ width: `${(trainRows / total) * 100}%` }} />
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

/** Lightweight actual-vs-predicted scatter, plain inline SVG (no chart library). */
function ActualVsPredicted({ actual, predicted }) {
  const n = Math.min(actual.length, predicted.length, 150);
  const points = Array.from({ length: n }, (_, i) => [actual[i], predicted[i]]);
  const all = points.flat();
  const min = Math.min(...all);
  const max = Math.max(...all);
  const pad = (max - min) * 0.05 || 0.1;
  const lo = min - pad;
  const hi = max + pad;
  const size = 320;
  const scale = (v) => ((v - lo) / (hi - lo)) * size;

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-full w-full max-w-md">
      <rect x="0" y="0" width={size} height={size} fill="none" />
      <line
        x1={scale(lo)}
        y1={size - scale(lo)}
        x2={scale(hi)}
        y2={size - scale(hi)}
        stroke="#5C6169"
        strokeDasharray="4 4"
        strokeWidth="1"
      />
      {points.map(([a, p], i) => (
        <circle key={i} cx={scale(a)} cy={size - scale(p)} r="2.5" fill="#4FB4A8" fillOpacity="0.7" />
      ))}
    </svg>
  );
}

function ModelPicker({ selectedId, onSelect }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {MODELS.map((m) => {
        const isSelected = m.id === selectedId;
        return (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            className={`rounded border p-4 text-left transition-colors ${
              isSelected
                ? "border-amber/60 bg-raised"
                : "border-border bg-surface hover:border-amber/40"
            }`}
          >
            <p className="font-medium text-ink">{m.label}</p>
            <p className="mt-1 font-mono text-xs text-muted">{m.blurb}</p>
          </button>
        );
      })}
    </div>
  );
}

function ModelResults({ model, data }) {
  const { model: modelInfo, data: dataInfo, metrics, model_details: details } = data;
  const isLinear = model.id === "linear";

  const predictionRows = data.actual_values.slice(0, 20).map((actual, i) => ({
    "#": i + 1,
    Actual: actual,
    Predicted: data.predictions[i],
    Residual: data.residuals[i],
  }));

  return (
    <div className="space-y-8">
      {/* 1. Processed Data Used */}
      <div>
        <SectionLabel>1. Processed data used</SectionLabel>
        <StatGrid
          items={[
            ["Target", dataInfo.target_column],
            ["Feature count", formatInt(dataInfo.feature_count)],
            ["Training rows", formatInt(dataInfo.train_rows)],
            ["Test rows", formatInt(dataInfo.test_rows)],
            ["Random state", formatInt(dataInfo.random_state)],
          ]}
        />
        <div className="mt-4">
          <BarSplit trainRows={dataInfo.train_rows} testRows={dataInfo.test_rows} />
        </div>
      </div>

      {/* 2. Model Configuration */}
      <div>
        <SectionLabel>2. Model configuration</SectionLabel>
        <WhatHappened>
          <ul className="list-inside list-disc space-y-1">
            <li>
              Algorithm: <code className="font-mono text-ink/80">{modelInfo.name}</code>
            </li>
            <li>Polynomial degree: {modelInfo.polynomial_degree}</li>
            {modelInfo.alpha_search && (
              <>
                <li>
                  Alpha search:{" "}
                  <code className="font-mono text-ink/80">
                    {modelInfo.alpha_search.min.toExponential(0)} → {modelInfo.alpha_search.max.toExponential(0)} (
                    {modelInfo.alpha_search.count} values)
                  </code>
                </li>
                <li>Cross-validation: {modelInfo.cv_folds}-fold</li>
              </>
            )}
          </ul>
        </WhatHappened>
      </div>

      {/* 4. Predictions */}
      <div>
        <SectionLabel>
          3. Predictions <span className="text-muted">(first 20 of {formatInt(dataInfo.test_rows)} test rows)</span>
        </SectionLabel>
        <DataTable rows={predictionRows} pageSize={20} />
      </div>

      {/* 5. Evaluation Metrics */}
      <div>
        <SectionLabel>4. Evaluation metrics</SectionLabel>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard label="R²" value={formatCell(metrics.r2)} accent="amber" />
          <MetricCard label="MSE" value={formatCell(metrics.mse)} accent="amber" />
          {isLinear ? (
            <MetricCard label="4-Fold CV Mean R²" value={formatCell(metrics.cv_mean_r2)} accent="teal" />
          ) : (
            <>
              <MetricCard label="Best Alpha" value={formatCell(metrics.best_alpha)} accent="teal" />
              <MetricCard label="Best CV R²" value={formatCell(metrics.best_cv_r2)} accent="teal" />
            </>
          )}
        </div>
      </div>

      {/* 6. Model Details */}
      <div>
        <SectionLabel>5. Model details</SectionLabel>
        <StatGrid
          items={[
            ["Coefficient count", formatInt(details.coefficient_count)],
            ["Intercept", formatCell(details.intercept)],
            ["Polynomial degree", details.polynomial_degree],
          ]}
        />
        {details.best_params && (
          <div className="mt-3">
            <p className="mb-2 text-xs text-muted">Best parameters (GridSearchCV)</p>
            <ChipList
              items={Object.entries(details.best_params).map(([k, v]) => `${k}=${formatCell(v)}`)}
            />
          </div>
        )}
      </div>

      {/* Optional visualization */}
      <div>
        <SectionLabel>Actual vs predicted (Log_SalePrice)</SectionLabel>
        <div className="flex justify-center rounded border border-border bg-surface p-4">
          <ActualVsPredicted actual={data.actual_values} predicted={data.predictions} />
        </div>
        <p className="mt-2 text-center text-xs text-muted">
          Dashed line = perfect prediction. Each dot is one test-set house (first {Math.min(150, data.actual_values.length)} shown).
        </p>
      </div>
    </div>
  );
}

export default function RegressionLab() {
  const [selectedId, setSelectedId] = useState("linear");
  const [results, setResults] = useState({});
  const [loadingMap, setLoadingMap] = useState({});
  const [errorMap, setErrorMap] = useState({});

  const model = MODELS.find((m) => m.id === selectedId);
  const isLoading = !!loadingMap[selectedId];
  const error = errorMap[selectedId];
  const result = results[selectedId];

  function runModel(id) {
    const target = MODELS.find((m) => m.id === id);
    setLoadingMap((prev) => ({ ...prev, [id]: true }));
    setErrorMap((prev) => ({ ...prev, [id]: null }));
    target
      .fetch()
      .then((data) => setResults((prev) => ({ ...prev, [id]: data })))
      .catch((err) => setErrorMap((prev) => ({ ...prev, [id]: err.message })))
      .finally(() => setLoadingMap((prev) => ({ ...prev, [id]: false })));
  }

  function handleSelect(id) {
    setSelectedId(id);
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <div className="mb-8">
        <p className="font-mono text-sm text-teal">GET /regression/{selectedId}</p>
        <h1 className="mt-2 text-3xl font-semibold text-ink">Regression Lab</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Choose a model below, then run it against the processed dataset from the Dataset Preprocessing
          pipeline. Each run fits a real scikit-learn model on the backend — nothing here is precomputed.
        </p>
      </div>

      <div className="mb-4 flex items-center gap-2">
        <FlaskConical className="h-4 w-4 text-amber" strokeWidth={1.75} />
        <p className="text-sm font-medium text-ink">Choose a model</p>
      </div>
      <ModelPicker selectedId={selectedId} onSelect={handleSelect} />

      <div className="mt-8">
        {isLoading && <LoadingState variant="loading" message={model.loadingMessage} />}

        {!isLoading && error && (
          <LoadingState variant="error" message={error} onRetry={() => runModel(selectedId)} />
        )}

        {!isLoading && !error && !result && (
          <div className="flex flex-col items-center justify-center gap-4 rounded border border-dashed border-border bg-surface/50 px-8 py-16 text-center">
            <p className="text-sm text-muted">
              {model.label} hasn't been run yet. Training starts only when you press the button below.
            </p>
            <button
              onClick={() => runModel(selectedId)}
              className="flex items-center gap-2 rounded border border-amber/60 bg-raised px-5 py-2.5 text-sm font-medium text-amber transition-colors hover:bg-amber/10"
            >
              <Play className="h-4 w-4" strokeWidth={1.75} />
              Run {model.label}
            </button>
          </div>
        )}

        {!isLoading && !error && result && (
          <div>
            <div className="mb-6 flex items-center justify-end">
              <button
                onClick={() => runModel(selectedId)}
                className="flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs text-muted transition-colors hover:border-amber/60 hover:text-amber"
              >
                <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.75} />
                Run again
              </button>
            </div>
            <ModelResults model={model} data={result} />
          </div>
        )}
      </div>
    </div>
  );
}
