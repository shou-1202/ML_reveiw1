import { useEffect, useState } from "react";
import { fetchPreprocessing } from "../services/api";
import DatasetOverview from "../components/DatasetOverview";
import PipelineStepper from "../components/PipelineStepper";
import StageDetails from "../components/StageDetails";
import LoadingState from "../components/LoadingState";

export default function Preprocessing() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);

  function load() {
    setLoading(true);
    setError(null);
    fetchPreprocessing()
      .then((result) => {
        setData(result);
        setSelectedIndex(0);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  if (loading) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-12">
        <LoadingState variant="loading" message="Running the preprocessing pipeline…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-12">
        <LoadingState variant="error" message={error} onRetry={load} />
      </div>
    );
  }

  const stages = data.stages;
  const finalStage = stages.find((s) => s.name === "Final Dataset");

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <div className="mb-8">
        <p className="font-mono text-sm text-teal">GET /preprocessing</p>
        <h1 className="mt-2 text-3xl font-semibold text-ink">Dataset Preprocessing</h1>
      </div>

      <DatasetOverview
        rows={data.dataset.rows}
        columns={data.dataset.columns}
        missingValues={stages[0]?.total_missing_values}
        features={finalStage?.feature_columns_count}
      />

      <div className="mt-10 grid gap-8 lg:grid-cols-[360px_1fr]">
        <div>
          <p className="mb-4 text-sm font-medium text-ink">Preprocessing pipeline</p>
          <PipelineStepper
            stages={stages}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
          />
        </div>

        <div>
          <p className="mb-4 text-sm font-medium text-ink">Selected stage details</p>
          <StageDetails
            stage={stages[selectedIndex]}
            previousStage={stages[selectedIndex - 1]}
          />
        </div>
      </div>
    </div>
  );
}
