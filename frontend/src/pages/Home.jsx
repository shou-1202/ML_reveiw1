import { Link } from "react-router-dom";
import { ArrowUpRight, Database, SlidersHorizontal } from "lucide-react";

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-20">
      <div className="max-w-2xl">
        <p className="font-mono text-sm text-teal">House Prices · Ames, Iowa dataset</p>
        <h1 className="mt-3 text-4xl font-semibold leading-tight text-ink sm:text-5xl">
          Housing Price Regression Lab
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-muted">
          Explore how raw housing data is transformed into machine-learning-ready data.
        </p>
      </div>

      <div className="mt-16 grid gap-5 sm:grid-cols-2">
        <Link
          to="/preprocessing"
          className="group relative flex flex-col justify-between rounded border border-border bg-surface p-6 transition-colors hover:border-amber/50"
        >
          <div>
            <Database className="h-5 w-5 text-amber" strokeWidth={1.5} />
            <h2 className="mt-4 text-lg font-medium text-ink">Dataset Preprocessing</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Explore every stage of the preprocessing pipeline.
            </p>
          </div>
          <span className="mt-6 flex items-center gap-1 text-sm text-amber">
            Explore Preprocessing
            <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
        </Link>

        <Link
          to="/regression"
          className="group relative flex flex-col justify-between rounded border border-border bg-surface p-6 transition-colors hover:border-teal/50"
        >
          <div>
            <SlidersHorizontal className="h-5 w-5 text-teal" strokeWidth={1.5} />
            <h2 className="mt-4 text-lg font-medium text-ink">Regression Lab</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Run Linear, Lasso and Ridge regression models on the processed dataset.
            </p>
          </div>
          <span className="mt-6 flex items-center gap-1 text-sm text-teal">
            Explore Regression Lab
            <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
        </Link>
      </div>
    </div>
  );
}
