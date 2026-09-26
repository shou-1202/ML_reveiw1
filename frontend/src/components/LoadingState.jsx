import { AlertTriangle, Loader2 } from "lucide-react";

/**
 * Shared loading / error state for any view waiting on the backend.
 * Errors are rendered from the friendly message produced by services/api.js —
 * never a raw Axios/network error object.
 */
export default function LoadingState({ variant = "loading", message, onRetry }) {
  if (variant === "error") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded border border-danger/30 bg-danger/5 px-8 py-16 text-center">
        <AlertTriangle className="h-6 w-6 text-danger" strokeWidth={1.5} />
        <div>
          <p className="font-medium text-ink">Couldn't reach the backend</p>
          <p className="mt-1 max-w-sm text-sm text-muted">{message}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="rounded border border-border px-4 py-2 text-sm text-ink transition-colors hover:border-amber/60 hover:text-amber"
          >
            Try again
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-24 text-center">
      <Loader2 className="h-5 w-5 animate-spin text-amber" strokeWidth={1.5} />
      <p className="text-sm text-muted">{message || "Running the preprocessing pipeline…"}</p>
    </div>
  );
}
