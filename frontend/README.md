# Housing Price Regression Lab — Frontend

React + Vite + Tailwind frontend for the Dataset Preprocessing step. Connects
to the FastAPI backend's `GET /preprocessing` endpoint.

## Setup

```
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173`.

## Backend connection

The API base URL is read from `VITE_API_URL` (see `.env`, defaults to
`http://localhost:8000`). Make sure the FastAPI backend is running first:

```
cd ../backend
uvicorn main:app --reload
```

## Pages

- `/` — Home, with links to Dataset Preprocessing (live) and Regression Lab
  (disabled, coming soon).
- `/preprocessing` — calls `GET /preprocessing` on load, shows a loading
  state, an error state if the backend is unreachable, and — once loaded —
  the dataset overview stats plus an interactive stage-by-stage pipeline
  timeline. Clicking a stage opens its details (stats, columns, "What
  happened?" / "Why is this step performed?", and a preview data table) in
  the panel on the right.

## Notes

- All numbers, column names, and preview rows shown in the UI come directly
  from the `/preprocessing` response — nothing is hardcoded or invented.
- No regression/model logic is implemented yet; the Regression Lab nav item
  and Home card are intentionally inert placeholders.
