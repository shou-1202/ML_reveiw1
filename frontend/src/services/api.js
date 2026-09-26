import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_URL,
  timeout: 30000,
});

/**
 * A small, predictable error shape the UI can render directly, so components
 * never have to inspect raw Axios/network error internals.
 */
function toFriendlyError(error) {
  if (error.code === "ECONNABORTED") {
    return {
      message: "The request to the ML backend timed out. It may still be starting up.",
    };
  }
  if (error.response) {
    // Backend responded, but with an error status (e.g. FastAPI HTTPException)
    const detail = error.response.data?.detail;
    return {
      message:
        typeof detail === "string"
          ? detail
          : "The ML backend returned an error while processing the dataset.",
      status: error.response.status,
    };
  }
  // No response at all — the server is very likely not running
  return {
    message: "Unable to connect to the ML backend. Make sure the FastAPI server is running.",
  };
}

export async function fetchPreprocessing() {
  try {
    const response = await client.get("/preprocessing");
    return response.data;
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export async function fetchHealth() {
  try {
    const response = await client.get("/");
    return response.data;
  } catch (error) {
    throw toFriendlyError(error);
  }
}

// Regression endpoints. Lasso/Ridge run a GridSearchCV, so they're given a
// longer timeout than the default client — the requests are still made
// through the same Axios instance/base URL, just with a longer allowance.
const REGRESSION_TIMEOUT_MS = 60000;

export async function fetchLinearRegression() {
  try {
    const response = await client.get("/regression/linear", { timeout: REGRESSION_TIMEOUT_MS });
    return response.data;
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export async function fetchLassoRegression() {
  try {
    const response = await client.get("/regression/lasso", { timeout: REGRESSION_TIMEOUT_MS });
    return response.data;
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export async function fetchRidgeRegression() {
  try {
    const response = await client.get("/regression/ridge", { timeout: REGRESSION_TIMEOUT_MS });
    return response.data;
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export default client;
