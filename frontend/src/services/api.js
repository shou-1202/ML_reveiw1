import axios from "axios";

// If VITE_API_URL is explicitly configured, use it.
// In dev mode, default to localhost:8000. In production (e.g. GitHub Pages),
// default to static mode unless an external backend is provided.
const configuredApiUrl = import.meta.env.VITE_API_URL;
const API_URL =
  configuredApiUrl !== undefined
    ? configuredApiUrl
    : import.meta.env.DEV
    ? "http://localhost:8000"
    : "";

const BASE_URL = import.meta.env.BASE_URL || "./";

const client = axios.create({
  baseURL: API_URL || undefined,
  timeout: 30000,
});

/**
 * Loads precomputed static JSON data from the public/data directory.
 * This guarantees the website works 100% on static hosts like GitHub Pages.
 */
async function fetchStatic(filename) {
  const normalizedBase = BASE_URL.endsWith("/") ? BASE_URL : `${BASE_URL}/`;
  const response = await fetch(`${normalizedBase}data/${filename}`);
  if (!response.ok) {
    throw new Error(`Failed to load static dataset: ${filename} (${response.status})`);
  }
  return await response.json();
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
    const detail = error.response.data?.detail;
    return {
      message:
        typeof detail === "string"
          ? detail
          : "The ML backend returned an error while processing the dataset.",
      status: error.response.status,
    };
  }
  return {
    message: error.message || "Unable to load data. Please try again.",
  };
}

export async function fetchPreprocessing() {
  if (API_URL) {
    try {
      const response = await client.get("/preprocessing");
      return response.data;
    } catch (error) {
      console.warn("API request failed, loading static precomputed data:", error.message);
    }
  }

  try {
    return await fetchStatic("preprocessing.json");
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export async function fetchHealth() {
  if (API_URL) {
    try {
      const response = await client.get("/");
      return response.data;
    } catch {
      // ignore
    }
  }
  return { status: "ok", mode: "static" };
}

const REGRESSION_TIMEOUT_MS = 60000;

export async function fetchLinearRegression() {
  if (API_URL) {
    try {
      const response = await client.get("/regression/linear", { timeout: REGRESSION_TIMEOUT_MS });
      return response.data;
    } catch (error) {
      console.warn("API request failed, loading static precomputed data:", error.message);
    }
  }

  try {
    // Brief UX delay so model execution feedback is visible
    await delay(400);
    return await fetchStatic("regression_linear.json");
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export async function fetchLassoRegression() {
  if (API_URL) {
    try {
      const response = await client.get("/regression/lasso", { timeout: REGRESSION_TIMEOUT_MS });
      return response.data;
    } catch (error) {
      console.warn("API request failed, loading static precomputed data:", error.message);
    }
  }

  try {
    await delay(600);
    return await fetchStatic("regression_lasso.json");
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export async function fetchRidgeRegression() {
  if (API_URL) {
    try {
      const response = await client.get("/regression/ridge", { timeout: REGRESSION_TIMEOUT_MS });
      return response.data;
    } catch (error) {
      console.warn("API request failed, loading static precomputed data:", error.message);
    }
  }

  try {
    await delay(600);
    return await fetchStatic("regression_ridge.json");
  } catch (error) {
    throw toFriendlyError(error);
  }
}

export default client;
