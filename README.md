# Housing Price Regression Lab 🏠📊

An interactive Machine Learning exploration and laboratory web application built with **React**, **Tailwind CSS**, and **Vite**, demonstrating the end-to-end data preprocessing and regression modeling pipeline on the **Ames Housing Dataset** (`train.csv`).

## 🌟 Features

- **Dataset Preprocessing Pipeline**: Interactive step-by-step walkthrough of all 10 stages of the data cleaning pipeline:
  - Missing value imputation
  - High-missing feature removal
  - Target variable log transformation
  - One-hot encoding & StandardScaler
  - Multicollinearity / redundant feature removal
  - Outlier detection (IQR) & trimming
- **Regression Lab**: Live evaluation and comparison of machine learning models:
  - **Polynomial Linear Regression** (`degree=2`)
  - **Lasso Regression (L1)** with 4-fold cross-validation GridSearchCV
  - **Ridge Regression (L2)** with 4-fold cross-validation GridSearchCV
  - Real metrics: $R^2$ score, Mean Squared Error (MSE), CV scores, residuals, and actual vs predicted comparisons.
- **GitHub Pages Ready**: Can be deployed as a 100% static website with zero server configuration or costs!
- **Dual Mode**: Seamlessly switches between live FastAPI backend (for local development/custom models) and static fallback datasets (for production on GitHub Pages).

---

## 🚀 Live Demo on GitHub Pages

This repository includes automated GitHub Actions deployment. Once pushed to GitHub:
1. Go to **Settings > Pages** in your GitHub repository.
2. Under **Build and deployment > Source**, select **GitHub Actions**.
3. Push to `main` and your website will be live at:
   `https://<your-username>.github.io/<repository-name>/`

---

## 💻 Local Development

### 1. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The app will be available at `http://localhost:5173`.

### 2. (Optional) Python Backend Setup
If you want to run the live FastAPI backend server:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
The backend API runs on `http://localhost:8000`.

---

## 🛠 Re-exporting Static Datasets

If you modify `train.csv` or adjust preprocessing/models, you can regenerate the static JSON files anytime:
```bash
cd backend
python export_static_data.py
```
This updates the JSON files in `frontend/public/data/` for GitHub Pages.
