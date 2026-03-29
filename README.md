# PULSE-OPS — Autonomous MLOps Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-orange.svg)](https://mlflow.org/)
[![Prefect](https://img.shields.io/badge/Prefect-2.19.0-blue.svg)](https://www.prefect.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

PULSE-OPS is a **production-grade, autonomous MLOps platform** that automates the full machine-learning lifecycle — from raw data ingestion and feature engineering through model training, hyperparameter optimisation, deployment, drift detection, and autonomous retraining. It integrates the most widely-used open-source MLOps tools into a single cohesive platform with a real-time monitoring dashboard.

**Key capabilities:**
- Automated data pipelines for multiple public datasets (Adult Income, Wine Quality, Bike Sharing, German Credit)
- Hyperparameter optimisation via Optuna with MLflow experiment tracking
- Feature store powered by Feast for consistent train/serve feature access
- Data-version control with DVC for full reproducibility
- Continuous drift detection using Evidently AI
- Autonomous retraining triggered by statistically significant drift
- Real-time Prometheus metrics exposed and visualised in Grafana
- Interactive Dash dashboard with model comparison, drift reports, and pipeline DAG views
- FastAPI serving layer for low-latency predictions
- Prefect orchestration for robust, observable workflow scheduling

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PULSE-OPS Platform                          │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐    │
│  │   Data   │   │ Feature  │   │  Model   │   │   Serving    │    │
│  │ Ingestion│──▶│  Store   │──▶│Training  │──▶│  (FastAPI)   │    │
│  │  (DVC)   │   │ (Feast)  │   │(MLflow+  │   │              │    │
│  └──────────┘   └──────────┘   │ Optuna)  │   └──────┬───────┘    │
│                                └────┬─────┘          │            │
│                                     │                ▼            │
│  ┌──────────┐   ┌──────────┐   ┌────▼─────┐   ┌──────────────┐   │
│  │ Dashboard│◀──│Monitoring│◀──│  Drift   │   │   Registry   │   │
│  │  (Dash)  │   │(Prom/Graf│   │Detection │   │  (MLflow)    │   │
│  └──────────┘   └──────────┘   │(Evidently│   └──────────────┘   │
│                                └────┬─────┘                       │
│  ┌──────────────────────────────────▼──────────────────────────┐  │
│  │              Orchestration (Prefect 2)                       │  │
│  │  training_flow  │  drift_flow  │  retraining_flow            │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

Data flows left-to-right: raw CSV/API data → DVC-versioned storage → Feast feature store → model training → MLflow model registry → FastAPI endpoints. Evidently monitors live predictions against a rolling reference window; when drift exceeds configured thresholds Prefect triggers a full retraining run automatically.

---

## 3. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Experiment Tracking | MLflow | 2.13.0 |
| Orchestration | Prefect | 2.19.0 |
| Drift Detection | Evidently AI | 0.4.26 |
| Feature Store | Feast | 0.39.0 |
| Data Versioning | DVC | 3.51.2 |
| HPO | Optuna | 3.6.1 |
| Gradient Boosting | XGBoost / LightGBM | 2.0.3 / 4.3.0 |
| Deep Learning | PyTorch | 2.3.0 |
| Classical ML | scikit-learn | 1.4.2 |
| Data Validation | Great Expectations | 0.18.15 |
| Metrics | Prometheus Client | 0.20.0 |
| API Serving | FastAPI + Uvicorn | 0.111.0 / 0.29.0 |
| Dashboard | Dash + Plotly | 2.17.0 / 5.22.0 |
| Experiment UI | Weights & Biases | 0.16.6 |
| Containerisation | Docker / Compose | latest |
| Monitoring | Prometheus + Grafana | latest |
| Cache / Queue | Redis | 7-alpine |

---

## 4. Prerequisites

- **Python** ≥ 3.11
- **Docker** ≥ 24.0 and **Docker Compose** ≥ 2.24
- **Git** ≥ 2.40
- **Make** (optional but recommended)
- 8 GB RAM minimum (16 GB recommended for PyTorch models)
- 10 GB free disk space for datasets, artefacts, and container images

---

## 5. Installation

### Clone and set up environment

```bash
git clone https://github.com/PULSE-OPS/PULSE-OPS.git
cd PULSE-OPS

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Copy and configure environment variables
cp .env.example .env
# Edit .env as needed
```

### Start infrastructure services

```bash
make stack
# or
docker-compose up -d
```

Services started:
| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Prefect UI | http://localhost:4200 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / pulse123) |
| Redis | localhost:6379 |

---

## 6. Quick Start

```bash
# 1. Start infrastructure
make stack

# 2. Fetch and version datasets
make fetch
make dvc

# 3. Apply feature store definitions
make feast

# 4. Run the full training pipeline
make train

# 5. Check for data / concept drift
make drift

# 6. Start the prediction API
make api

# 7. Launch the interactive dashboard
make run
```

To trigger an autonomous retraining cycle manually:

```bash
make retrain
```

---

## 7. Project Structure

```
PULSE-OPS/
├── configs/                    # YAML configuration files
│   ├── pipeline_config.yaml    # MLflow, Prefect, dataset settings
│   ├── drift_config.yaml       # Drift detection thresholds
│   ├── model_config.yaml       # Model hyperparameters & HPO settings
│   ├── feast_config.yaml       # Feature store configuration
│   └── monitoring_config.yaml  # Prometheus / Grafana settings
│
├── data/                       # Data layer
│   ├── fetchers/               # Dataset download utilities
│   ├── processors/             # dataset_builder.py and transformers
│   └── raw/                    # Raw data files (DVC-tracked, gitignored)
│
├── feature_store/              # Feast feature store
│   └── feast_repo/             # Feature views, entities, data sources
│
├── models/                     # Model implementations
│   ├── classification/         # XGBoost, LightGBM, Neural Net classifiers
│   └── regression/             # XGBoost, LightGBM regressors
│
├── registry/                   # Model registry helpers (MLflow wrapper)
│
├── drift/                      # Drift detection
│   └── reports/                # Generated Evidently HTML reports
│
├── retraining/                 # Autonomous retraining logic
│
├── orchestration/              # Prefect flows
│   ├── training_flow.py        # End-to-end training pipeline
│   ├── drift_flow.py           # Scheduled drift monitoring flow
│   └── retraining_flow.py      # Autonomous retraining flow
│
├── serving/                    # Model serving
│   └── fastapi_server.py       # REST prediction API
│
├── monitoring/                 # Prometheus metrics & alerting
│
├── pipeline/                   # Pipeline utilities and helpers
│
├── dashboard/                  # Dash application
│   ├── app.py                  # Main Dash app entry point
│   ├── pages/                  # Individual page layouts
│   └── callbacks/              # Dash callback definitions
│
├── utils/                      # Shared utilities
│
├── tests/                      # Test suite
│
├── notebooks/                  # Exploratory notebooks
│
├── assets/                     # Static assets
│   └── results/                # Saved plots & artefacts
│
├── prometheus/                 # Prometheus configuration
│   └── prometheus.yml
│
├── grafana/                    # Grafana dashboards
│   └── dashboards/
│
├── mlruns/                     # MLflow run artefacts (gitignored)
│
├── docker-compose.yml          # Full platform stack
├── Makefile                    # Developer convenience targets
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

---

## 8. Data Sources

PULSE-OPS ships with automated fetchers for four public UCI/Kaggle datasets:

| Dataset | Task | Features | Rows | Source |
|---------|------|----------|------|--------|
| **Adult Income** | Binary classification | 14 | 48,842 | UCI ML Repository |
| **Wine Quality** | Regression | 12 | 6,497 | UCI ML Repository |
| **Bike Sharing** | Regression | 16 | 17,389 | UCI ML Repository |
| **German Credit** | Binary classification | 20 | 1,000 | UCI ML Repository |

Run `make fetch` to download and version all datasets. Raw files are stored in `data/raw/` and tracked with DVC.

---

## 9. MLOps Components

### Experiment Tracking (MLflow)
Every training run logs parameters, metrics, model artefacts, and conda/pip environments. Access the experiment UI at `http://localhost:5000`. Autologging is enabled for XGBoost, LightGBM, and scikit-learn.

### Hyperparameter Optimisation (Optuna)
Each model type has an Optuna study with up to 50 trials (configurable in `configs/model_config.yaml`). Best parameters are logged to MLflow and persisted in the Optuna SQLite journal.

### Feature Store (Feast)
Feast manages feature definitions, point-in-time joins for training, and low-latency online feature retrieval for serving. Run `make feast` to apply feature definitions.

### Data Versioning (DVC)
All raw and processed datasets are tracked with DVC. The default remote is `./dvc_remote` (local), but can be changed to S3, GCS, or Azure in `.dvc/config`.

### Drift Detection (Evidently)
The `drift_flow` runs on a configurable schedule and computes:
- **Data Drift** — Jensen-Shannon divergence per feature (threshold: 0.3)
- **Target / Prediction Drift** — KS-test p-value (alpha: 0.05)
- **Data Quality** — missing values, range violations (threshold: 0.9)

HTML reports are saved to `drift/reports/`.

### Autonomous Retraining
When drift exceeds configured thresholds, the `retraining_flow` automatically:
1. Fetches the latest data window
2. Re-runs the full HPO training pipeline
3. Evaluates the challenger model against the incumbent
4. Promotes the challenger if it improves by > `eval_metric_threshold`

### Model Registry (MLflow Model Registry)
Models are versioned and staged (`Staging` → `Production` → `Archived`) via the MLflow Model Registry. The serving layer always loads the `Production` alias.

---

## 10. Dashboard

The Dash dashboard (`make run`) provides:

- **Overview** — platform health, active models, recent run summary
- **Model Performance** — accuracy/RMSE trends, confusion matrices, ROC curves
- **Drift Monitor** — feature distribution comparisons, drift score timelines
- **Pipeline DAG** — interactive Cytoscape graph of the Prefect flow topology
- **HPO Explorer** — Optuna parallel-coordinate and contour plots
- **Data Quality** — Great Expectations validation result summaries
- **Experiment Comparison** — side-by-side MLflow run comparison

---

## 11. Testing

```bash
# Run the full test suite
make test

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=html

# Run a specific test module
pytest tests/test_drift.py -v
```

Test categories:
- `tests/test_data_pipeline.py` — data fetching and preprocessing
- `tests/test_models.py` — model training, prediction shapes, serialisation
- `tests/test_drift.py` — drift detection logic and threshold evaluation
- `tests/test_serving.py` — FastAPI endpoint contracts
- `tests/test_registry.py` — model registry promotion logic
- `tests/test_feature_store.py` — Feast feature retrieval

---

## 12. CI/CD

GitHub Actions workflows (`.github/workflows/`) provide:

| Workflow | Trigger | Steps |
|----------|---------|-------|
| `ci.yml` | Push / PR to `main` | lint → test → build |
| `train.yml` | Push to `main` + schedule | fetch → train → register |
| `drift.yml` | Schedule (every 6 h) | drift check → alert / retrain |
| `release.yml` | Tag `v*.*.*` | build Docker image → push GHCR |

Required secrets: `MLFLOW_TRACKING_URI`, `PREFECT_API_KEY`, `WANDB_API_KEY`.

---

## 13. Configuration

All runtime settings live in `configs/`:

| File | Purpose |
|------|---------|
| `pipeline_config.yaml` | MLflow URI, experiment name, dataset list, active model mapping |
| `drift_config.yaml` | Drift thresholds, check schedule, reference/drift window sizes |
| `model_config.yaml` | Model hyperparameters, HPO trial budget, evaluation threshold |
| `feast_config.yaml` | Feast project name, online/offline store paths |
| `monitoring_config.yaml` | Prometheus port, Grafana port, alert thresholds |

Environment variables override YAML values at runtime. Copy `.env.example` to `.env` and populate as needed.

---

## 14. Contributing

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install** development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Write tests** for new functionality and ensure all existing tests pass:
   ```bash
   make test
   ```

4. **Lint** your code:
   ```bash
   make lint
   ```

5. **Commit** with a descriptive message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat(drift): add KL-divergence detector for continuous features"
   ```

6. **Open a Pull Request** against `main`. The CI pipeline will run automatically.

### Code Style
- Line length: 100 characters (Black + Flake8)
- Type hints required for all public functions
- Docstrings follow NumPy style

### Reporting Issues
Please open a GitHub Issue with a minimal reproducible example and the output of:
```bash
python --version && pip list | grep -E "mlflow|prefect|evidently|feast"
```

---

*PULSE-OPS — Closing the loop between data science and production.*
