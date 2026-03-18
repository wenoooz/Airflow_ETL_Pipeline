# Airflow ETL Pipeline

An Apache Airflow pipeline that runs reproducible Sionna simulations, collects raw outputs, transforms them into a clean analytics dataset, and generates a KPI report.

## Prerequisites

- Docker and Docker Compose

## Quick Start (Docker)

From the **project root** (where `docker-compose.yml` and `Dockerfile` are):

```bash
docker compose build
docker compose up -d
```

Open **http://localhost:8081** in your browser.

### Login and default password

- **Username:** `airflow`
- **Password:** `airflow` 

If a random admin password was generated on first run, look it up in the container logs:

```bash
docker logs airflow-sionna 2>&1 | findstr -i "password"
```

## Running the pipeline

1. In the Airflow UI, find the DAG **sionna_etl_pipeline** and turn it **on** (unpause).
2. Click **Trigger DAG** (play button) to run it once.
3. After **compute_kpis**, the DAG pauses at **review_results** (HITLOperator). Click **Approve** to generate the report, or **Reject** to skip report generation (downstream `generate_report` will be skipped).
4. When all tasks are green, results are written under **`artifacts/<run_id>/`** on your machine.

### Reproducing with seeds

Trigger the DAG with config to control random seeds :

**Option 1 – Custom seed ID** (recommended for reproducibility):
```json
{"seed_id": "11"}
```
Use any fixed string and same `seed_id` always yields the same seeds.

**Option 2 – Reproduce a specific run:**
```json
{"reproduce_run_id": "manual__2026-03-18T16-43-40+00-00"}
```
Use the `run_id` or artifacts folder name of the run you want to reproduce.

### Output location

Results appear in the **`artifacts`** folder in the project root. Each run has a subfolder named from the run ID (colons in the ID are replaced by dashes for Windows compatibility), for example:

- `artifacts/manual__2026-03-14T21-07-06.679336+00-00/report.html`
- `artifacts/manual__2026-03-14T21-07-06.679336+00-00/dataset.csv`
- `artifacts/manual__2026-03-14T21-07-06.679336+00-00/kpis.json`
- `artifacts/manual__2026-03-14T21-07-06.679336+00-00/raw/` (per-simulation JSONs)

Open `report.html` in a browser to view the report.

## Project layout

- **`DAG/`** – Airflow DAG definition (`sionna_etl_pipeline.py`)
- **`scripts/`** – Run plan, Sionna simulator, transform, quality checks, KPIs, report generator
- **`config/`** – Parameter grid (`param_grid.yaml`) for simulations
- **`artifacts/`** – Pipeline output (per-run folders; ignored by Git)
- **`Dockerfile`** – Airflow image with LLVM, TensorFlow, Sionna
- **`docker-compose.yml`** – Airflow standalone + triggerer (for HITL/HITLOperator), project and `artifacts` mounted


