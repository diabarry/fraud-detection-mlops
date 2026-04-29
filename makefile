# Full ML Pipeline

# Variables
PYTHON = venv\Scripts\python.exe
DOCKER_COMPOSE = docker-compose -f docker/docker-compose.yml

.PHONY: setup help train serve stop monitor test test-api clean report

help:
	@echo "--- Available Commands ---"
	@echo "make setup     : Install environment and initialize DVC"
	@echo "make train     : Run full pipeline (DVC Pull -> Training)"
	@echo "make serve     : Start API and Monitoring stack (Docker)"
	@echo "make stop      : Shut down all Docker services"
	@echo "make test      : Run unit tests (pytest)"
	@echo "make test-api  : Send a test curl request to the API"
	@echo "make report    : Generate Drift Analysis report (Evidently)"
	@echo "make clean     : Remove temporary files and logs"

# Step 0: Initial Installation
setup:
	python -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	dvc init --no-scm

# Steps 1, 2 & 3: ML Pipeline
train:
	@echo "--- Syncing data (DVC) ---"
	dvc pull
	@echo "---  Validation, Training & MLflow Tracking ---"
	$(PYTHON) main.py

# Step 4: Deployment (Docker)
serve:
	@echo "---  Starting Stack (API + Monitoring) ---"
	$(DOCKER_COMPOSE) up --build -d

stop:
	@echo "---  Stopping services ---"
	$(DOCKER_COMPOSE) down

# Monitoring & Quality Assurance
report:
	@echo "---  Generating Evidently Drift Report ---"
	$(PYTHON) monitoring_drift.py

test:
	@echo "---  Running unit tests ---"
	$(PYTHON) -m pytest tests/ -v --disable-warnings

test-api:
	@echo "---  Testing Predict endpoint ---"
	curl -X 'POST' 'http://localhost:8000/predict' \
		-H 'Content-Type: application/json' \
		-d '{"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0,"Amount":100.0}'
explain:
	@echo "--- Generating SHAP Explanations ---"
	$(PYTHON) explainability.py

# Full Cleanup
clean:
	@echo "--- 🧹 Project cleanup ---"
	rm -rf mlruns/
	rm -rf monitoring/drift_report.html
	rm -rf pipeline.log
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +