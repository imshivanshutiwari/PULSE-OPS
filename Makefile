.PHONY: install stack fetch dvc feast train drift retrain run api test lint notebook clean

install:
	pip install -r requirements.txt

stack:
	docker-compose up -d

fetch:
	python data/processors/dataset_builder.py --fetch

dvc:
	dvc add data/raw/ && dvc push

feast:
	cd feature_store/feast_repo && feast apply

train:
	python orchestration/training_flow.py

drift:
	python orchestration/drift_flow.py

retrain:
	python orchestration/retraining_flow.py

run:
	python dashboard/app.py

api:
	uvicorn serving.fastapi_server:app --port 8000 --host 0.0.0.0

test:
	pytest tests/ -v --color=yes

lint:
	black . --line-length 100 && flake8 . --max-line-length=100

notebook:
	jupyter notebook notebooks/

clean:
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
