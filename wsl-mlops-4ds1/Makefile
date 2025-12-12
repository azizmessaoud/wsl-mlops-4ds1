
.PHONY: help install train run-api test-api docker-build docker-run mlflow

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make train         - Train the model (with MLflow)"
	@echo "  make run-api       - Run FastAPI server"
	@echo "  make test-api      - Test API with curl"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make mlflow        - Start MLflow UI with SQLite"

install:
	pip install -r requirements.txt

train:
	python train_mlflow.py

run-api:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test-api:
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{"size": 100, "rooms": 3}"

docker-build:
	docker build -t mlops-app .

docker-run:
	docker run -p 8000:8000 mlops-app

mlflow:
	@echo "🚀 Starting MLflow UI with SQLite Backend..."
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
