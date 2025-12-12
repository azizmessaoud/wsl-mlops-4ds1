# Variables
IMAGE_NAME = aziz_messaoud_4ds1_mlops
DOCKER_USER = azizmessaoud
REPO_NAME = aziz-messaoud-4ds1-mlops

.PHONY: help api test-api build run push

help:
	@echo "🚀 ML Pipeline Commands"
	@echo "========================"
	@echo "  make api        - Start FastAPI server locally"
	@echo "  make test-api   - Test API endpoint"
	@echo "  make build      - Build Docker image"
	@echo "  make run        - Run Docker container"
	@echo "  make push       - Push to Docker Hub"

# --- Local Commands ---
api:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test-api:
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"size": 100, "rooms": 3}'

# --- Docker Commands ---
build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8000:8000 $(IMAGE_NAME)

push:
	docker tag $(IMAGE_NAME) $(DOCKER_USER)/$(REPO_NAME):latest
	docker push $(DOCKER_USER)/$(REPO_NAME):latest

mlflow:
	@echo "🚀 Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000
