.PHONY: help api test-api

help:
	@echo "🚀 ML Pipeline Commands"
	@echo "========================"
	@echo "  make api        - Start FastAPI server"
	@echo "  make test-api   - Test API endpoint"

api:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test-api:
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"size": 100, "rooms": 3}'
