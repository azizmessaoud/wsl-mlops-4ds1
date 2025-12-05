.PHONY: install train run all

install:
	pip install -r requirements.txt

train:
	python main.py --train

run:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

all: install train run

