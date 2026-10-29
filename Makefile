.PHONY: install samples es seed api demo test

install:
	pip install -r requirements.txt

samples:
	python -m samples.make_samples

es:
	docker compose up -d elasticsearch

seed:
	python -m scripts.seed_corpus

api:
	uvicorn app.main:app --port 8000

demo:
	python -m scripts.demo

test:
	pytest -q
