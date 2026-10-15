export PYTHONPATH := src

.PHONY: install pipeline dashboard test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
pipeline:   ## claims FWA model + bed forecast -> outputs/report.md
	python -m healthins.pipeline
dashboard:  ## http://localhost:8501
	streamlit run dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t uae-health-insurance-analytics . && docker run -p 8501:8501 uae-health-insurance-analytics
