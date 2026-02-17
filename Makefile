.PHONY: help install install-dev lint format test train generate-metadata generate-ppts run clean

PYTHON := python
PIP := pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"

lint: ## Run linter (ruff)
	ruff check src/ app/ scripts/ tests/

format: ## Auto-format code (ruff)
	ruff format src/ app/ scripts/ tests/

test: ## Run tests
	pytest tests/ -v --tb=short

train: ## Train model and save artifacts
	$(PYTHON) scripts/train_model.py

generate-metadata: ## Generate model metadata JSON
	$(PYTHON) scripts/generate_metadata.py

generate-ppts: ## Generate PowerPoint presentations
	$(PYTHON) scripts/generate_ppts.py

run: ## Run Streamlit app locally
	streamlit run app/app.py

clean: ## Remove generated files
	rm -rf outputs/charts/ outputs/ppts/
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
