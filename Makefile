.PHONY: help install install-dev lint format test train generate-metadata generate-ppts run clean

PYTHON := python
PIP := pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install dev dependencies (pytest, ruff, ipykernel)
	$(PIP) install -e ".[dev]"

# ── Quality ───────────────────────────────────────────────────────────────────
lint: ## Run linter (ruff)
	ruff check src/ app/ scripts/ tests/

format: ## Auto-format code (ruff)
	ruff format src/ app/ scripts/ tests/

test: ## Run tests with coverage
	pytest tests/ -v --tb=short

# ── Training & Generation ─────────────────────────────────────────────────────
train: ## Train model -> models/liver_disease_model.pkl
	$(PYTHON) scripts/train_model.py

generate-metadata: ## Generate model metadata JSON
	$(PYTHON) scripts/generate_metadata.py

generate-ppts: ## Generate EDA + Modelling PowerPoint presentations
	$(PYTHON) scripts/generate_ppts.py

# ── Run ───────────────────────────────────────────────────────────────────────
run: ## Launch Streamlit app locally
	streamlit run app/app.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove generated outputs and caches
	rm -rf outputs/charts/ outputs/ppts/
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
