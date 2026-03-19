# ╔══════════════════════════════════════════════════════════════╗
# ║  SurfaceAlpha — Makefile                                    ║
# ║  Run the full pipeline with short commands                  ║
# ╚══════════════════════════════════════════════════════════════╝

.DEFAULT_GOAL := help
PYTHON := python
SCRIPTS := scripts

# ── Data Pipeline ──────────────────────────────────────────────

.PHONY: fetch-options
fetch-options:  ## Fetch option_chain + volatility_history from DoltHub
	$(PYTHON) $(SCRIPTS)/fetch_options.py

.PHONY: fetch-underlying
fetch-underlying:  ## Fetch underlying OHLCV from yfinance
	$(PYTHON) $(SCRIPTS)/fetch_underlying.py

.PHONY: fetch-market
fetch-market:  ## Fetch VIX, SPY returns, risk-free rate
	$(PYTHON) $(SCRIPTS)/fetch_market_state.py

.PHONY: fetch
fetch: fetch-options fetch-underlying fetch-market  ## Fetch all raw data

.PHONY: canonical
canonical:  ## Clean, join, and build canonical tables
	$(PYTHON) $(SCRIPTS)/build_canonical.py

.PHONY: surfaces
surfaces:  ## Build model-ready surface tensors + feature arrays
	$(PYTHON) $(SCRIPTS)/build_surfaces.py

.PHONY: data
data: fetch canonical surfaces  ## Full data pipeline: fetch → clean → surfaces

# ── Training ───────────────────────────────────────────────────

.PHONY: train
train:  ## Train all models (baselines + multimodal) via walk-forward
	$(PYTHON) $(SCRIPTS)/train.py

.PHONY: evaluate
evaluate:  ## Evaluate trained models: forecast + economic metrics
	$(PYTHON) $(SCRIPTS)/evaluate.py

# ── Backtesting ────────────────────────────────────────────────

.PHONY: backtest
backtest:  ## Run portfolio overlay backtest
	$(PYTHON) $(SCRIPTS)/backtest.py

# ── Explainability ─────────────────────────────────────────────

.PHONY: explain
explain:  ## Generate SHAP values, ViT attributions, regime importance
	$(PYTHON) $(SCRIPTS)/explain.py

# ── Full Pipeline ──────────────────────────────────────────────

.PHONY: all
all: data train evaluate backtest explain  ## Run everything end-to-end

# ── Development ────────────────────────────────────────────────

.PHONY: install
install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

.PHONY: test
test:  ## Run all tests
	pytest tests/ -v

.PHONY: test-fast
test-fast:  ## Run tests excluding slow/integration
	pytest tests/ -v -m "not slow and not integration"

.PHONY: lint
lint:  ## Lint and type-check
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/
	mypy src/volregime/

.PHONY: format
format:  ## Auto-format code
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

.PHONY: clean
clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

# ── Help ───────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
