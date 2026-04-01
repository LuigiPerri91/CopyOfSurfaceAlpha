# ╔══════════════════════════════════════════════════════════════╗
# ║  SurfaceAlpha — Makefile                                    ║
# ║  Run the full pipeline with short commands                  ║
# ╚══════════════════════════════════════════════════════════════╝

.DEFAULT_GOAL := help
PYTHON  := python
SCRIPTS := scripts

# ── Per-core directory roots ────────────────────────────────────
PILOT_DATA  := ./data/pilot
PILOT_OUT   := ./runs/pilot
LIQUID_DATA := ./data/liquid_core
LIQUID_OUT  := ./runs/liquid_core

# ── Full per-core pipelines ────────────────────────────────────

.PHONY: run-pilot
run-pilot: data-pilot train-pilot evaluate-pilot backtest-pilot explain-pilot ## Full pipeline for pilot core

.PHONY: run-liquid
run-liquid: data-liquid train-liquid evaluate-liquid backtest-liquid explain-liquid ## Full pipeline for liquid_core

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
	$(PYTHON) $(SCRIPTS)/build_surface.py

.PHONY: data
data: fetch canonical surfaces  ## Full data pipeline: fetch → clean → surfaces

.PHONY: data-pilot
data-pilot:  ## Full data pipeline for pilot core (SPY only) → data/pilot/
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) $(PYTHON) $(SCRIPTS)/fetch_options.py
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) $(PYTHON) $(SCRIPTS)/fetch_underlying.py
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) $(PYTHON) $(SCRIPTS)/fetch_market_state.py
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) $(PYTHON) $(SCRIPTS)/build_canonical.py
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) $(PYTHON) $(SCRIPTS)/build_surface.py

.PHONY: data-liquid
data-liquid:  ## Full data pipeline for liquid_core → data/liquid_core/
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) $(PYTHON) $(SCRIPTS)/fetch_options.py
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) $(PYTHON) $(SCRIPTS)/fetch_underlying.py
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) $(PYTHON) $(SCRIPTS)/fetch_market_state.py
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) $(PYTHON) $(SCRIPTS)/build_canonical.py
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) $(PYTHON) $(SCRIPTS)/build_surface.py

# ── Training ───────────────────────────────────────────────────

.PHONY: train
train:  ## Train all models (baselines + multimodal) via walk-forward
	$(PYTHON) $(SCRIPTS)/train.py

.PHONY: train-pilot
train-pilot:  ## Train pilot core (SPY only) → runs/pilot/
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) \
	$(PYTHON) $(SCRIPTS)/train.py --output $(PILOT_OUT)

.PHONY: train-liquid
train-liquid:  ## Train liquid_core → runs/liquid_core/
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) \
	$(PYTHON) $(SCRIPTS)/train.py --output $(LIQUID_OUT)

.PHONY: evaluate
evaluate:  ## Evaluate trained models: forecast + economic metrics
	$(PYTHON) $(SCRIPTS)/evaluate.py

.PHONY: evaluate-pilot
evaluate-pilot:  ## Evaluate pilot run
	$(PYTHON) $(SCRIPTS)/evaluate.py \
		--predictions-dir $(PILOT_OUT)/outputs/predictions \
		--output-dir $(PILOT_OUT)/outputs/evaluation

.PHONY: evaluate-liquid
evaluate-liquid:  ## Evaluate liquid_core run
	$(PYTHON) $(SCRIPTS)/evaluate.py \
		--predictions-dir $(LIQUID_OUT)/outputs/predictions \
		--output-dir $(LIQUID_OUT)/outputs/evaluation

# ── Backtesting ────────────────────────────────────────────────

.PHONY: backtest
backtest:  ## Run portfolio overlay backtest
	$(PYTHON) $(SCRIPTS)/backtest.py

.PHONY: backtest-pilot
backtest-pilot:  ## Backtest pilot core
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) \
	$(PYTHON) $(SCRIPTS)/backtest.py \
		--predictions-dir $(PILOT_OUT)/outputs/predictions \
		--output $(PILOT_OUT)

.PHONY: backtest-liquid
backtest-liquid:  ## Backtest liquid_core
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) \
	$(PYTHON) $(SCRIPTS)/backtest.py \
		--predictions-dir $(LIQUID_OUT)/outputs/predictions \
		--output $(LIQUID_OUT)

# ── Explainability ─────────────────────────────────────────────

.PHONY: explain
explain:  ## Generate SHAP values, ViT attributions, regime importance
	$(PYTHON) $(SCRIPTS)/explain.py

.PHONY: explain-pilot
explain-pilot:  ## Explain pilot model (fold 0)
	ACTIVE_SYMBOLS=pilot DATA_DIR=$(PILOT_DATA) \
	$(PYTHON) $(SCRIPTS)/explain.py \
		--checkpoint $(PILOT_OUT)/outputs/checkpoints/fold_0/best.pt \
		--explain-output $(PILOT_OUT)/outputs/explain

.PHONY: explain-liquid
explain-liquid:  ## Explain liquid_core model (fold 0)
	ACTIVE_SYMBOLS=liquid_core DATA_DIR=$(LIQUID_DATA) \
	$(PYTHON) $(SCRIPTS)/explain.py \
		--checkpoint $(LIQUID_OUT)/outputs/checkpoints/fold_0/best.pt \
		--explain-output $(LIQUID_OUT)/outputs/explain

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

.PHONY: clean-data
clean-data:  ## Delete all generated data (keeps DoltHub clone + philippdubach download cache)
	rm -f data/raw/option_chain.parquet
	rm -f data/raw/volatility_history.parquet
	rm -f data/raw/underlying.parquet
	rm -f data/raw/underlying_all.parquet
	rm -f data/raw/market_state.parquet
	rm -f data/raw/fetch_*.json
	rm -f data/raw/dubach_provenance.json
	rm -rf data/raw/dolt_cache
	rm -f data/canonical/*.parquet data/canonical/*.json
	rm -f data/processed/sample_index.parquet data/processed/build_surface_meta.json
	rm -rf data/processed/surfaces data/processed/returns data/processed/vol_history data/processed/market_state
	@echo "Data cleaned. Download caches preserved (data/raw/dolt_clone, data/raw/dubach_cache)."

.PHONY: clean-data-all
clean-data-all:  ## Delete all generated data AND download caches (full reset, slow re-fetch)
	$(MAKE) clean-data
	rm -rf data/raw/dolt_clone data/raw/dubach_cache
	@echo "Full data reset complete."

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
