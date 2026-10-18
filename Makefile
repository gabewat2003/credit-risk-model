.PHONY: help data features train evaluate cutoff pricing sensitivity fairness test clean all

PY := uv run python -m

help:
	@echo "Targets:"
	@echo "  make data      - Load data/raw/loan.csv into Postgres via COPY"
	@echo "  make features  - Run sql/01_build_target.sql + sql/02_features.sql"
	@echo "  make train     - Fit LogReg + LightGBM, calibrate, write reports/"
	@echo "  make evaluate  - Metrics table, decile table, calibration plot"
	@echo "  make cutoff    - Expected-profit sweep + cutoff.png"
	@echo "  make pricing   - Decile break-even APR table + pricing.png"
	@echo "  make sensitivity - LGD sweep over cutoff + pricing"
	@echo "  make fairness  - Disparate-impact audit across protected proxies"
	@echo "  make test      - Run pytest"
	@echo "  make all       - full pipeline through fairness"
	@echo "  make clean     - Remove reports/ artifacts"

data:
	$(PY) src.load_data

features:
	$(PY) src.build_features

train:
	$(PY) src.train

evaluate:
	$(PY) src.evaluate

cutoff:
	$(PY) src.cutoff

pricing:
	$(PY) src.pricing

sensitivity:
	$(PY) src.sensitivity

fairness:
	$(PY) src.fairness

test:
	uv run pytest

all: data features train evaluate cutoff pricing sensitivity fairness

clean:
	rm -f reports/*.png reports/*.csv reports/*.parquet reports/*.joblib reports/*.md
