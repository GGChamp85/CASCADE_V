.PHONY: install build train embed evaluate test serve-api serve-dash demo stop clean

VENV ?= .venv
PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
ACT = . $(VENV)/bin/activate

# --- bootstrap ---
$(VENV)/bin/python:
	python3.13 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/python
	$(ACT) && pip install -e ".[server,dev]"
	cd dashboard && npm install --legacy-peer-deps

# --- pipeline (sequential) ---
build:
	$(ACT) && cascade-build-catalog

train:
	$(ACT) && cascade-train

embed:
	$(ACT) && cascade-embed

evaluate:
	$(ACT) && cascade-evaluate

test:
	$(ACT) && pytest tests/test_pipeline_smoke.py tests/test_determinism.py tests/test_encoder_shape.py -q

# --- runtime ---
serve-api:
	$(ACT) && uvicorn server.app.main:app --port 8000 --reload

serve-dash:
	cd dashboard && npm run dev

# --- one-shot demo ---
demo: build train embed evaluate
	@echo ""
	@echo "Pipeline ready. Now in two terminals:"
	@echo "    make serve-api"
	@echo "    make serve-dash"
	@echo "Then open http://localhost:3000"

stop:
	-pkill -f "uvicorn server.app.main" 2>/dev/null
	-pkill -f "next dev -p 3000" 2>/dev/null

clean:
	rm -rf data/catalog data/test_outputs data/uploads data/*.json data/*.npy
	rm -rf models/*.pt outputs/receipts/*.json outputs/proofs/*.smt2
	rm -rf outputs/results.csv outputs/plots/*.png logs/*.json logs/*.jsonl
