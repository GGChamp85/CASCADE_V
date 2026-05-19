# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CASCADE-V is a multi-source audio attribution prototype: it ingests an AI-generated mix, finds which catalog stems contributed, splits payout via exact Shapley with fairness axioms, and emits a Z3-verified SMT-LIB proof certificate per receipt. The verification layer is the patent contribution (US 12,536,365 B1); everything else composes published methods (TRAK, group counterfactuals, Shapley) into one pipeline.

Target hardware: Apple Silicon (M-series) with the PyTorch MPS backend. `config.select_device()` falls back to CUDA, then CPU. Catalog/encoder sizes are tuned for an end-to-end laptop demo (~10 min), not production scale.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .              # editable install (pyproject.toml is the source of truth)
```

`requirements.txt` exists as a pin-list but `pyproject.toml` is what `pip install -e .` reads. `faiss-cpu` is an optional extra (`pip install -e ".[faiss]"`); the pipeline does not require it at the current catalog size.

## Common commands

The full demo is four scripts run in order. Each step's output is the next step's input, so they are independently re-runnable:

```bash
python scripts/build_catalog.py                 # 60 stems × 12 creators -> data/catalog/
python scripts/train_encoder.py                 # SimCLR-style training -> models/encoder.pt
python scripts/build_embeddings_and_tests.py    # catalog embeddings + 30 test mixes
python scripts/attribute.py output_001 --total-payout 1.00   # single attribution + receipt + SMT proof
python scripts/evaluate_all.py                  # CASCADE-V vs Shapley/LOO/TRAK on all 30 outputs
```

`scripts/attribute.py` is built with Typer; its `--help` lists flags.

## Tests

Tests are plain-Python smoke scripts, not pytest-collected (despite the `pytest_cache` ignore line). Run them directly:

```bash
python tests/test_core_math.py             # math-only: stages 1/2/3 + intervals + SMT writer
python tests/test_synth.py                 # synth + nonlinear mixer
python tests/test_imports.py               # import smoke
python tests/test_pipeline_integration.py  # full orchestrator on synthetic embeddings
```

`test_core_math.py` and `test_pipeline_integration.py` deliberately stub `torch` and `z3` (see the top of `test_core_math.py` and the shared stubs in `test_imports.py`) so they run without the heavy deps. **Do not add real torch/z3 imports at module top-level in code these tests cover** — keep heavy imports inside functions or behind the existing stubs, otherwise the stubbed tests break.

To verify a generated proof certificate independently:

```bash
brew install z3
z3 outputs/proofs/output_001.smt2   # expect: sat
```

## Architecture

### Pipeline stages (`src/cascade_v/`)

The orchestrator is `pipeline.run_cascade_v` (`pipeline.py`). It composes three numerical stages plus a verification stage, with invariant validation at every boundary:

1. **Stage 1 — Triage** (`stages/stage1_triage.py`): TRAK-style cosine influence on L2-normalized embeddings; reduces a 60- (or 3M-) source catalog to `TRIAGE_TOP_K` candidates.
2. **Stage 2 — Grouping** (`stages/stage2_grouping.py`): Ward-linkage hierarchical cluster on triaged embeddings, then group-wise counterfactual influence by re-evaluating the value function with each cluster removed. **This is what handles the "creator-DNA" case** — multiple stems by the same producer cluster, get one cluster weight, and don't double-split credit.
3. **Stage 3 — Shapley** (`stages/stage3_shapley.py`): exact `O(2^n)` enumeration when `|cluster| ≤ SHAPLEY_MAX_EXACT_N`, otherwise Monte Carlo with Hoeffding intervals at `HOEFFDING_CONFIDENCE`.
4. **Compose**: final per-source weight = `cluster_weight × within_cluster_shapley`, with `mpmath` interval arithmetic propagating bounds end-to-end (`verification/intervals.py`).
5. **Stage V — Proof** (`verification/proofs.py`): Z3 checks efficiency, symmetry, and dummy axioms numerically over the actual result, writes an SMT-LIB file to `outputs/proofs/<id>.smt2`, and returns a `ProofCertificate` with a sha256 of the proof text.

### Cascade Validation Protocol

`verification/validators.py` defines `validate_*_invariants` functions per stage, called by `pipeline.run_cascade_v` after each stage. Each stage module exposes its own `validate_<stage>_invariants` returning a `dict[str, bool]`. Default behavior on failure is to record a `StageValidation` and continue; pass `raise_on_validation_failure=True` to abort. When adding a new stage or changing an existing one, **add invariants to the matching validator** — the patent claim hinges on topologically-ordered pre/post-condition checks at every boundary.

### Value function abstraction

Coalitional value `v(S)` is the single seam between embeddings and attribution math. `types.make_embedding_value_function` returns a `ValueFunction = Callable[[tuple[int, ...]], float]` that's used by Stage 2 (counterfactual) and Stage 3 (Shapley). To swap embedding models or domains (e.g., images, text), replace the embedding pipeline and keep the same `ValueFunction` contract.

### Baselines (`baselines.py`)

Pure-Shapley, pure-LOO, and pure-TRAK methods all return the same `AttributionResult` (`types.py`) so `evaluate_all.py` can compare them apples-to-apples. When adding a new baseline, return `AttributionResult` with the same fields populated.

### Configuration (`config.py`)

Single source of truth for paths, hyperparameters, device, and seeds. **Always import constants from `config`** — do not hardcode paths or hyperparameters elsewhere. `ensure_dirs()` is idempotent and called at the start of every script. `GLOBAL_SEED = 42`; reproducibility breaks if scripts use `np.random` / `torch.random` without re-seeding from config.

### Receipts and proofs

`receipts.make_receipt` → `outputs/receipts/<id>.json` is the user-facing artifact (per-creator aggregation, per-source breakdown with intervals, verification block, proof hash). `outputs/proofs/<id>.smt2` is the auditor-facing artifact. Both must be regenerated together when attribution changes; never edit a `.smt2` by hand — `proofs.generate_proof_certificate` is the only writer.

## Generated/gitignored layout

`data/catalog/`, `data/test_outputs/`, `data/*.json`, `data/*.npy`, `models/*.pt`, and everything under `outputs/` are produced by the scripts and gitignored. Empty `.gitkeep` files preserve the directory structure. If a path looks "missing", run the corresponding script — don't recreate the file by hand.
