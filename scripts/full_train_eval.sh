#!/usr/bin/env bash
# Chain: train custom encoder → re-embed catalog → cascade-evaluate.
# Each step only runs if the previous one succeeded.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs"
STAMP="$(date +%Y%m%d_%H%M)"
TRAIN_LOG="${LOG_DIR}/train_${STAMP}.log"
EMBED_LOG="${LOG_DIR}/embed_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/eval_${STAMP}.log"

echo "[chain] $(date -u +%FT%TZ) — start"
echo "[chain] removing old encoder + log"
rm -f models/encoder.pt logs/training.json

echo "[chain] $(date -u +%FT%TZ) — training (80 epochs, batch=16, MAX_PARTNERS=2)"
CASCADE_TRAIN_EPOCHS=80 CASCADE_TRAIN_BATCH_SIZE=16 \
  caffeinate -i -s -d \
  .venv/bin/python scripts/train_encoder.py --force 2>&1 | tee "${TRAIN_LOG}"

echo "[chain] $(date -u +%FT%TZ) — re-embedding catalog"
.venv/bin/python -c "
from cascade_v.embeddings import build_catalog_embeddings
from cascade_v.train import load_encoder
from cascade_v.config import CATALOG_DIR, CATALOG_EMBEDDINGS_PATH, CATALOG_METADATA_PATH
from cascade_v.utils.synth import load_catalog_metadata
_, sources = load_catalog_metadata(CATALOG_METADATA_PATH)
paths = [CATALOG_DIR / f\"{s['source_id']}.wav\" for s in sources]
enc = load_encoder()
print(f're-embedding {len(paths)} stems...')
embs = build_catalog_embeddings(paths, enc, out_path=CATALOG_EMBEDDINGS_PATH, verbose=False)
print(f'done. shape: {embs.shape}')
" 2>&1 | tee "${EMBED_LOG}"

echo "[chain] $(date -u +%FT%TZ) — cascade-evaluate"
.venv/bin/python scripts/evaluate_all.py --force 2>&1 | tee "${EVAL_LOG}"

echo "[chain] $(date -u +%FT%TZ) — DONE"
echo "[chain] artifacts:"
echo "    train log:  ${TRAIN_LOG}"
echo "    embed log:  ${EMBED_LOG}"
echo "    eval log:   ${EVAL_LOG}"
echo "    csv:        outputs/results.csv"
echo "    receipts:   outputs/receipts/output_*.json"

# Touch a sentinel so the user can `ls -t logs/` and know it's done
touch "${LOG_DIR}/chain_${STAMP}.DONE"
