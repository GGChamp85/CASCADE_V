"""cascade-embed — embed catalog and generate test outputs."""

from __future__ import annotations

import sys

import typer

from cascade_v.config import (
    CASCADE_LOG_PATH,
    CATALOG_DIR,
    CATALOG_EMBEDDINGS_PATH,
    CATALOG_METADATA_PATH,
    GLOBAL_SEED,
    GROUND_TRUTH_PATH,
    TEST_OUTPUTS_DIR,
    ensure_dirs,
)
from cascade_v.embeddings import build_catalog_embeddings
from cascade_v.generate import generate_test_outputs
from cascade_v.logging_setup import configure_logging, event, get_logger
from cascade_v.train import load_encoder
from cascade_v.utils.determinism import set_global_seeds
from cascade_v.utils.synth import load_catalog_metadata


app = typer.Typer(add_completion=False)


@app.command()
def main(
    seed: int = typer.Option(GLOBAL_SEED, "--seed"),
    force: bool = typer.Option(False, "--force"),
):
    configure_logging(CASCADE_LOG_PATH)
    log = get_logger("cli.build_embeddings")
    ensure_dirs()
    set_global_seeds(seed)

    if not CATALOG_METADATA_PATH.exists():
        event(log, "script.error", reason="catalog not found",
              suggestion="cascade-build-catalog")
        sys.exit(1)

    if (CATALOG_EMBEDDINGS_PATH.exists() and GROUND_TRUTH_PATH.exists()
            and not force):
        event(log, "script.skip", reason="embeddings + ground truth exist")
        return

    _, sources = load_catalog_metadata(CATALOG_METADATA_PATH)
    encoder = load_encoder()
    source_paths = [CATALOG_DIR / f"{s['source_id']}.wav" for s in sources]
    event(log, "script.start", n_stems=len(source_paths))

    embeddings = build_catalog_embeddings(
        source_paths, encoder, out_path=CATALOG_EMBEDDINGS_PATH
    )
    event(log, "embed.done", shape=list(embeddings.shape))

    records = generate_test_outputs(
        sources=sources,
        catalog_dir=CATALOG_DIR,
        output_dir=TEST_OUTPUTS_DIR,
        ground_truth_path=GROUND_TRUTH_PATH,
        seed=seed,
    )
    n_dna = sum(
        1 for r in records
        if max(r.creator_weights.values()) > 0.5
        and len([w for w in r.creator_weights.values() if w > 0]) >= 2
    )
    event(log, "script.done", n_outputs=len(records), n_dna=n_dna,
          ground_truth=str(GROUND_TRUTH_PATH))


if __name__ == "__main__":
    app()
