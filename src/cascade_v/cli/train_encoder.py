"""cascade-train — train the audio encoder via SimCLR contrastive learning."""

from __future__ import annotations

import sys

import typer

from cascade_v.config import (
    CASCADE_LOG_PATH,
    CATALOG_DIR,
    CATALOG_METADATA_PATH,
    DEVICE,
    ENCODER_CHECKPOINT,
    GLOBAL_SEED,
    TRAIN_EPOCHS,
    ensure_dirs,
)
from cascade_v.logging_setup import configure_logging, event, get_logger


app = typer.Typer(add_completion=False)


@app.command()
def main(
    seed: int = typer.Option(GLOBAL_SEED, "--seed"),
    epochs: int = typer.Option(None, "--epochs"),
    embedding_dim: int = typer.Option(None, "--embedding-dim"),
    force: bool = typer.Option(False, "--force"),
):
    configure_logging(CASCADE_LOG_PATH)
    log = get_logger("cli.train_encoder")

    overrides = {}
    if epochs is not None:
        overrides["TRAIN_EPOCHS"] = epochs
    if embedding_dim is not None:
        overrides["EMBEDDING_DIM"] = embedding_dim
    if overrides:
        from cascade_v.settings import override_settings
        override_settings(**overrides)
        import importlib
        import cascade_v.config
        importlib.reload(cascade_v.config)

    from cascade_v.config import (  # noqa
        CATALOG_DIR as _CD,
        CATALOG_METADATA_PATH as _CMP,
        DEVICE as _D,
        ENCODER_CHECKPOINT as _CKPT,
        ENCODER_KIND as _EK,
        TRAIN_EPOCHS as _EP,
    )
    from cascade_v.utils.synth import load_catalog_metadata

    ensure_dirs()

    # Resolve the actual checkpoint slot for this encoder kind. Custom
    # encoder writes to ENCODER_CHECKPOINT; clap_proj writes the head only
    # to a sibling file. clap (no head) doesn't train.
    if _EK == "clap":
        event(log, "script.skip",
              reason="ENCODER_KIND=clap is the frozen foundation model — nothing to train",
              path=None)
        return
    if _EK == "clap_proj":
        ckpt_for_skip = _CKPT.parent / "clap_proj_head.pt"
    else:
        ckpt_for_skip = _CKPT

    if ckpt_for_skip.exists() and not force:
        event(log, "script.skip", reason="checkpoint exists", path=str(ckpt_for_skip))
        return

    if not _CMP.exists():
        event(log, "script.error", reason="catalog not found",
              suggestion="cascade-build-catalog")
        sys.exit(1)

    _, sources = load_catalog_metadata(_CMP)
    audio_paths = [_CD / f"{s['source_id']}.wav" for s in sources]

    if _EK == "clap_proj":
        from cascade_v.train import train_clap_projection_head
        event(log, "script.start", encoder_kind=_EK, device=str(_D),
              n_stems=len(audio_paths), epochs=_EP, seed=seed)
        _, metrics = train_clap_projection_head(
            audio_paths, sources, epochs=_EP, seed=seed,
        )
    else:
        from cascade_v.train import train_encoder
        event(log, "script.start", encoder_kind=_EK, device=str(_D),
              n_stems=len(audio_paths), epochs=_EP, seed=seed)
        _, metrics = train_encoder(audio_paths, epochs=_EP, seed=seed)

    event(log, "script.done", final_loss=round(metrics[-1].loss, 6),
          checkpoint=str(ckpt_for_skip))


if __name__ == "__main__":
    app()
