"""cascade-build-catalog — generate the synthetic catalog of audio stems."""

from __future__ import annotations

import typer

from cascade_v.config import (
    CASCADE_LOG_PATH,
    CATALOG_DIR,
    CATALOG_METADATA_PATH,
    GLOBAL_SEED,
    ensure_dirs,
)
from cascade_v.logging_setup import configure_logging, event, get_logger
from cascade_v.settings import override_settings
from cascade_v.utils.determinism import set_global_seeds
from cascade_v.utils.synth import build_catalog


app = typer.Typer(add_completion=False)


@app.command()
def main(
    seed: int = typer.Option(GLOBAL_SEED, "--seed"),
    catalog_size: int = typer.Option(None, "--catalog-size"),
    n_creators: int = typer.Option(None, "--n-creators"),
    force: bool = typer.Option(False, "--force", help="Regenerate even if metadata exists"),
    source: str = typer.Option(
        "synthetic", "--source",
        help="Catalog source: 'synthetic' (oscillator-based, offline) or "
             "'slakh' (real MIDI-rendered stems; requires --slakh-root).",
    ),
    slakh_root: str = typer.Option(
        None, "--slakh-root",
        help="Path to Slakh2100 root (required when --source=slakh).",
    ),
):
    configure_logging(CASCADE_LOG_PATH)
    log = get_logger("cli.build_catalog")

    overrides = {}
    if catalog_size is not None:
        overrides["CATALOG_SIZE"] = catalog_size
    if n_creators is not None:
        overrides["N_CREATORS"] = n_creators
    if overrides:
        override_settings(**overrides)
        # Re-import to pick up new constants
        import importlib
        import cascade_v.config
        importlib.reload(cascade_v.config)

    from cascade_v.config import (  # noqa
        CATALOG_DIR as _CD, CATALOG_METADATA_PATH as _CMP, CATALOG_SIZE, N_CREATORS,
    )

    ensure_dirs()
    set_global_seeds(seed)

    if _CMP.exists() and not force:
        event(log, "script.skip", reason="metadata exists", path=str(_CMP))
        return

    if source == "slakh":
        if not slakh_root:
            raise typer.BadParameter("--slakh-root is required when --source=slakh")
        # Delegate to the standalone Slakh ingestion script — keep the
        # logic in one place. Re-using as a subprocess avoids polluting
        # the CLI module with PyYAML / pretty_midi imports.
        import subprocess
        import sys

        cmd = [
            sys.executable,
            str((__import__("pathlib").Path(__file__).resolve().parents[3]
                 / "scripts" / "build_catalog_from_slakh.py")),
            "--slakh-root", slakh_root,
            "--catalog-size", str(CATALOG_SIZE),
            "--seed", str(seed),
            "--output-dir", str(_CD),
            "--metadata-path", str(_CMP),
        ]
        event(log, "script.start", source="slakh", catalog_size=CATALOG_SIZE,
              slakh_root=slakh_root, seed=seed)
        ret = subprocess.call(cmd)
        if ret != 0:
            raise typer.Exit(ret)
        event(log, "script.done", source="slakh", metadata=str(_CMP))
        return

    event(log, "script.start", source="synthetic",
          catalog_size=CATALOG_SIZE, n_creators=N_CREATORS, seed=seed)
    records = build_catalog(
        output_dir=_CD,
        metadata_path=_CMP,
        n_sources=CATALOG_SIZE,
        n_creators=N_CREATORS,
        seed=seed,
    )
    event(log, "script.done", source="synthetic", n_stems=len(records), metadata=str(_CMP))


if __name__ == "__main__":
    app()
