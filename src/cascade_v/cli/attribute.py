"""cascade-attribute — run CASCADE-V on a single test output."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from cascade_v.config import (
    CASCADE_LOG_PATH,
    CATALOG_EMBEDDINGS_PATH,
    CATALOG_METADATA_PATH,
    GLOBAL_SEED,
    PROOFS_DIR,
    RECEIPTS_DIR,
    TEST_OUTPUTS_DIR,
    ensure_dirs,
)
from cascade_v.embeddings import embed_audio, load_catalog_embeddings
from cascade_v.logging_setup import configure_logging, event, get_logger
from cascade_v.pipeline import run_cascade_v
from cascade_v.receipts import make_receipt, save_receipt
from cascade_v.train import load_encoder
from cascade_v.utils.audio import load_wav
from cascade_v.utils.determinism import set_global_seeds
from cascade_v.utils.synth import load_catalog_metadata


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    output_id: str = typer.Argument(..., help="ID of the test output (e.g. 'output_001')"),
    total_payout: float = typer.Option(1.0, "--total-payout"),
    seed: int = typer.Option(GLOBAL_SEED, "--seed"),
    strict: bool = typer.Option(True, "--strict/--no-strict",
                                help="Abort on validation failure (default true)"),
    currencies: bool = typer.Option(
        None, "--currencies/--no-currencies",
        help="Attach the QWS + three-currency block to the receipt. "
             "Default follows ENABLE_CURRENCIES setting.",
    ),
    alpha: float = typer.Option(
        None, "--alpha",
        help="Currency dial α — fairness strictness on monetary payouts (0..1). "
             "Default from CURRENCY_ALPHA setting (0.85).",
    ),
    beta: float = typer.Option(
        None, "--beta",
        help="Currency dial β — recognition spread on lottery feature (0..1). "
             "Default from CURRENCY_BETA setting (0.60).",
    ),
    gamma: float = typer.Option(
        None, "--gamma",
        help="Currency dial γ — opportunity redistribution (0..1). "
             "Default from CURRENCY_GAMMA setting (0.30).",
    ),
):
    configure_logging(CASCADE_LOG_PATH)
    log = get_logger("cli.attribute")
    ensure_dirs()
    set_global_seeds(seed)

    creators, sources = load_catalog_metadata(CATALOG_METADATA_PATH)
    catalog_ids = [s["source_id"] for s in sources]
    source_to_creator = {s["source_id"]: s["creator_id"] for s in sources}
    creator_names = {c["creator_id"]: c["name"] for c in creators}
    catalog_embeddings = load_catalog_embeddings(CATALOG_EMBEDDINGS_PATH)

    target_path = TEST_OUTPUTS_DIR / f"{output_id}.wav"
    if not target_path.exists():
        console.print(f"[red]target not found: {target_path}[/red]")
        raise typer.Exit(1)

    encoder = load_encoder()
    target_audio = load_wav(target_path)
    target_embedding = embed_audio(target_audio, encoder)

    event(log, "script.start", output_id=output_id, total_payout=total_payout, strict=strict)
    result = run_cascade_v(
        target_id=output_id,
        target_embedding=target_embedding,
        catalog_embeddings=catalog_embeddings,
        catalog_ids=catalog_ids,
        proofs_dir=PROOFS_DIR,
        raise_on_validation_failure=strict,
        seed=seed,
    )

    console.print(f"\n[bold green]Pipeline complete[/bold green]")
    console.print(f"  Catalog size: {result.attribution.metadata['catalog_size']}")
    console.print(f"  Triaged: {result.attribution.metadata['candidates_from_triage']}")
    console.print(f"  Clusters: {result.attribution.metadata['n_clusters']}")
    console.print(f"  Total latency: {result.latency_ms.get('total', 0):.1f}ms")
    console.print(f"  Proof status: [bold]{result.proof.overall_status}[/bold]")

    table = Table(title="Per-source attribution")
    for col in ("Source", "Creator", "Weight", "Interval", "Payout"):
        table.add_column(col, justify="right" if col in ("Weight", "Interval", "Payout") else "left")
    n_show = min(8, len(result.attribution.source_ids))
    sorted_idx = sorted(range(len(result.attribution.source_ids)),
                        key=lambda i: -result.attribution.weights[i])
    for k in range(n_show):
        i = sorted_idx[k]
        sid = result.attribution.source_ids[i]
        cid = source_to_creator.get(sid, "?")
        cname = creator_names.get(cid, cid)
        w = result.attribution.weights[i]
        lo, hi = result.attribution.intervals[i]
        table.add_row(
            sid, cname, f"{w * 100:.1f}%",
            f"[{lo * 100:.1f}, {hi * 100:.1f}]%",
            f"${w * total_payout:.4f}",
        )
    console.print(table)

    creator_table = Table(title="Per-creator payout")
    for col in ("Creator", "Sources", "Weight", "Payout"):
        creator_table.add_column(col, justify="right" if col != "Creator" else "left")
    creator_totals: dict[str, dict] = {}
    for i, sid in enumerate(result.attribution.source_ids):
        cid = source_to_creator.get(sid, "unknown")
        agg = creator_totals.setdefault(cid, {"weight": 0.0, "n": 0,
                                              "name": creator_names.get(cid, cid)})
        agg["weight"] += float(result.attribution.weights[i])
        agg["n"] += 1
    for cid, agg in sorted(creator_totals.items(), key=lambda kv: -kv[1]["weight"]):
        creator_table.add_row(
            agg["name"], str(agg["n"]),
            f"{agg['weight'] * 100:.1f}%",
            f"${agg['weight'] * total_payout:.4f}",
        )
    console.print(creator_table)

    axiom_table = Table(title="Fairness axioms (Z3 verified)")
    for col in ("Axiom", "Status", "Detail"):
        axiom_table.add_column(col)
    for ax in result.proof.axioms:
        color = {"PROVEN": "green", "VIOLATED": "red", "NA": "dim"}[ax.status]
        axiom_table.add_row(ax.name, f"[{color}]{ax.status}[/{color}]", ax.detail)
    console.print(axiom_table)
    console.print(f"\nSMT-LIB proof: {result.proof.smt_lib_file}")
    console.print(f"Hash: {result.proof.smt_lib_hash[:16]}...")

    receipt = make_receipt(
        result, source_to_creator=source_to_creator,
        creator_names=creator_names, total_payout_usd=total_payout,
        catalog_embeddings=catalog_embeddings,
        catalog_ids=catalog_ids,
        target_embedding=target_embedding,
        enable_currencies=currencies,
        currency_alpha=alpha, currency_beta=beta, currency_gamma=gamma,
        seed=seed,
    )
    receipt_path = save_receipt(receipt, RECEIPTS_DIR)
    console.print(f"\nReceipt: {receipt_path}")
    if "currencies" in receipt:
        cb = receipt["currencies"]
        console.print(
            f"Currencies (α={cb['dials']['alpha']}, β={cb['dials']['beta']}, "
            f"γ={cb['dials']['gamma']}) — quantile={cb['qws']['quantile_active']}, "
            f"contributors={len(cb['per_contributor'])}, "
            f"dignity={cb['dignity_proof'].get('every_creator_received_all_three')}"
        )
    event(log, "script.done", receipt=str(receipt_path),
          status=result.proof.overall_status,
          latency_ms=result.latency_ms.get("total"))


if __name__ == "__main__":
    app()
