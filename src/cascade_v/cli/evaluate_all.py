"""cascade-evaluate — run all 4 methods on all test outputs and produce comparison artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from cascade_v.baselines import (
    attribute_loo_alone,
    attribute_shapley_alone,
    attribute_trak_alone,
)
from cascade_v.config import (
    CASCADE_LOG_PATH,
    CATALOG_EMBEDDINGS_PATH,
    CATALOG_METADATA_PATH,
    GLOBAL_SEED,
    GROUND_TRUTH_PATH,
    OUTPUTS_DIR,
    PLOTS_DIR,
    PROOFS_DIR,
    RECEIPTS_DIR,
    SAMPLE_RATE,
    TEST_OUTPUTS_DIR,
    USE_DEMUCS_SEPARATION,
    ensure_dirs,
)
from cascade_v.embeddings import embed_audio, load_catalog_embeddings
from cascade_v.evaluate import (
    PerOutputMetrics,
    aggregate_metrics,
    compute_coverage_at_k,
    compute_creator_mae,
    compute_creator_precision_recall_at_k,
    compute_instance_mae,
    compute_precision_recall_at_k,
    compute_top1_hit,
)
from cascade_v.generate import load_ground_truth
from cascade_v.logging_setup import configure_logging, event, get_logger
from cascade_v.pipeline import run_cascade_v
from cascade_v.receipts import make_receipt, save_receipt
from cascade_v.train import load_encoder
from cascade_v.utils.audio import load_wav
from cascade_v.utils.determinism import set_global_seeds
from cascade_v.utils.synth import load_catalog_metadata


app = typer.Typer(add_completion=False)
console = Console()


def _detect_dna_case(gt) -> bool:
    counts: dict[str, int] = {}
    for cid in gt.creator_ids:
        counts[cid] = counts.get(cid, 0) + 1
    return any(c >= 2 for c in counts.values())


@app.command()
def main(
    seed: int = typer.Option(GLOBAL_SEED, "--seed"),
    strict: bool = typer.Option(True, "--strict/--no-strict"),
    force: bool = typer.Option(False, "--force"),
    save_receipts: bool = typer.Option(True, "--save-receipts/--no-save-receipts",
                                       help="Persist receipt JSON for each test output"),
):
    configure_logging(CASCADE_LOG_PATH)
    log = get_logger("cli.evaluate_all")
    ensure_dirs()
    set_global_seeds(seed)

    csv_path = OUTPUTS_DIR / "results.csv"
    if csv_path.exists() and not force:
        event(log, "script.skip", reason="results.csv exists", path=str(csv_path))
        return

    creators, sources = load_catalog_metadata(CATALOG_METADATA_PATH)
    catalog_ids = [s["source_id"] for s in sources]
    source_to_creator = {s["source_id"]: s["creator_id"] for s in sources}
    creator_names = {c["creator_id"]: c["name"] for c in creators}
    catalog_embeddings = load_catalog_embeddings(CATALOG_EMBEDDINGS_PATH)
    encoder = load_encoder()

    ground_truths = load_ground_truth(GROUND_TRUTH_PATH)
    event(log, "script.start", n_outputs=len(ground_truths), n_methods=4)

    all_metrics: list[PerOutputMetrics] = []

    for gt in tqdm(ground_truths, desc="evaluating"):
        target_path = TEST_OUTPUTS_DIR / f"{gt.output_id}.wav"
        target_audio = load_wav(target_path)
        target_embedding = embed_audio(target_audio, encoder)
        is_dna = _detect_dna_case(gt)

        if USE_DEMUCS_SEPARATION:
            from cascade_v.pipeline_demucs import run_cascade_v_with_demucs
            cv_result = run_cascade_v_with_demucs(
                target_id=gt.output_id,
                target_audio=target_audio,
                sample_rate=SAMPLE_RATE,
                encoder=encoder,
                catalog_embeddings=catalog_embeddings,
                catalog_ids=catalog_ids,
                proofs_dir=PROOFS_DIR,
                raise_on_validation_failure=strict,
                seed=seed,
            )
        else:
            cv_result = run_cascade_v(
                target_id=gt.output_id,
                target_embedding=target_embedding,
                catalog_embeddings=catalog_embeddings,
                catalog_ids=catalog_ids,
                proofs_dir=PROOFS_DIR,
                raise_on_validation_failure=strict,
                seed=seed,
            )
        if save_receipts:
            receipt = make_receipt(
                cv_result, source_to_creator=source_to_creator,
                creator_names=creator_names, total_payout_usd=1.0,
                catalog_embeddings=catalog_embeddings,
                catalog_ids=catalog_ids,
                target_embedding=target_embedding,
                seed=seed,
            )
            save_receipt(receipt, RECEIPTS_DIR)

        proven = sum(1 for a in cv_result.proof.axioms if a.status == "PROVEN")
        total = sum(1 for a in cv_result.proof.axioms if a.status != "NA")
        prec_k, rec_k = compute_precision_recall_at_k(gt, cv_result.attribution)
        cprec_k, crec_k = compute_creator_precision_recall_at_k(
            gt, cv_result.attribution, source_to_creator
        )
        all_metrics.append(PerOutputMetrics(
            output_id=gt.output_id, method="cascade_v",
            instance_mae=compute_instance_mae(gt, cv_result.attribution),
            creator_mae=compute_creator_mae(gt, cv_result.attribution, source_to_creator),
            top1_hit=compute_top1_hit(gt, cv_result.attribution),
            coverage_at_k=compute_coverage_at_k(gt, cv_result.attribution),
            precision_at_k=prec_k, recall_at_k=rec_k,
            creator_precision_at_k=cprec_k, creator_recall_at_k=crec_k,
            axioms_proven=proven, axioms_total=total, is_dna_case=is_dna,
        ))

        for fn, name in (
            (attribute_trak_alone, "trak_alone"),
            (attribute_loo_alone, "loo_alone"),
            (attribute_shapley_alone, "shapley_alone"),
        ):
            kwargs = {"seed": seed} if name == "shapley_alone" else {}
            attr = fn(target_embedding, catalog_embeddings, catalog_ids, **kwargs)
            prec_k, rec_k = compute_precision_recall_at_k(gt, attr)
            cprec_k, crec_k = compute_creator_precision_recall_at_k(gt, attr, source_to_creator)
            all_metrics.append(PerOutputMetrics(
                output_id=gt.output_id, method=name,
                instance_mae=compute_instance_mae(gt, attr),
                creator_mae=compute_creator_mae(gt, attr, source_to_creator),
                top1_hit=compute_top1_hit(gt, attr),
                coverage_at_k=compute_coverage_at_k(gt, attr),
                precision_at_k=prec_k, recall_at_k=rec_k,
                creator_precision_at_k=cprec_k, creator_recall_at_k=crec_k,
                axioms_proven=0, axioms_total=0, is_dna_case=is_dna,
            ))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_metrics[0]).keys()))
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(asdict(m))
    event(log, "results.csv.written", path=str(csv_path), n_rows=len(all_metrics))

    by_method: dict[str, list] = {}
    for m in all_metrics:
        by_method.setdefault(m.method, []).append(m)
    aggregates = {name: aggregate_metrics(ms) for name, ms in by_method.items()}

    table = Table(title="Method comparison")
    for col in ("Method", "Instance MAE", "Creator MAE", "DNA Creator MAE",
                "Top-1", "Cov@K", "Prec@K", "CreatorP@K", "Axioms"):
        table.add_column(col, justify="right" if col != "Method" else "left")
    for name in ["cascade_v", "shapley_alone", "loo_alone", "trak_alone"]:
        if name not in aggregates:
            continue
        a = aggregates[name]
        ax_str = (f"{a.axiom_satisfaction_rate * 100:.1f}%"
                  if not np.isnan(a.axiom_satisfaction_rate) else "—")
        table.add_row(
            name,
            f"{a.instance_mae_mean:.4f} ± {a.instance_mae_std:.4f}",
            f"{a.creator_mae_mean:.4f} ± {a.creator_mae_std:.4f}",
            f"{a.dna_creator_mae_mean:.4f} (n={a.dna_n})",
            f"{a.top1_hit_rate * 100:.1f}%",
            f"{a.coverage_at_k_mean * 100:.1f}%",
            f"{a.precision_at_k_mean * 100:.1f}%",
            f"{a.creator_precision_at_k_mean * 100:.1f}%",
            ax_str,
        )
    console.print(table)

    methods = [m for m in ["cascade_v", "shapley_alone", "loo_alone", "trak_alone"]
               if m in aggregates]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#0d9488", "#d97706", "#3b82f6", "#ef4444"][: len(methods)]
    inst = [aggregates[m].instance_mae_mean for m in methods]
    inst_err = [aggregates[m].instance_mae_std for m in methods]
    axes[0].bar(methods, inst, yerr=inst_err, capsize=4, color=colors)
    axes[0].set_title("Instance-level MAE")
    axes[0].tick_params(axis="x", rotation=20)
    cre = [aggregates[m].creator_mae_mean for m in methods]
    cre_err = [aggregates[m].creator_mae_std for m in methods]
    axes[1].bar(methods, cre, yerr=cre_err, capsize=4, color=colors)
    axes[1].set_title("Creator-level MAE")
    axes[1].tick_params(axis="x", rotation=20)
    cov = [aggregates[m].coverage_at_k_mean * 100 for m in methods]
    axes[2].bar(methods, cov, color=colors)
    axes[2].set_title("Coverage@K (%)")
    axes[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "comparison.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    event(log, "script.done", csv=str(csv_path), plot=str(plot_path))


if __name__ == "__main__":
    app()
