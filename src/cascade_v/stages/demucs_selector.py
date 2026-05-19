"""
demucs_selector.py — Pick the Demucs checkpoint that best covers the
union of catalog categories.

Why this exists: a 4-stem separator on a catalog with 12 instrument
categories (kick, snare, hat, bass, sub_bass, lead, pluck, arp, pad,
ambient, vocal_chop, fx) collapses 6 of 12 categories into the "other"
bucket, which kills attribution quality on those stems. The 6-stem
checkpoint (htdemucs_6s) splits piano and guitar out of "other" and
covers most of our catalog with dedicated channels.

This module inspects the catalog at startup and picks the model whose
stem set covers the most catalog stems with a non-"other" channel.
Override via CASCADE_DEMUCS_MODEL=htdemucs (or other explicit name).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-model: which Demucs stem each catalog category routes to. Anything
# not listed routes to "other". The mapping is intentionally hand-curated:
# "lead", "pluck", "arp" → guitar (most synth leads are guitar-like in
# spectral content); "pad", "ambient" → piano (sustained harmonic).
# ---------------------------------------------------------------------------

# Catalog category → Demucs stem (per model)
_CATEGORY_TO_STEM_4: dict[str, str] = {
    # htdemucs / htdemucs_ft / mdx_extra (4-stem)
    "kick": "drums",
    "snare": "drums",
    "hat": "drums",
    "drums": "drums",
    "percussion": "drums",
    "bass": "bass",
    "sub_bass": "bass",
    "vocal_chop": "vocals",
    "vocals": "vocals",
    "vocal": "vocals",
    # everything else → "other"
}

_CATEGORY_TO_STEM_6: dict[str, str] = {
    # htdemucs_6s — adds piano + guitar channels
    **_CATEGORY_TO_STEM_4,
    "lead": "guitar",
    "pluck": "guitar",
    "arp": "guitar",
    "guitar": "guitar",
    "synth_lead": "guitar",
    "pad": "piano",
    "ambient": "piano",
    "piano": "piano",
    "synth_pad": "piano",
    "keys": "piano",
    "organ": "piano",
}

# Each entry: model_name → (n_stems, category_to_stem_map, stem_names)
MODEL_REGISTRY: dict[str, dict] = {
    "htdemucs": {
        "n_stems": 4,
        "stems": ("drums", "bass", "other", "vocals"),
        "category_map": _CATEGORY_TO_STEM_4,
    },
    "htdemucs_ft": {
        "n_stems": 4,
        "stems": ("drums", "bass", "other", "vocals"),
        "category_map": _CATEGORY_TO_STEM_4,
    },
    "mdx_extra": {
        "n_stems": 4,
        "stems": ("drums", "bass", "other", "vocals"),
        "category_map": _CATEGORY_TO_STEM_4,
    },
    "htdemucs_6s": {
        "n_stems": 6,
        "stems": ("drums", "bass", "other", "vocals", "guitar", "piano"),
        "category_map": _CATEGORY_TO_STEM_6,
    },
}


# ---------------------------------------------------------------------------
# Coverage scoring + selection
# ---------------------------------------------------------------------------

def coverage_score(
    catalog_categories: list[str],
    model_name: str,
) -> tuple[float, int, int, dict]:
    """
    Compute coverage of a catalog by a Demucs model.

    Returns (covered_fraction, n_covered, n_total, per_category_routing).
    A category is "covered" when it routes to a non-"other" stem.
    """
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        return (0.0, 0, len(catalog_categories), {})
    cat_map = spec["category_map"]
    routing: dict[str, str] = {}
    n_covered = 0
    for cat in catalog_categories:
        target = cat_map.get(cat, "other")
        routing[cat] = target
        if target != "other":
            n_covered += 1
    n_total = max(len(catalog_categories), 1)
    return (n_covered / n_total, n_covered, n_total, routing)


def select_demucs_model(
    catalog_categories: list[str],
    candidates: tuple[str, ...] = ("htdemucs", "htdemucs_6s"),
) -> tuple[str, dict]:
    """
    Pick the model whose stem set covers the most catalog categories with
    a non-"other" channel. Ties broken by smaller model (fewer stems = less
    compute per attribution).

    `catalog_categories` should be the union of category strings across all
    catalog entries — built from catalog_metadata.json at startup.

    Returns (chosen_model_name, decision_log).
    """
    if not catalog_categories:
        # No information; default to the small model.
        return "htdemucs", {"reason": "no catalog categories provided", "fallback": True}

    scores: list[tuple[str, float, int, int, dict]] = []
    for name in candidates:
        cov, n_cov, n_total, routing = coverage_score(catalog_categories, name)
        scores.append((name, cov, n_cov, n_total, routing))

    # Highest coverage wins; on tie pick smaller model (fewer stems).
    scores.sort(
        key=lambda s: (-s[1], MODEL_REGISTRY[s[0]]["n_stems"])
    )
    chosen, cov, n_cov, n_total, routing = scores[0]

    decision = {
        "chosen": chosen,
        "coverage": cov,
        "n_covered": n_cov,
        "n_total_categories": n_total,
        "considered": [
            {
                "model": s[0],
                "coverage": s[1],
                "n_covered": s[2],
                "n_stems": MODEL_REGISTRY[s[0]]["n_stems"],
            }
            for s in scores
        ],
        "routing": routing,
    }
    return chosen, decision
