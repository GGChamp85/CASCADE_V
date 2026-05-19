"use client";

import { useEffect } from "react";

export type StageKey =
  | "stage0"
  | "stage1"
  | "stage2"
  | "stage3"
  | "compose"
  | "stageV"
  | "currencies"
  | "overview";

type Reference = {
  label: string;
  href: string;
  authors?: string;
  year?: number;
};

type StageInfo = {
  number: string;
  title: string;
  oneLiner: string;
  what: string;
  why: string;
  how: string[];
  inputs: string;
  outputs: string;
  hyperparams?: { name: string; value: string; explain: string }[];
  references: Reference[];
  iconBg: string;
};

const STAGES: Record<StageKey, StageInfo> = {
  stage0: {
    number: "STEP 0",
    title: "Separate — Demucs source separation (optional)",
    oneLiner:
      "Split the input mix into drums, bass, other, vocals before running attribution on each stem.",
    what:
      "Demucs (Meta 2019-2023) is the state-of-the-art waveform music source separator. We run it on the input mix to produce 4 stems. Stages 1-V then run on each stem separately and per-source weights are aggregated by stem-energy share. Empirically halves instance-MAE on dominant-source-masking cases because each separated stem has fewer competing contributors for the encoder to disentangle. Stage 0 is bypassed when USE_DEMUCS_SEPARATION is False (set via env var CASCADE_USE_DEMUCS_SEPARATION).",
    why:
      "Stage 1 cosine retrieval finds the dominant source in a mix and misses masked secondary contributors. Splitting the mix first means each downstream attribution sees a cleaner signal — drums-only attribution catches drum stems, bass-only catches bass, etc.",
    how: [
      "stems = demucs(audio)                      # ~3 sec per file",
      "for stem in stems:",
      "    if rms(stem) > silence_threshold:",
      "        per_stem_result = run_cascade_v(stem)",
      "aggregated_weights = Σ_stem energy_share[stem] · per_stem_weight[stem]",
      "renormalize to Σ = 1, generate fresh Z3 proof on aggregated weights",
    ],
    inputs: "Raw mix audio (any sample rate, auto-resampled)",
    outputs:
      "4 separated stems passed to Stage 1, plus per-stem energy shares used for aggregation in Compose.",
    hyperparams: [
      {
        name: "USE_DEMUCS_SEPARATION",
        value: "True",
        explain:
          "Enable Stage 0. False reverts to single-pass attribution on the raw mix.",
      },
      {
        name: "DEMUCS_MODEL",
        value: "htdemucs",
        explain: "Hybrid Transformer Demucs — 4-stem separator from Meta.",
      },
      {
        name: "silence_rms_threshold",
        value: "0.005",
        explain:
          "Stems with RMS below this are skipped (no useful signal).",
      },
    ],
    references: [
      {
        label: "Hybrid Spectrogram and Waveform Source Separation",
        href: "https://arxiv.org/abs/2111.03600",
        authors: "Défossez",
        year: 2021,
      },
      {
        label: "Music Source Separation in the Waveform Domain",
        href: "https://arxiv.org/abs/1911.13254",
        authors: "Défossez, Usunier, Bottou, Bach",
        year: 2019,
      },
      {
        label: "facebookresearch/demucs",
        href: "https://github.com/facebookresearch/demucs",
      },
    ],
    iconBg: "bg-gradient-to-br from-fuchsia-500 to-pink-600",
  },
  overview: {
    number: "Overview",
    title: "CASCADE-V — Coalition-aware Source Crediting And Decomposed Engine, Verified",
    oneLiner:
      "Multi-source attribution for AI-generated audio with formal Z3-verified Shapley fairness.",
    what:
      "Given an AI-generated audio output and a catalog of source stems, identify which sources contributed to the output and how to split a payout fairly among their creators. Every receipt carries a Z3 SMT-LIB proof certificate that an auditor can re-verify independently.",
    why:
      "Three published methods solve parts of the problem; nobody has wired them together with formal verification on top. CASCADE-V composes TRAK (scale), GUDA-style group counterfactuals (creator-DNA), and exact Shapley (fairness axioms) — then proves the result with Z3.",
    how: [
      "Optionally separate the mix with Demucs (Stage 0) into drums/bass/other/vocals.",
      "Encode each stem with the encoder — custom 1.4M-param ResCNN trained with mix-consistency loss, or frozen LAION-CLAP backbone + trainable MLP head when ENCODER_KIND=clap_proj.",
      "Triage (Stage 1): NNLS-first rank in encoder embedding space + optional BPM/key pre-filter + residual matching pursuit when needed.",
      "Cluster candidates with Ward (default) or HDBSCAN, run group-counterfactual influence (Stage 2). Stems below MIN_CLUSTER_WEIGHT=0.07 are sparsified.",
      "Within each cluster: exact 2ⁿ Shapley or antithetic + stratified-MC with Hoeffding intervals (Stage 3); Owen value if OWEN_OR_DECOMP=owen.",
      "Compose with mpmath interval arithmetic; sparsify final weights below MIN_FINAL_WEIGHT=0.04 (false-positive control).",
      "Verify the composed result against the three Shapley fairness axioms with Z3 SMT.",
      "Optional currencies layer (Stage 6): QWS quantile-conditioned Shapley + Rawlsian floor monetary + role-tag recognition + opportunity routing, controlled by α/β/γ dials.",
    ],
    inputs: "Target audio (any sample rate; auto-resampled) + catalog embeddings (N × D, L2-normalized).",
    outputs:
      "Per-source weights with confidence intervals + per-creator aggregation + SMT-LIB proof certificate (sha-256 hashed) + optional triple-currency block with dignity meta-proof.",
    references: [
      {
        label: "US Patent 12,536,365 B1 — Cascade Validation Protocol",
        href: "https://patents.google.com/patent/US12536365B1",
        year: 2026,
      },
    ],
    iconBg: "bg-gradient-to-br from-violet-500 to-indigo-600",
  },

  stage1: {
    number: "STEP 1",
    title: "Triage — NNLS-first rank with optional BPM/key pre-filter",
    oneLiner:
      "Reduce a 5k–3M-source catalog to TRIAGE_TOP_K=40 candidates via NNLS decomposition in the encoder's embedding space, with cosine as a tie-breaker.",
    what:
      "Given the L2-normalized target and catalog embeddings: (a) optionally drop catalog entries whose BPM differs by >3 from the mix's estimated BPM and whose key isn't equal-or-relative-or-parallel-major-minor; (b) compute baseline-centered cosine similarities and run a quick NNLS probe on the global top-N — if it finds ≥5 active sources, skip the residual passes (they over-suppress legitimate same-creator near-neighbors); (c) otherwise run residual matching pursuit for n_residual_passes; (d) solve target ≈ Σ wᵢ · catalog[shortlist] s.t. wᵢ ≥ 0 (NNLS); (e) rank by w_nnls + 1e-4·cos so NNLS dominates and cosine is a pure tie-breaker.\n\nNOTE: this is *inspired by* TRAK's centering step (Park et al. 2023) but is NOT the full MadryLab/trak library. TRAK computes per-training-point gradient features via JVP, random-projects, and applies EKFAC preconditioning — that's the right tool for attributing *model behavior* to training data. CASCADE-V attributes a target audio mix to source stems — a content counterfactual, not a parameter counterfactual — so fast retrieval in a learned embedding space + NNLS sparse decomposition is the right Stage 1.",
    why:
      "Stages 2 and 3 are O(K²) and up to O(2^K). At a 5k–3M-source catalog you must pre-filter to ~K=40 to make later stages tractable. NNLS-first ranking matches the 'few real contributors' prior — w_nnls is naturally sparse, so contributors with non-zero NNLS weight float to the top automatically. The earlier 'cosine + 0.05·NNLS' formulation made cosine decisive on ties (most of the catalog) and produced a long tail of false positives ranked by raw cosine.",
    how: [
      "if USE_BPM_KEY_FILTER:",
      "    drop entries where |bpm - target_bpm| > 3 OR key not compatible",
      "scores = catalog_embeddings @ target          // (N,) raw cosines",
      "baseline = scores[random_subset].mean()        // centering",
      "w_probe = nnls(top-shortlist, target)         // pre-NNLS probe",
      "if (w_probe > 0).sum() >= 5:  n_residual_passes = 0  // skip residuals",
      "for pass_i in range(effective_residual_passes + 1):",
      "    residual subtracts top-3 contribution; re-rank",
      "w_nnls = nnls(catalog[shortlist], target)     // final NNLS",
      "rank = w_nnls + 1e-4 * (cos - baseline)        // NNLS-first",
      "return top-K of rank",
    ],
    inputs: "(N, D) catalog · (D,) target · K=40 default · optional (catalog_bpms, catalog_keys)",
    outputs:
      "Top-K candidate indices + ranking scores + NNLS metadata (n_active, n_residual_passes_used, n_filtered_bpm, n_filtered_key); validated for monotone-decreasing scores and unique indices.",
    hyperparams: [
      {
        name: "TRIAGE_TOP_K",
        value: "40",
        explain:
          "Number of candidates passed to Stage 2. Default 40 for the 5k catalog. Must exceed the realistic contributor count (3–6 in our test mixes, with encoder-rank slack), while staying small enough that no Stage-2 cluster exceeds SHAPLEY_MAX_EXACT_N=10 (so exact Shapley runs in most clusters). Below 20 you start missing real contributors at scale; above ~150 admits noise that dilutes cluster weights.",
      },
      {
        name: "USE_BPM_KEY_FILTER",
        value: "False",
        explain:
          "Pre-filter catalog by BPM (±BPM_FILTER_TOLERANCE=3) and harmonic key compatibility (equal / relative-major-minor / parallel-major-minor). Major false-positive reducer when catalog metadata is reliable. No-op when BPM/key are missing — falls back to cosine retrieval over the full catalog.",
      },
      {
        name: "n_residual_passes",
        value: "2 (auto-skipped to 0 if NNLS probe finds ≥5 actives)",
        explain:
          "Matching-pursuit-style residual passes that subtract the top-3 contribution and re-rank to expose secondary contributors. Was always-on in v0.1; now skipped when NNLS already finds enough actives, because residual passes also suppress legitimate same-creator near-neighbors.",
      },
    ],
    references: [
      {
        label: "TRAK: Attributing Model Behavior at Scale",
        href: "https://arxiv.org/abs/2303.14186",
        authors: "Park, Georgiev, Ilyas, Leclerc, Madry",
        year: 2023,
      },
      {
        label: "Influence Functions for Black-Box Predictions",
        href: "https://arxiv.org/abs/1703.04730",
        authors: "Koh, Liang",
        year: 2017,
      },
    ],
    iconBg: "bg-gradient-to-br from-rose-400 to-orange-500",
  },

  stage2: {
    number: "STEP 2",
    title: "Group — Ward / HDBSCAN clustering + group counterfactual + sparsify",
    oneLiner:
      "Cluster similar candidates, then measure each cluster's causal contribution by dropping it from the value function. Sparsify clusters below MIN_CLUSTER_WEIGHT before Stage 3.",
    what:
      "Cluster the K candidates with Ward (default, SMT-stable) or HDBSCAN (CLUSTERING_METHOD=hdbscan, density-adaptive, no distance-threshold knob, handles outliers as their own group). For each cluster C, compute v(N) − v(N\\C) — the value drop when the entire cluster is removed. Normalize, then sparsify clusters whose normalized weight falls below MIN_CLUSTER_WEIGHT=0.07 (false-positive control) and renormalize the survivors so Σ = 1.\n\nNOTE on libraries: Ward clustering is real off-the-shelf scipy (Ward 1963, fcluster). HDBSCAN is the `hdbscan` package (Campello et al. 2013) — noise points (-1) are promoted to singleton clusters so the partition stays complete. The group counterfactual itself is a few lines built on the cosine-of-mean value function — not an implementation of a specific named paper.",
    why:
      "Two stems by the same producer cluster in embedding space. Pure instance-level Shapley splits credit between them, which is correct at the source level and wrong at the creator level. Group-level counterfactuals assign one weight to the cluster, then Stage 3 splits within — the creator-DNA fix. Sparsification at 7% kills the long tail of weak-counterfactual clusters whose members were ranked into the top-K only by encoder noise.",
    how: [
      "if CLUSTERING_METHOD == 'hdbscan':",
      "    labels = hdbscan(min_cluster_size=2).fit_predict(candidates)",
      "else:  # ward (default)",
      "    Z = linkage(pdist(candidates), method='ward')",
      "    labels = fcluster(Z, t=0.6)              // cut threshold",
      "for each cluster C in labels:",
      "    raw_score[C] = v(all) - v(all minus C)",
      "normalized = clip(raw_score, 0, ∞) / sum(...)",
      "sparsified = where(normalized >= MIN_CLUSTER_WEIGHT, normalized, 0) / sum",
    ],
    inputs: "(K, D) candidate embeddings · target embedding · value-function temperature",
    outputs:
      "Cluster labels per candidate + per-cluster weight (sums to 1.0 over surviving clusters); validated for partition completeness and non-negative weights.",
    hyperparams: [
      {
        name: "CLUSTERING_METHOD",
        value: "ward",
        explain:
          "ward = SMT-stable hierarchical clustering with the CLUSTER_DISTANCE_THRESHOLD knob. hdbscan = density-adaptive, no threshold needed, better for variable cluster sizes / outliers. HDBSCAN is the recommended switch on real catalogs.",
      },
      {
        name: "CLUSTER_DISTANCE_THRESHOLD",
        value: "0.6",
        explain:
          "Ward-linkage cut threshold (only used when CLUSTERING_METHOD=ward). Lower → more, smaller clusters. 0.6 corresponds to cosine ≈ 0.82 boundary.",
      },
      {
        name: "MIN_CLUSTER_WEIGHT",
        value: "0.07",
        explain:
          "Tightened from 0.03 to 0.07. Drops clusters whose normalized counterfactual weight is below 7% before Stage 3, then renormalizes so efficiency holds. Suppresses the long tail of weak contributors that earlier inflated receipts to 13+ creators when ground truth was 3-6.",
      },
    ],
    references: [
      {
        label: "Understanding Black-box Predictions via Influence Functions (group influence section)",
        href: "https://arxiv.org/abs/1703.04730",
        authors: "Koh, Liang",
        year: 2017,
      },
      {
        label: "On the Accuracy of Influence Functions for Measuring Group Effects",
        href: "https://arxiv.org/abs/1905.13289",
        authors: "Koh, Ang, Teo, Liang",
        year: 2019,
      },
      {
        label: "Hierarchical grouping to optimize an objective function (Ward linkage)",
        href: "https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500845",
        authors: "Ward",
        year: 1963,
      },
      {
        label: "scipy.cluster.hierarchy — Ward linkage implementation we actually use",
        href: "https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html",
      },
    ],
    iconBg: "bg-gradient-to-br from-violet-500 to-indigo-600",
  },

  stage3: {
    number: "STEP 3",
    title: "Shapley — exact 2ⁿ or antithetic + stratified MC with Hoeffding bounds",
    oneLiner:
      "Within each cluster, compute exact Shapley values when feasible, otherwise antithetic + stratified Monte Carlo with Hoeffding-bounded intervals.",
    what:
      "For each cluster of size n ≤ SHAPLEY_MAX_EXACT_N=10: exact φᵢ = Σ_{S⊆N\\{i}} |S|!(n−|S|−1)!/n! · [v(S∪{i}) − v(S)] (O(2ⁿ)). For larger clusters: variance-reduced MC. Two tricks on top of standard permutation MC: (1) **antithetic sampling** — for each random permutation π also evaluate its reverse π'; the marginals are negatively correlated, so variance roughly halves at the same wall-clock budget; (2) **prefix-length stratification** — bin the permutation budget into √T strata so coalition sizes spread evenly across the budget. Together: ~3–5× variance reduction at equal sample count (Kolpaczki et al. 2024). The Hoeffding bound remains a valid (no longer tight) upper bound on |φ̂ᵢ − φᵢ|.\n\nValue function: v(S) = max(0, cos(mean(emb_S), target))^(T/4). Bounded in [0,1], non-negative cosine clamps to 0 (anti-aligned coalition contributes nothing), monotone-friendly. The exponent T/4 sharpens concentration on coalitions that genuinely point toward the target.\n\nNOTE on libraries: the Shapley computation is a from-scratch implementation of the published formulas (~60 lines total). We do NOT use SHAP (Lundberg) or Captum.ShapleyValueSampling because those target ML feature attribution against a black-box predictor — they don't accept arbitrary cooperative-game value functions like ours. From-scratch is also better for auditability.",
    why:
      "Shapley is the unique value function satisfying the four fairness axioms (efficiency, symmetry, dummy, additivity). Any other split satisfying all four reduces to Shapley. Antithetic + stratification cuts MC variance without breaking the Hoeffding bound; the exponent on cosine sharpens the value function so weak contributors don't get spurious mass.",
    how: [
      "if n ≤ SHAPLEY_MAX_EXACT_N:",
      "    iterate over all 2^n subsets, sum weighted marginal contributions",
      "else:",
      "    n_strata = √(n_permutations / 2)",
      "    for stratum in n_strata:",
      "        for _ in pairs_per_stratum:",
      "            π = rand_perm; accumulate(π)",
      "            accumulate(π[::-1])           // antithetic counterpart",
      "    ε = 2 · √(ln(2/α) / (2 · n_marginals))    // Hoeffding (R=2)",
      "    intervals = [φ̂ - ε, φ̂ + ε]",
    ],
    inputs: "Cluster member indices · candidate embeddings · target embedding · temperature",
    outputs:
      "Per-source Shapley values + normalized weights summing to 1.0 + [lower, upper] confidence intervals + metadata {antithetic, n_strata, pairs_per_stratum}.",
    hyperparams: [
      {
        name: "SHAPLEY_MAX_EXACT_N",
        value: "10",
        explain:
          "Cluster sizes ≤ 10 use exact 2^n enumeration (1024 evaluations). Beyond, fall back to antithetic + stratified MC.",
      },
      {
        name: "SHAPLEY_MC_PERMUTATIONS",
        value: "400",
        explain:
          "Permutation budget. With antithetic doubling and √T stratification, variance ≈ 3-5× lower than naive MC at the same budget. ε ≈ 2·√(ln(2/0.05) / 800) ≈ 0.137 at 95% confidence (Hoeffding upper bound, conservative under variance reduction).",
      },
      {
        name: "SHAPLEY_VALUE_TEMPERATURE",
        value: "8.0",
        explain:
          "Sharpens v(S) = max(0, cos(mean(emb_S), target))^(T/4). Sharpened from 4.0 to 8.0 to suppress the long tail of weak contributors. Higher T → tighter concentration on coalitions that strongly point toward the target.",
      },
      {
        name: "HOEFFDING_CONFIDENCE",
        value: "0.95",
        explain: "Confidence level for Hoeffding-bounded intervals.",
      },
    ],
    references: [
      {
        label: "A value for n-person games",
        href: "https://www.rand.org/pubs/papers/P295.html",
        authors: "Shapley",
        year: 1953,
      },
      {
        label: "Values of games with a priori unions (Owen value — optional, OWEN_OR_DECOMP=owen)",
        href: "https://link.springer.com/chapter/10.1007/978-3-642-45494-3_7",
        authors: "Owen",
        year: 1977,
      },
      {
        label: "Probability inequalities for sums of bounded random variables",
        href: "https://www.jstor.org/stable/2282952",
        authors: "Hoeffding",
        year: 1963,
      },
      {
        label: "Polynomial calculation of the Shapley value based on sampling",
        href: "https://www.sciencedirect.com/science/article/pii/S0305054808000804",
        authors: "Castro, Gómez, Tejada",
        year: 2009,
      },
    ],
    iconBg: "bg-gradient-to-br from-emerald-400 to-teal-500",
  },

  compose: {
    number: "STEP 4",
    title: "Compose — interval arithmetic + final sparsification",
    oneLiner:
      "Multiply cluster_weight × within_cluster_shapley with mpmath rounding-aware intervals, then sparsify final weights below MIN_FINAL_WEIGHT=0.04.",
    what:
      "Each per-source weight is composed as cluster_weight · shapley_within_cluster, with both factors carrying [lower, upper] intervals. mpmath at 80-bit precision propagates the intervals through multiplication, addition, and renormalization with conservative rounding so the auditor's interval contains the true value. After composition, sources whose final weight falls below MIN_FINAL_WEIGHT=0.04 are zeroed and the remaining weights renormalized — efficiency holds across the renorm. This is the second of two false-positive sparsifications (the first happens at the cluster level in Stage 2).\n\nWith Owen value (OWEN_OR_DECOMP=owen) Stage 3 directly emits per-source weights for the active subset; the heuristic cluster_weight × within_cluster_shapley path is replaced by an Owen-value lookup.",
    why:
      "Float64 is fine for a single multiplication but errors compound across multiple stages. The patent claim (US 12,536,365 B1) is partly about provably-correct interval propagation: the proof certificate must contain an interval the auditor can independently verify. Final sparsification at 4% is what concentrates payouts on real contributors and keeps receipts to ~5 creators rather than the 13+ that uncontrolled value-function tails produce.",
    how: [
      "for each candidate i in cluster C:",
      "    final[i] = Interval(c_w) * Interval(shapley[i])    // mpmath rounding-aware",
      "renormalize so Σ final[i] = 1.0",
      "below = final_weights < MIN_FINAL_WEIGHT",
      "final_weights[below] = 0; renormalize survivors",
    ],
    inputs:
      "Cluster weights (from Stage 2) · per-cluster Shapley intervals (from Stage 3)",
    outputs:
      "(K, 2) per-candidate [lower, upper] intervals + (K,) point estimates summing to 1.0; sparsified-to-zero entries excluded from the active set passed to Stage V and the currencies layer.",
    hyperparams: [
      {
        name: "INTERVAL_PRECISION_BITS",
        value: "80",
        explain:
          "mpmath precision in bits. 80 bits ≈ 24 decimal digits — well beyond float64 (≈ 15 digits).",
      },
      {
        name: "MIN_FINAL_WEIGHT",
        value: "0.04",
        explain:
          "Tightened from 0.015 to 0.04. After cluster_weight × within_cluster_shapley composition, sources below 4% are zeroed and the rest renormalized so Σ = 1. Concentrates payouts on real contributors; suppresses the long tail.",
      },
    ],
    references: [
      {
        label: "Validated Numerics: A Short Introduction",
        href: "https://www.cambridge.org/core/books/validated-numerics/0AC7A1A3F1F4D4F2E0C7B7D5E5E5E5E5",
        authors: "Tucker",
        year: 2011,
      },
      {
        label: "mpmath: a Python library for arbitrary-precision floating-point",
        href: "https://mpmath.org",
      },
    ],
    iconBg: "bg-gradient-to-br from-sky-400 to-blue-600",
  },

  stageV: {
    number: "STEP 5",
    title: "Verify — Z3 SMT proof of fairness axioms",
    oneLiner:
      "Re-check the actual numerical weights against the three Shapley fairness axioms with the Z3 SMT solver.",
    what:
      "Encode the per-source weights and the three axioms (efficiency, symmetry, null player / dummy) as constraints in QF_LRA SMT-LIB. Z3 checks them against the actual values; the SMT-LIB file is written to disk with a sha-256 hash. Anyone with z3 installed can run the same check independently — no need to trust our pipeline.",
    why:
      "Per-instance proofs are what regulators and creators actually want for audit. Even if the algorithm is correct in theory, every receipt should carry a separately verifiable certificate. This is the patent contribution: the Cascade Validation Protocol.",
    how: [
      "for each axiom (efficiency, symmetry, dummy):",
      "    encode constraints over (w_0..w_n) as QF_LRA assertions",
      "    s.check() == sat → axiom holds within tolerance",
      "write .smt2 file → sha256 hash → embed in receipt JSON",
    ],
    inputs:
      "Per-source weights · candidate embeddings (for symmetry detection) · raw scores (for null-player detection)",
    outputs:
      "ProofCertificate{axioms: [PROVEN/VIOLATED/NA], smt_lib_file path, sha256 hash, overall_status}",
    hyperparams: [
      {
        name: "EFFICIENCY_TOLERANCE",
        value: "1e-3",
        explain: "|Σwᵢ − 1| must be below this for efficiency to PASS.",
      },
      {
        name: "SYMMETRY_TOLERANCE",
        value: "1e-3",
        explain:
          "Two near-identical sources (cosine ≥ 0.985) must have weights within this tolerance. Cosine threshold tightened from 0.99 → 0.985 to catch near-duplicate same-creator stems that previously escaped the symmetry check. Note: the Demucs-aggregate path can produce asymmetric final weights (different Demucs stems give different per-source weights to the same catalog item), which is why the symmetry axiom is the most-violated of the three.",
      },
      {
        name: "DUMMY_TOLERANCE",
        value: "1e-3",
        explain: "Null sources (raw score < 1e-3) must have weight below this.",
      },
    ],
    references: [
      {
        label: "Z3: An Efficient SMT Solver",
        href: "https://link.springer.com/chapter/10.1007/978-3-540-78800-3_24",
        authors: "de Moura, Bjørner",
        year: 2008,
      },
      {
        label: "SMT-LIB Standard v2.6",
        href: "https://smt-lib.org",
      },
      {
        label: "US Patent 12,536,365 B1 — Cascade Validation Protocol",
        href: "https://patents.google.com/patent/US12536365B1",
        year: 2026,
      },
    ],
    iconBg: "bg-gradient-to-br from-amber-400 to-orange-500",
  },

  currencies: {
    number: "STEP 6",
    title: "Currencies — QWS + three-currency receipt with α/β/γ dials",
    oneLiner:
      "Quantile-weighted Shapley + Rawlsian floor monetary + role-tag recognition + opportunity-routing, with explicit fairness/appreciation tradeoff dials.",
    what:
      "Pure-numpy extension to the receipt path. Operates only on the active set (post-Stage-V sparsification) so the Rawlsian floor doesn't re-inflate suppressed false positives. Three layers:\n\n• **QWS (Quantile-Weighted Shapley)** — estimates the output's empirical quality q ∈ [0, 1] (centroid distance proxy by default; empirical CDF when historical data is provided), then shapes the weights at quantile q: q ≤ 0.5 pulls toward uniform (mediocre outputs split credit flatly), q > 0.5 sharpens via power-law (excellent outputs reward top contributors).\n• **Currency 1 — Money (α)**: Rawlsian maximin floor lifts every share above 0.8(1-α)/n. Excess is taken from above-floor entries proportional to their excess. Invariant proven: total excess (1 - n·floor) ≥ 0.2 always exceeds total deficit, so the floor is honored and Σ wᵢ = 1 exactly.\n• **Currency 2 — Recognition (β)**: top-8 target embedding dimensions assign symbolic role tags (RHYTHMIC ANCHOR, HARMONIC FOUNDATION, MELODIC LEAD, …). Counterfactual loss > 0.20 → ESSENTIAL CONTRIBUTOR badge. Every contributor with positive weight gets ≥1 role. Lottery feature picks one contributor per output via P(featured) ∝ β·wᵢ + (1-β)/n (Friedman-Savage utility: small probability of large recognition).\n• **Currency 3 — Opportunity (γ)**: priority adjustment for future Stage-1 triage routing — quality-recency bonus (tapered by 1-γ), diversity bonus (gated by γ), under-representation protection (gated by γ). All terms non-negative ⇒ adjustment ≥ 0 always.\n\nA dignity meta-proof verifies all three currencies were honored for every contributor; recorded under `dignity_proof` in the receipt JSON alongside the Z3 SMT proof.",
    why:
      "Money alone doesn't capture the felt-appreciation that drives long-term creator participation. Sen's capability approach (1985) argues capability includes both functioning and freedom; we operationalize that with three currencies that are independently dialed: α for fairness strictness, β for recognition spread, γ for opportunity redistribution. All dials live in [0, 1] and are recorded per-receipt — they're public, not hidden. Rawlsian flooring (Rawls 1971) is the standard fairness constraint that maximizes the position of the worst-off contributor while preserving relative ordering.",
    how: [
      "// Active subset only (base_weight > 0)",
      "q = estimate_output_quality(target, catalog)        // ∈ [0, 1]",
      "weights_qws = quantile_weighted_shapley(base, q)    // shape by quantile",
      "// Currency 1",
      "floor = 0.8 · (1 - α) / n_active",
      "weights_with_floor = lift_below(weights_qws, floor)  // Σ = 1 preserved",
      "// Currency 2",
      "roles = assign_roles(weights, embeddings, target, counterfactuals)",
      "featured = sample_lottery(weights, β)",
      "// Currency 3",
      "adjust = quality_bonus·(1-γ) + γ·(diversity + under_rep)",
      "dignity_proof = verify_all_three_currencies_honored(...)",
    ],
    inputs:
      "CascadeVResult (post-Stage-V) + candidate_embeddings + target_embedding + dials (α, β, γ) + optional contributor_history + optional quality_reference",
    outputs:
      "TripleCurrencyReceipt with `dials`, `qws`, `monetary`, `reputational`, `opportunity`, and `dignity_proof` blocks. Attached to receipt JSON under the `currencies` key when ENABLE_CURRENCIES=True.",
    hyperparams: [
      {
        name: "ENABLE_CURRENCIES",
        value: "True",
        explain:
          "Whether the currencies layer is run and attached to receipts. Set to False to ship a Shapley-only receipt with no currency block.",
      },
      {
        name: "CURRENCY_ALPHA",
        value: "0.85",
        explain:
          "Fairness strictness on monetary payouts. 1.0 = pure Shapley (no floor), 0.0 = full equal split. 0.85 ⇒ floor ≈ 0.12/n_active.",
      },
      {
        name: "CURRENCY_BETA",
        value: "0.60",
        explain:
          "Recognition spread on the lottery feature. 1.0 = P(featured) ∝ weight (top wins most), 0.0 = uniform across contributors (egalitarian).",
      },
      {
        name: "CURRENCY_GAMMA",
        value: "0.30",
        explain:
          "Opportunity redistribution for future-triage routing. 0.0 = pure merit (only quality-recency bonus fires), 1.0 = strong protection (diversity + under-representation bonuses fully active).",
      },
    ],
    references: [
      {
        label: "Regression Quantiles (the Q in QWS)",
        href: "https://www.jstor.org/stable/1913643",
        authors: "Koenker, Bassett",
        year: 1978,
      },
      {
        label: "A Theory of Justice (Rawlsian maximin / dignity floor)",
        href: "https://www.hup.harvard.edu/books/9780674000780",
        authors: "Rawls",
        year: 1971,
      },
      {
        label: "Commodities and Capabilities (Sen's three-currency framing)",
        href: "https://global.oup.com/academic/product/commodities-and-capabilities-9780195650389",
        authors: "Sen",
        year: 1985,
      },
      {
        label: "The Utility Analysis of Choices Involving Risk (Friedman-Savage utility, lottery feature)",
        href: "https://www.jstor.org/stable/1826045",
        authors: "Friedman, Savage",
        year: 1948,
      },
    ],
    iconBg: "bg-gradient-to-br from-pink-400 to-rose-500",
  },
};

export const STAGE_ICONS: Record<StageKey, string> = {
  overview: "ⓘ",
  stage0: "✂",
  stage1: "◎",
  stage2: "◫",
  stage3: "Σ",
  compose: "⏚",
  stageV: "✓",
  currencies: "★",
};

export default function StageInfoModal({
  stage,
  onClose,
}: {
  stage: StageKey;
  onClose: () => void;
}) {
  const info = STAGES[stage];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-slate-900/50 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="card mt-12 w-full max-w-3xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-4 border-b border-border p-5">
          <div className={`step-icon shrink-0 ${info.iconBg}`}>
            {STAGE_ICONS[stage]}
          </div>
          <div className="flex-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-subt">
              {info.number}
            </div>
            <h2 className="text-xl font-bold leading-tight">{info.title}</h2>
            <p className="mt-1 text-sm text-subt">{info.oneLiner}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-full p-1 text-slate-500 hover:bg-slate-200 hover:text-slate-800"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4 p-5">
          <Section title="What it does">
            <p className="text-sm leading-relaxed">{info.what}</p>
          </Section>

          <Section title="Why this approach">
            <p className="text-sm leading-relaxed">{info.why}</p>
          </Section>

          <Section title="How (algorithm sketch)">
            <pre className="overflow-x-auto rounded-md border border-border bg-slate-50 p-3 font-mono text-[11px] leading-relaxed">
              {info.how.join("\n")}
            </pre>
          </Section>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Section title="Inputs">
              <p className="text-sm leading-relaxed">{info.inputs}</p>
            </Section>
            <Section title="Outputs">
              <p className="text-sm leading-relaxed">{info.outputs}</p>
            </Section>
          </div>

          {info.hyperparams && (
            <Section title="Hyperparameters">
              <ul className="flex flex-col gap-2">
                {info.hyperparams.map((h) => (
                  <li
                    key={h.name}
                    className="rounded-md border border-border bg-slate-50 p-2.5"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <code className="font-mono text-xs font-semibold">
                        {h.name}
                      </code>
                      <code className="font-mono text-xs text-violet-700">
                        {h.value}
                      </code>
                    </div>
                    <div className="mt-1 text-xs leading-snug text-subt">
                      {h.explain}
                    </div>
                  </li>
                ))}
              </ul>
            </Section>
          )}

          <Section title="References">
            <ul className="flex flex-col gap-1.5">
              {info.references.map((r) => (
                <li key={r.href}>
                  <a
                    href={r.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex flex-col rounded-md border border-border p-2.5 hover:border-violet-300 hover:bg-violet-50"
                  >
                    <span className="text-sm font-medium text-violet-700 group-hover:underline">
                      {r.label} ↗
                    </span>
                    {(r.authors || r.year) && (
                      <span className="text-[11px] text-subt">
                        {r.authors}
                        {r.authors && r.year ? " · " : ""}
                        {r.year}
                      </span>
                    )}
                  </a>
                </li>
              ))}
            </ul>
          </Section>
        </div>

        <div className="flex justify-end gap-2 border-t border-border p-4">
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-subt">
        {title}
      </div>
      {children}
    </div>
  );
}
