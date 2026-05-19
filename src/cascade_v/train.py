"""
train.py — Training loop for the audio encoder.

Contrastive learning (SimCLR-style) on the catalog stems. For each stem,
generate two random augmentations per training step; the loss pulls the
two augmented versions of the same stem together and pushes them apart
from other stems in the batch.

Real training, runs on Apple MPS / CUDA / CPU. Saves checkpoint and
training metrics to disk.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cascade_v.config import (
    AUGMENTATIONS_PER_SAMPLE,
    CONTRASTIVE_TEMPERATURE,
    DEVICE,
    ENCODER_CHECKPOINT,
    GLOBAL_SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
    TRAINING_LOG,
)
from cascade_v.encoder import AudioEncoder, nt_xent_loss
from cascade_v.logging_setup import event, get_logger
from cascade_v.utils.audio import audio_to_melspec, augment_audio, fix_length, load_wav
from cascade_v.utils.determinism import set_global_seeds


_log = get_logger("train")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AugmentedAudioDataset(Dataset):
    """
    Per item we return:
      - m1, m2: two augmented mel-specs of the SAME stem (NT-Xent positive pair)
      - mix:    a nonlinear mix of N partners (N drawn from {2,3,4})
      - partners: padded list of mel-specs of the partner stems (length MAX_PARTNERS)
      - weights: padded list of mixing weights (length MAX_PARTNERS)
      - n_partners: actual count (for masking)

    The mix-consistency loss aligns emb(mix) with the weighted normalized sum of
    partner embeddings — directly the additive structure used at inference time
    by the value function v(S) = exp(-||mean(emb_S) - target||/T).
    """

    # Lowered 4 → 2 for the 5k × 10s catalog: each batch step encodes
    # (m1, m2, mix, MAX_PARTNERS partners) mel-specs, so MAX_PARTNERS
    # dominates the per-batch compute. 2 partners still gives a non-trivial
    # mix-consistency signal (the loss is averaged over partners, so its
    # gradient direction is preserved).
    MAX_PARTNERS = 2

    def __init__(self, audio_paths: list[Path], seed: int = GLOBAL_SEED):
        self.audio_paths = audio_paths
        self.audios = [fix_length(load_wav(p)) for p in audio_paths]
        self.rngs = [
            np.random.default_rng(seed + i) for i in range(len(audio_paths))
        ]
        self.master_rng = np.random.default_rng(seed + 9999)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> dict:
        audio = self.audios[idx]
        rng = self.rngs[idx]
        a1 = augment_audio(audio, rng)
        a2 = augment_audio(audio, rng)

        # Sample N partners (N ∈ {2,3,4}) — including this stem with prob 0.5
        n = int(self.master_rng.integers(2, self.MAX_PARTNERS + 1))
        partner_indices = list(self.master_rng.choice(
            len(self.audios), size=n, replace=False
        ))
        if self.master_rng.random() < 0.5 and idx not in partner_indices:
            partner_indices[0] = idx  # bias for inclusion of anchor

        partner_audios = [self.audios[i] for i in partner_indices]
        weights = self.master_rng.dirichlet(np.ones(n)).astype(np.float32)
        # Avoid pathologically tiny weights — clip and renormalize
        weights = np.clip(weights, 0.05, None)
        weights = weights / weights.sum()
        mix_audio = _multi_mix(partner_audios, weights)

        # Pad partners + weights to MAX_PARTNERS for batch collation
        pad_n = self.MAX_PARTNERS - n
        partner_mels = [audio_to_melspec(a) for a in partner_audios]
        if pad_n > 0:
            zero_mel = torch.zeros_like(partner_mels[0])
            partner_mels = partner_mels + [zero_mel] * pad_n
            weights = np.concatenate([weights, np.zeros(pad_n, dtype=np.float32)])

        return {
            "m1": audio_to_melspec(a1),
            "m2": audio_to_melspec(a2),
            "mix": audio_to_melspec(mix_audio),
            "partners": torch.stack(partner_mels, dim=0),  # (P, n_mels, T)
            "weights": torch.from_numpy(weights),          # (P,)
            "n_partners": torch.tensor(n, dtype=torch.long),
        }


def _multi_mix(audios: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted sum + soft saturation (mirrors the test-time nonlinear mixer)."""
    n = max(len(a) for a in audios)
    stacked = np.stack([
        np.pad(a, (0, n - len(a))) if len(a) < n else a[:n] for a in audios
    ])
    mix = (stacked * weights[:, None]).sum(axis=0)
    mix = np.tanh(mix * 1.3)
    peak = float(np.max(np.abs(mix))) + 1e-9
    return (mix / peak * 0.85).astype(np.float32)


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    lr: float
    elapsed_sec: float
    # Per-loss breakdown
    contrastive_loss: float = 0.0
    mix_loss: float = 0.0
    # Embedding-quality diagnostics (Wang & Isola 2020):
    #   alignment = E[||emb_a - emb_b||^2] for positive pairs (lower is better)
    #   uniformity = log E[exp(-2 ||emb_i - emb_j||^2)] over negatives (lower is better)
    align: float = 0.0
    uniformity: float = 0.0
    # Mix-consistency cosine similarity (1.0 = perfect additivity)
    mix_cosine: float = 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_encoder(
    audio_paths: list[Path],
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = TRAIN_LR,
    weight_decay: float = TRAIN_WEIGHT_DECAY,
    temperature: float = CONTRASTIVE_TEMPERATURE,
    mix_loss_weight: float = 0.5,
    device: torch.device = DEVICE,
    checkpoint_path: Path = ENCODER_CHECKPOINT,
    log_path: Path = TRAINING_LOG,
    seed: int = GLOBAL_SEED,
    verbose: bool = True,
) -> tuple[AudioEncoder, list[EpochMetrics]]:
    """
    Train the audio encoder via contrastive learning.

    Returns the trained encoder and per-epoch metrics.
    """
    set_global_seeds(seed)

    dataset = AugmentedAudioDataset(audio_paths)
    # Use small batch size relative to dataset; if dataset < batch_size, drop_last must be False
    effective_batch = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,           # keep simple for cross-platform
        drop_last=len(dataset) > effective_batch,
    )

    model = AudioEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    metrics: list[EpochMetrics] = []
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    event(_log, "train.start",
          device=str(device), dataset=len(dataset), batch=effective_batch,
          epochs=epochs, params=n_params)

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_contrast = 0.0
        epoch_mix = 0.0
        epoch_align = 0.0
        epoch_unif = 0.0
        epoch_mix_cos = 0.0
        n_batches = 0

        for batch in loader:
            m1 = batch["m1"].to(device)
            m2 = batch["m2"].to(device)
            mix = batch["mix"].to(device)
            partners = batch["partners"].to(device)   # (B, P, n_mels, T)
            weights = batch["weights"].to(device)     # (B, P)

            B, P = partners.shape[:2]

            e1 = model(m1)              # (B, D)
            e2 = model(m2)              # (B, D)
            e_mix = model(mix)          # (B, D)
            # Encode all partners in one shot
            e_partners = model(partners.view(B * P, *partners.shape[2:]))
            e_partners = e_partners.view(B, P, -1)  # (B, P, D)

            # Contrastive loss
            l_contrast = nt_xent_loss(e1, e2, temperature=temperature)

            # Multi-source mix consistency:
            #   emb(mix(s_1..s_n; w)) ≈ normalize(Σ w_i emb(s_i))
            # weighted sum across partners
            weighted = (e_partners * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
            target_mix = torch.nn.functional.normalize(weighted, p=2, dim=-1)
            l_mix = (1.0 - (e_mix * target_mix).sum(dim=-1)).mean()

            loss = l_contrast + mix_loss_weight * l_mix

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Alignment (positive pairs) and uniformity (negative pairs)
                #   Wang & Isola, "Understanding Contrastive Representation
                #   Learning through Alignment and Uniformity on the Hypersphere"
                # alignment: E[||a - b||^2] over positive pairs (lower better)
                align = (e1 - e2).pow(2).sum(dim=-1).mean()
                # uniformity: log E[exp(-2 ||a - b||^2)] over all pairs (lower better)
                if e1.shape[0] > 1:
                    sq_dists = torch.cdist(e1, e1).pow(2)
                    mask = ~torch.eye(e1.shape[0], device=e1.device, dtype=torch.bool)
                    unif = sq_dists[mask].mul(-2.0).exp().mean().log()
                else:
                    unif = torch.tensor(0.0, device=e1.device)
                # mix-cosine fidelity
                mix_cos = (e_mix * target_mix).sum(dim=-1).mean()

            epoch_loss += loss.item()
            epoch_contrast += l_contrast.item()
            epoch_mix += l_mix.item()
            epoch_align += float(align.item())
            epoch_unif += float(unif.item())
            epoch_mix_cos += float(mix_cos.item())
            n_batches += 1

        scheduler.step()
        nb = max(n_batches, 1)
        avg_loss = epoch_loss / nb
        elapsed = time.time() - epoch_start
        m = EpochMetrics(
            epoch=epoch,
            loss=avg_loss,
            lr=scheduler.get_last_lr()[0],
            elapsed_sec=elapsed,
            contrastive_loss=epoch_contrast / nb,
            mix_loss=epoch_mix / nb,
            align=epoch_align / nb,
            uniformity=epoch_unif / nb,
            mix_cosine=epoch_mix_cos / nb,
        )
        metrics.append(m)

        event(_log, "train.epoch",
              epoch=epoch + 1, total=epochs,
              loss=round(avg_loss, 6),
              contrastive=round(m.contrastive_loss, 4),
              mix=round(m.mix_loss, 4),
              mix_cos=round(m.mix_cosine, 4),
              align=round(m.align, 4),
              unif=round(m.uniformity, 4),
              lr=m.lr, elapsed_sec=round(elapsed, 2))

        # Persist training log incrementally so the dashboard can poll it
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump([mm.__dict__ for mm in metrics], f, indent=2)

        # Per-epoch checkpoint: lets us stop training at any epoch and still
        # have a usable encoder on disk. Overwritten each epoch so we always
        # have the latest weights.
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "epochs_run": epoch + 1,
                "epochs_target": epochs,
                "batch_size": effective_batch,
                "lr": lr,
                "weight_decay": weight_decay,
                "temperature": temperature,
            },
        }, checkpoint_path)

    # Final save (redundant with the per-epoch save above, but kept for clarity)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "epochs": epochs,
            "batch_size": effective_batch,
            "lr": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
        },
    }, checkpoint_path)

    # Final write of metrics log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump([m.__dict__ for m in metrics], f, indent=2)

    event(_log, "train.done",
          checkpoint=str(checkpoint_path), final_loss=round(metrics[-1].loss, 6))

    return model, metrics


def load_encoder(checkpoint_path: Path = ENCODER_CHECKPOINT, device: torch.device = DEVICE):
    """Load the configured encoder.

    Dispatches on ENCODER_KIND:
      - "custom"    → 1.4M-param ResCNN trained from scratch (legacy default)
      - "clap"      → frozen LAION-CLAP foundation model, no training
      - "clap_proj" → frozen CLAP backbone + trainable MLP projection head
                      (Workstream 2: small head trained with creator-conditioned
                      contrastive loss; checkpoint is just the head weights)
    """
    from cascade_v.config import ENCODER_KIND

    if ENCODER_KIND == "clap":
        from cascade_v.encoders.clap_encoder import load_clap_encoder
        enc = load_clap_encoder()
        try:
            enc.model.to(device)
        except Exception:
            pass
        return enc

    if ENCODER_KIND == "clap_proj":
        from cascade_v.encoders.clap_projection import load_clap_projection_encoder
        # Head checkpoint lives next to the legacy encoder.pt — reuse the
        # same MODELS_DIR slot but under a distinct filename.
        head_ckpt = checkpoint_path.parent / "clap_proj_head.pt"
        enc = load_clap_projection_encoder(head_ckpt if head_ckpt.exists() else None)
        try:
            enc.backbone.model.to(device)
            enc.head.to(device)
        except Exception:
            pass
        return enc

    # Default: custom checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run cascade-train first, "
            "or set ENCODER_KIND=clap or ENCODER_KIND=clap_proj to use the "
            "LAION CLAP foundation model (with optional trainable head)."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AudioEncoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Projection-head trainer (Workstream 2)
# ---------------------------------------------------------------------------
#
# Trains only the MLP head sitting on top of the frozen CLAP backbone.
# Loss: supervised contrastive (Khosla et al. 2020). Two views of the same
# stem are positives (the SimCLR pair); additionally any pair sharing a
# creator_id is treated as a positive. This is the creator-conditioned loss
# that makes creator-DNA visible in the embedding space — pure SimCLR has
# no creator signal at all and that's the dominant reason the v1 receipts
# placed only 3-4% of mass on legitimate same-creator contributors.
#
# Features are pre-computed once over K augmentations per stem (the CLAP
# backbone is frozen, so this is correct and ~100× faster than re-running
# the backbone every batch). The head trains over a (N*K, 512) tensor.

def _supcon_loss(
    features: torch.Tensor,            # (M, D), already L2-normalized
    creator_ids: list[str],            # length M
    instance_ids: list[int],           # length M (same id ⇒ same audio stem, different aug)
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss with two positive sources:
      1. Same instance (different augmentation) — the SimCLR positive
      2. Same creator_id (different audio) — the creator-DNA positive

    For each anchor i, the loss is:
        L_i = - 1/|P(i)| · Σ_{p ∈ P(i)} log( exp(s_ip/τ) / Σ_{a ≠ i} exp(s_ia/τ) )
    where P(i) = positives. This is the SupCon formulation; reduces to
    NT-Xent when |P(i)| = 1.
    """
    m = features.size(0)
    device = features.device
    sim = features @ features.T / temperature                            # (M, M)
    # Numerical stability: subtract row-max
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    self_mask = torch.eye(m, device=device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    # log Σ exp over non-self
    log_denom = torch.logsumexp(sim, dim=1, keepdim=True)               # (M, 1)

    cre = np.array(creator_ids)
    inst = np.array(instance_ids)
    pos = (cre[:, None] == cre[None, :]) | (inst[:, None] == inst[None, :])
    pos_mask = torch.tensor(pos, device=device, dtype=torch.bool) & (~self_mask)

    pos_per_anchor = pos_mask.sum(dim=1).clamp(min=1).float()
    log_prob = sim - log_denom                                          # (M, M)
    masked = log_prob.masked_fill(~pos_mask, 0.0)
    loss_per_anchor = -masked.sum(dim=1) / pos_per_anchor
    return loss_per_anchor.mean()


def _precompute_clap_features(
    audio_paths: list[Path],
    sources: list[dict],
    k_views: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """
    Run the frozen CLAP backbone once over k_views augmentations of every
    stem and return:
      features:    (N*k_views, 512) torch.Tensor (CPU; small enough to fit)
      creator_ids: length N*k_views
      instance_ids: length N*k_views (same int ⇒ same stem, different aug)
    """
    from cascade_v.encoders.clap_projection import ClapProjectionEncoder

    enc = ClapProjectionEncoder()
    enc.backbone.model.to(device)
    rngs = [np.random.default_rng(seed + i) for i in range(len(audio_paths))]

    feats_chunks: list[torch.Tensor] = []
    creators: list[str] = []
    instances: list[int] = []

    for stem_idx, (path, meta, rng) in enumerate(zip(audio_paths, sources, rngs)):
        audio = fix_length(load_wav(path))
        views = [augment_audio(audio, rng) for _ in range(k_views)]
        feats = enc._backbone_features(
            views, sample_rate=22050, device=device, batch_size=k_views,
        )
        feats_chunks.append(feats.cpu())
        creators.extend([meta["creator_id"]] * k_views)
        instances.extend([stem_idx] * k_views)

    return torch.cat(feats_chunks, dim=0), creators, instances


def train_clap_projection_head(
    audio_paths: list[Path],
    sources: list[dict],
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = TRAIN_LR,
    weight_decay: float = TRAIN_WEIGHT_DECAY,
    temperature: float = CONTRASTIVE_TEMPERATURE,
    k_views: int = AUGMENTATIONS_PER_SAMPLE,
    device: torch.device = DEVICE,
    checkpoint_path: Path | None = None,
    log_path: Path = TRAINING_LOG,
    seed: int = GLOBAL_SEED,
    min_epochs: int = 20,
    early_stop_loss_rel_improvement: float = 0.005,  # 0.5%
    early_stop_patience: int = 5,
    verbose: bool = True,
) -> tuple[object, list[EpochMetrics]]:
    """
    Train only the projection head of the ClapProjectionEncoder.

    Pre-computes CLAP backbone features once (K augmentations per stem),
    then trains the MLP head with the creator-conditioned SupCon loss.

    Convergence diagnostics tracked per epoch:
      - `loss`        — mean SupCon loss
      - `align`       — same-instance pair distance² (SimCLR alignment)
      - `creator_align` — same-creator-different-instance pair distance²
                          (the metric the SupCon loss is *actually* trying
                          to drive down; CLAP doesn't satisfy this for free)
      - `uniformity`  — log E[exp(-2·d²)] over all batch pairs

    Early-stop: triggered only after `min_epochs` epochs, when the loss
    fails to improve by more than `early_stop_loss_rel_improvement`
    relative to the best-so-far for `early_stop_patience` consecutive
    epochs. The earlier "align < threshold" criterion was wrong here —
    a frozen CLAP backbone already satisfies SimCLR alignment with no
    head training, so that test fired at epoch 1 before the head learned
    any creator-DNA discrimination.
    """
    set_global_seeds(seed)
    if checkpoint_path is None:
        checkpoint_path = ENCODER_CHECKPOINT.parent / "clap_proj_head.pt"

    from cascade_v.encoders.clap_projection import ClapProjectionEncoder

    enc = ClapProjectionEncoder()
    enc.backbone.model.to(device)
    enc.head.to(device)

    if verbose:
        print(f"[train_proj] pre-computing CLAP features: {len(audio_paths)} stems × {k_views} views …")
    feats_cpu, creator_ids, instance_ids = _precompute_clap_features(
        audio_paths, sources, k_views, device, seed,
    )
    feats = feats_cpu.to(device)                                        # (N*K, 512)

    optimizer = torch.optim.AdamW(
        enc.head.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_total = feats.size(0)
    metrics: list[EpochMetrics] = []
    rng = np.random.default_rng(seed)

    n_params = sum(p.numel() for p in enc.head.parameters())
    event(_log, "train_proj.start",
          device=str(device), n_stems=len(audio_paths), k_views=k_views,
          n_total=n_total, head_params=n_params, epochs=epochs,
          min_epochs=min_epochs,
          early_stop_loss_rel_improvement=early_stop_loss_rel_improvement,
          early_stop_patience=early_stop_patience)

    best_loss = float("inf")
    epochs_since_improvement = 0
    best_head_state: Optional[dict] = None  # snapshot of head weights at best loss

    for epoch in range(epochs):
        epoch_start = time.time()
        order = rng.permutation(n_total)
        epoch_loss = 0.0
        epoch_align = 0.0
        epoch_creator_align = 0.0
        epoch_creator_align_n = 0
        epoch_unif = 0.0
        n_batches = 0

        enc.head.train()
        for s in range(0, n_total, batch_size):
            idx = order[s : s + batch_size]
            if len(idx) < 4:
                continue
            batch_feats = feats[idx]
            batch_cre = [creator_ids[i] for i in idx]
            batch_inst = [instance_ids[i] for i in idx]

            projected = enc.head(batch_feats)                            # (B, D)
            loss = _supcon_loss(projected, batch_cre, batch_inst, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                inst_arr = np.array(batch_inst)
                cre_arr = np.array(batch_cre)
                same_inst = (inst_arr[:, None] == inst_arr[None, :])
                same_creator = (cre_arr[:, None] == cre_arr[None, :])
                np.fill_diagonal(same_inst, False)
                np.fill_diagonal(same_creator, False)
                # Creator-only positives: same creator AND different instance
                # (this is what the SupCon loss is *actually* trying to drive
                # together — distinct from the SimCLR alignment that CLAP
                # handles for free)
                creator_only = same_creator & ~same_inst

                if same_inst.any():
                    pairs = np.argwhere(same_inst)
                    a_idx = torch.tensor(pairs[:, 0], device=device, dtype=torch.long)
                    b_idx = torch.tensor(pairs[:, 1], device=device, dtype=torch.long)
                    align = (projected[a_idx] - projected[b_idx]).pow(2).sum(dim=-1).mean()
                else:
                    align = torch.tensor(0.0, device=device)

                if creator_only.any():
                    cpairs = np.argwhere(creator_only)
                    ca = torch.tensor(cpairs[:, 0], device=device, dtype=torch.long)
                    cb = torch.tensor(cpairs[:, 1], device=device, dtype=torch.long)
                    creator_align = (
                        (projected[ca] - projected[cb]).pow(2).sum(dim=-1).mean()
                    )
                    epoch_creator_align += float(creator_align.item())
                    epoch_creator_align_n += 1

                if projected.size(0) > 1:
                    sq = torch.cdist(projected, projected).pow(2)
                    mask_e = ~torch.eye(projected.size(0), device=device, dtype=torch.bool)
                    unif = sq[mask_e].mul(-2.0).exp().mean().log()
                else:
                    unif = torch.tensor(0.0, device=device)

            epoch_loss += float(loss.item())
            epoch_align += float(align.item())
            epoch_unif += float(unif.item())
            n_batches += 1

        scheduler.step()
        nb = max(n_batches, 1)
        avg_loss = epoch_loss / nb
        avg_align = epoch_align / nb
        avg_creator_align = (
            epoch_creator_align / epoch_creator_align_n
            if epoch_creator_align_n > 0 else float("nan")
        )
        # Cosine-similarity proxy of same-instance alignment (legacy diagnostic)
        mix_cosine_proxy = max(0.0, 1.0 - 0.5 * avg_align)
        m = EpochMetrics(
            epoch=epoch,
            loss=avg_loss,
            lr=scheduler.get_last_lr()[0],
            elapsed_sec=time.time() - epoch_start,
            contrastive_loss=avg_loss,
            mix_loss=0.0,
            align=avg_align,
            uniformity=epoch_unif / nb,
            mix_cosine=mix_cosine_proxy,
        )
        # Stash creator_align in the metrics dict via a side attribute. We
        # extend EpochMetrics' serialization so it shows up in training.json.
        m.__dict__["creator_align"] = avg_creator_align
        metrics.append(m)

        event(_log, "train_proj.epoch",
              epoch=epoch + 1, total=epochs,
              loss=round(m.loss, 6),
              align=round(m.align, 4),
              creator_align=(
                  round(avg_creator_align, 4)
                  if not math.isnan(avg_creator_align) else None
              ),
              mix_cos=round(m.mix_cosine, 4),
              unif=round(m.uniformity, 4),
              lr=m.lr, elapsed_sec=round(m.elapsed_sec, 2))

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump([mm.__dict__ for mm in metrics], f, indent=2)

        # Loss-plateau early stop. We only consider stopping after
        # `min_epochs` to give the head room to actually learn creator-DNA.
        if avg_loss < best_loss * (1.0 - early_stop_loss_rel_improvement):
            best_loss = avg_loss
            epochs_since_improvement = 0
            best_head_state = {k: v.detach().cpu().clone()
                               for k, v in enc.head.state_dict().items()}
        else:
            epochs_since_improvement += 1

        if (epoch + 1) >= min_epochs and epochs_since_improvement >= early_stop_patience:
            event(_log, "train_proj.early_stop",
                  epoch=epoch + 1, best_loss=round(best_loss, 6),
                  epochs_since_improvement=epochs_since_improvement)
            break

    # Restore the best-loss head before checkpointing — guards against the
    # final epoch being a regression on the loss-plateau watch.
    if best_head_state is not None:
        enc.head.load_state_dict({
            k: v.to(device) for k, v in best_head_state.items()
        })
    enc.save_head(checkpoint_path)
    event(_log, "train_proj.done",
          checkpoint=str(checkpoint_path),
          epochs_run=len(metrics),
          final_loss=round(metrics[-1].loss, 6),
          best_loss=round(best_loss, 6))
    return enc, metrics
