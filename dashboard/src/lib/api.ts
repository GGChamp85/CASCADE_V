"use client";

import { useQuery } from "@tanstack/react-query";

// Always use relative URLs so requests go to the Next.js dev server, which
// proxies /api/* to FastAPI via next.config.js rewrites. This avoids CORS,
// avoids leaking the backend port into the browser bundle, and survives
// hard reloads after the backend port changes.
export const API_BASE = "";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return (await r.json()) as T;
}

export type HealthResponse = {
  status: string;
  device: string;
  catalog_size: number;
  n_creators: number;
  n_receipts: number;
};

export type CreatorRecord = {
  creator_id: string;
  name: string;
  base_freq_hz: number;
  detune_cents: number;
  decay_factor: number;
  harmonic_richness: number;
  noise_floor: number;
  rhythm_density: number;
};

export type SourceRecord = {
  source_id: string;
  creator_id: string;
  creator_name: string;
  category: string;
  bpm: number;
  key: string;
  file_path: string;
};

export type CatalogResponse = {
  creators: CreatorRecord[];
  sources: SourceRecord[];
};

export type GroundTruthRecord = {
  output_id: string;
  file_path: string;
  source_ids: string[];
  weights: number[];
  creator_ids: string[];
  creator_weights: Record<string, number>;
  sources_annotated?: Array<{
    source_id: string;
    creator_id: string;
    creator_name: string;
    weight: number;
  }>;
  creators_annotated?: Array<{
    creator_id: string;
    creator_name: string;
    weight: number;
  }>;
};

export type ReceiptSummary = {
  receipt_id: string;
  created_at_utc: string;
  overall_status: "PROVEN" | "VIOLATED";
  top_creator: string | null;
  top_creator_weight: number | null;
  n_clusters: number | null;
  candidates_from_triage: number | null;
  total_latency_ms: number | null;
};

export type AxiomCheck = {
  name: "efficiency" | "symmetry" | "dummy";
  status: "PROVEN" | "VIOLATED" | "NA";
  detail: string;
};

export type Receipt = {
  receipt_id: string;
  created_at_utc: string;
  total_payout_usd: number;
  method: string;
  version: string;
  per_creator: Array<{
    creator_id: string;
    creator_name: string;
    weight_point: number;
    weight_lower: number;
    weight_upper: number;
    payout_usd: number;
    n_sources: number;
  }>;
  per_source: Array<{
    source_id: string;
    creator_id: string;
    creator_name: string;
    weight_point: number;
    weight_lower: number;
    weight_upper: number;
    payout_usd: number;
    stage_path: string;
  }>;
  latency_ms: Record<string, number>;
  verification: {
    overall_status: "PROVEN" | "VIOLATED";
    axioms: AxiomCheck[];
    smt_lib_file: string;
    smt_lib_hash: string;
  };
  pipeline_metadata: {
    catalog_size: number;
    candidates_from_triage: number;
    n_clusters: number;
    validations_passed: boolean;
  };
};

export type ProofResponse = {
  receipt_id: string;
  smt_lib: string;
  annotated?: string;
  z3_verdict: string;
};

export type ResultRow = {
  output_id: string;
  method: string;
  instance_mae: number;
  creator_mae: number;
  top1_hit: boolean;
  coverage_at_k: number;
  axioms_proven: number;
  axioms_total: number;
  is_dna_case: boolean;
};

export type TrainingEpoch = {
  epoch: number;
  loss: number;
  lr: number;
  elapsed_sec: number;
  contrastive_loss?: number;
  mix_loss?: number;
  align?: number;
  uniformity?: number;
  mix_cosine?: number;
};

export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: () => getJSON<HealthResponse>("/api/health"),
    refetchInterval: 30_000,
  });

export const useCatalog = () =>
  useQuery({
    queryKey: ["catalog"],
    queryFn: () => getJSON<CatalogResponse>("/api/catalog"),
  });

export const useTestOutputs = () =>
  useQuery({
    queryKey: ["test-outputs"],
    queryFn: () => getJSON<GroundTruthRecord[]>("/api/test-outputs"),
  });

export const useGroundTruth = (output_id: string | null) =>
  useQuery({
    queryKey: ["ground-truth", output_id],
    queryFn: () => getJSON<GroundTruthRecord>(`/api/ground-truth/${output_id}`),
    enabled: !!output_id,
    retry: false,
  });

export const useReceipts = () =>
  useQuery({
    queryKey: ["receipts"],
    queryFn: () => getJSON<ReceiptSummary[]>("/api/receipts"),
  });

export const useReceipt = (id: string | null) =>
  useQuery({
    queryKey: ["receipt", id],
    queryFn: () => getJSON<Receipt>(`/api/receipts/${id}`),
    enabled: !!id,
  });

export const useProof = (id: string | null) =>
  useQuery({
    queryKey: ["proof", id],
    queryFn: () => getJSON<ProofResponse>(`/api/proofs/${id}`),
    enabled: !!id,
  });

export const useResults = () =>
  useQuery({
    queryKey: ["results"],
    queryFn: () => getJSON<ResultRow[]>("/api/results"),
  });

export const useTraining = () =>
  useQuery({
    queryKey: ["training"],
    queryFn: () => getJSON<TrainingEpoch[]>("/api/training"),
  });
