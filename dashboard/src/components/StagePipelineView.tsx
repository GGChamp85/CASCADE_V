"use client";

import { useState } from "react";

import StageInfoModal, { StageKey } from "./StageInfoModal";

type ReceiptWithStages = {
  receipt_id: string;
  pipeline_metadata?: any;
  verification?: any;
  latency_ms?: Record<string, number>;
  stages?: {
    stage1_triage?: {
      k: number;
      metadata: any;
      candidates: Array<{
        source_id: string;
        creator_id: string;
        creator_name: string;
        score: number;
        rank: number;
      }>;
    };
    stage2_grouping?: {
      n_clusters: number;
      metadata: any;
      clusters: Array<{
        cluster_id: number;
        weight: number;
        raw_score: number;
        member_source_ids: string[];
      }>;
    };
    stage3_shapley?: {
      per_cluster: Array<{
        cluster_id: number;
        method: string;
        n: number;
        metadata: any;
        members: Array<{
          source_id: string;
          shapley_value: number;
          weight_in_cluster: number;
          interval_lower: number;
          interval_upper: number;
        }>;
      }>;
    };
  };
};

const CLUSTER_COLORS = [
  "#0d9488",
  "#3b82f6",
  "#a855f7",
  "#d97706",
  "#ef4444",
  "#10b981",
  "#0ea5e9",
  "#f59e0b",
  "#ec4899",
  "#8b5cf6",
  "#14b8a6",
  "#f97316",
];

export default function StagePipelineView({
  receipt,
}: {
  receipt: ReceiptWithStages;
}) {
  const stages = receipt.stages;
  const [info, setInfo] = useState<StageKey | null>(null);
  const lat = receipt.latency_ms || {};

  if (!stages) {
    return (
      <div className="card p-5 text-sm text-subt">
        Step-by-step view requires a receipt produced by CASCADE-V v0.2.0+.
        Re-run <code className="kbd">cascade-evaluate --force</code> to upgrade
        older receipts.
      </div>
    );
  }

  const tri = stages.stage1_triage;
  const grp = stages.stage2_grouping;
  const shp = stages.stage3_shapley;

  // map source_id → cluster_id (from grouping)
  const sourceToCluster: Record<string, number> = {};
  grp?.clusters.forEach((c) =>
    c.member_source_ids.forEach((sid) => {
      sourceToCluster[sid] = c.cluster_id;
    }),
  );
  const colorOf = (cid?: number) =>
    cid != null ? CLUSTER_COLORS[(cid - 1) % CLUSTER_COLORS.length] : "#94a3b8";

  return (
    <div className="card p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <div className="text-base font-semibold">Step-by-step output</div>
          <div className="text-xs text-subt">
            Click any step's ⓘ for the algorithm details. Click a step header to
            expand its data for this receipt.
          </div>
        </div>
        <span className="pill pill-brand font-mono">
          {(lat.total ?? 0).toFixed(1)} ms total
        </span>
      </div>

      <div className="flex flex-col gap-3">
        {/* STEP 1 */}
        <Step
          number="STEP 1"
          title="Triage"
          subtitle={`top-${tri?.k ?? "?"} from catalog · NNLS-first rank + cosine tie-break · BPM/key gate optional`}
          color="from-rose-50 to-orange-50"
          ring="ring-rose-200"
          icon="◎"
          iconBg="bg-gradient-to-br from-rose-400 to-orange-500"
          latency={lat.stage1_triage}
          onInfo={() => setInfo("stage1")}
        >
          {tri ? (
            <div className="space-y-2">
              <div className="text-xs text-subt">
                Catalog: {tri.metadata?.catalog_size ?? "?"} stems · Score
                range: max{" "}
                <span className="font-mono">
                  {tri.metadata?.raw_max_similarity?.toFixed(3)}
                </span>{" "}
                · mean{" "}
                <span className="font-mono">
                  {tri.metadata?.raw_mean_similarity?.toFixed(3)}
                </span>
              </div>
              <div className="overflow-hidden rounded-lg border border-border">
                <table className="w-full text-xs">
                  <thead className="bg-slate-50 text-subt">
                    <tr>
                      <th className="px-2 py-1.5 text-left font-medium">#</th>
                      <th className="px-2 py-1.5 text-left font-medium">
                        Source
                      </th>
                      <th className="px-2 py-1.5 text-left font-medium">
                        Creator
                      </th>
                      <th className="px-2 py-1.5 text-right font-medium">
                        Score
                      </th>
                      <th className="px-2 py-1.5 text-left font-medium">
                        → cluster
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tri.candidates.map((c) => {
                      const cid = sourceToCluster[c.source_id];
                      return (
                        <tr
                          key={c.source_id}
                          className="border-t border-border/60"
                        >
                          <td className="px-2 py-1 font-mono text-subt">
                            {c.rank}
                          </td>
                          <td className="px-2 py-1 font-mono">{c.source_id}</td>
                          <td className="px-2 py-1">{c.creator_name}</td>
                          <td className="px-2 py-1 text-right font-mono">
                            {c.score.toFixed(4)}
                          </td>
                          <td className="px-2 py-1">
                            <span
                              className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold text-white"
                              style={{ background: colorOf(cid) }}
                            >
                              C{cid ?? "?"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-xs text-subt">no triage data</div>
          )}
        </Step>

        {/* STEP 2 */}
        <Step
          number="STEP 2"
          title="Group"
          subtitle={`${grp?.metadata?.clustering_method === "hdbscan" ? "HDBSCAN density-based" : "Ward linkage"} · ${tri?.k ?? "?"} candidates → ${grp?.n_clusters ?? "?"} clusters · sparsified at MIN_CLUSTER_WEIGHT=0.07`}
          color="from-violet-50 to-indigo-50"
          ring="ring-violet-200"
          icon="◫"
          iconBg="bg-gradient-to-br from-violet-500 to-indigo-600"
          latency={lat.stage2_grouping}
          onInfo={() => setInfo("stage2")}
        >
          {grp ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {grp.clusters
                .slice()
                .sort((a, b) => b.weight - a.weight)
                .map((c) => {
                  const color = colorOf(c.cluster_id);
                  return (
                    <div
                      key={c.cluster_id}
                      className="rounded-lg border border-border bg-white p-3"
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className="rounded-full px-2 py-0.5 text-[10px] font-semibold text-white"
                          style={{ background: color }}
                        >
                          Cluster C{c.cluster_id}
                        </span>
                        <span
                          className="font-mono text-sm font-bold"
                          style={{ color }}
                        >
                          {(c.weight * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(c.weight * 100, 2)}%`,
                            background: color,
                          }}
                        />
                      </div>
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {c.member_source_ids.map((sid) => (
                          <span
                            key={sid}
                            className="rounded border border-border bg-slate-50 px-1.5 py-0.5 font-mono text-[10px]"
                          >
                            {sid}
                          </span>
                        ))}
                      </div>
                      <div className="mt-1 text-[10px] text-subt">
                        counterfactual influence:{" "}
                        <span className="font-mono">
                          {c.raw_score.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          ) : (
            <div className="text-xs text-subt">no grouping data</div>
          )}
        </Step>

        {/* STEP 3 */}
        <Step
          number="STEP 3"
          title="Shapley"
          subtitle="Within each cluster: exact 2ⁿ enum or MC + Hoeffding"
          color="from-emerald-50 to-teal-50"
          ring="ring-emerald-200"
          icon="Σ"
          iconBg="bg-gradient-to-br from-emerald-400 to-teal-500"
          latency={lat.stage3_shapley_total}
          onInfo={() => setInfo("stage3")}
        >
          {shp ? (
            <div className="space-y-2">
              {shp.per_cluster.map((c) => {
                const color = colorOf(c.cluster_id);
                return (
                  <div
                    key={c.cluster_id}
                    className="rounded-lg border border-border bg-white p-3"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className="rounded-full px-2 py-0.5 text-[10px] font-semibold text-white"
                        style={{ background: color }}
                      >
                        Cluster C{c.cluster_id}
                      </span>
                      <span className="pill pill-mute font-mono">
                        {c.method.replace("shapley_", "")}
                      </span>
                      <span className="text-[11px] text-subt">
                        n = {c.n} ·{" "}
                        {c.metadata?.n_evaluations
                          ? `${c.metadata.n_evaluations} evaluations`
                          : ""}
                        {c.metadata?.hoeffding_epsilon
                          ? ` · ε = ${c.metadata.hoeffding_epsilon.toFixed(3)}`
                          : ""}
                      </span>
                    </div>
                    <table className="mt-2 w-full text-xs">
                      <thead className="text-subt">
                        <tr>
                          <th className="text-left font-medium">Source</th>
                          <th className="text-right font-medium">Shapley φ</th>
                          <th className="text-right font-medium">
                            Weight (in cluster)
                          </th>
                          <th className="text-right font-medium">Interval</th>
                        </tr>
                      </thead>
                      <tbody>
                        {c.members.map((m) => (
                          <tr key={m.source_id}>
                            <td className="font-mono">{m.source_id}</td>
                            <td className="text-right font-mono">
                              {m.shapley_value.toFixed(4)}
                            </td>
                            <td className="text-right font-mono font-medium">
                              {(m.weight_in_cluster * 100).toFixed(1)}%
                            </td>
                            <td className="text-right font-mono text-subt">
                              [{(m.interval_lower * 100).toFixed(1)},{" "}
                              {(m.interval_upper * 100).toFixed(1)}]%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-xs text-subt">no Shapley data</div>
          )}
        </Step>

        {/* STEP 4 */}
        <Step
          number="STEP 4"
          title="Compose"
          subtitle="cluster_weight × within_cluster_shapley → final per-source weight, with intervals"
          color="from-sky-50 to-blue-50"
          ring="ring-sky-200"
          icon="⏚"
          iconBg="bg-gradient-to-br from-sky-400 to-blue-600"
          latency={lat.compose}
          onInfo={() => setInfo("compose")}
        >
          <div className="rounded-lg border border-border bg-white p-3">
            <div className="text-xs text-subt">
              For every source <code className="kbd">i</code> in cluster{" "}
              <code className="kbd">C</code>:
            </div>
            <div className="mt-1.5 rounded-md bg-slate-50 p-2 font-mono text-xs leading-relaxed">
              <span className="text-violet-700">final[i]</span>{" "}
              <span className="text-subt">←</span>{" "}
              <span className="text-rose-700">cluster_weight[C]</span>{" "}
              <span className="text-subt">×</span>{" "}
              <span className="text-emerald-700">shapley_in_cluster[i]</span>
              <br />
              <span className="text-subt">// then renormalize so Σ final[i] = 1.0</span>
            </div>
            <div className="mt-2 text-xs text-subt">
              Intervals propagate via mpmath rounding-aware multiplication at 80-bit precision.
            </div>
          </div>
        </Step>

        {/* STEP 5 */}
        <Step
          number="STEP 5"
          title="Verify"
          subtitle="Z3 SMT proof: efficiency · symmetry · null-player axioms"
          color="from-amber-50 to-orange-50"
          ring="ring-amber-200"
          icon="✓"
          iconBg="bg-gradient-to-br from-amber-400 to-orange-500"
          latency={lat.stageV_proof}
          onInfo={() => setInfo("stageV")}
        >
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {receipt.verification?.axioms?.map((ax: any) => {
              const dot =
                ax.status === "PROVEN"
                  ? "bg-emerald-500"
                  : ax.status === "VIOLATED"
                    ? "bg-red-500"
                    : "bg-slate-400";
              const tone =
                ax.status === "PROVEN"
                  ? "border-emerald-200 bg-emerald-50/40"
                  : ax.status === "VIOLATED"
                    ? "border-red-200 bg-red-50/40"
                    : "border-border bg-slate-50/40";
              const labels: Record<string, string> = {
                efficiency: "Efficiency",
                symmetry: "Symmetry",
                dummy: "Null player",
              };
              return (
                <div
                  key={ax.name}
                  className={`flex items-start gap-2 rounded-lg border p-2.5 ${tone}`}
                >
                  <div
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] text-white ${dot}`}
                  >
                    {ax.status === "PROVEN"
                      ? "✓"
                      : ax.status === "VIOLATED"
                        ? "✗"
                        : "—"}
                  </div>
                  <div>
                    <div className="text-xs font-semibold">
                      {labels[ax.name] || ax.name}
                    </div>
                    <div className="text-[10px] leading-snug text-subt">
                      {ax.detail}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Step>
      </div>

      {info && <StageInfoModal stage={info} onClose={() => setInfo(null)} />}
    </div>
  );
}

function Step({
  number,
  title,
  subtitle,
  color,
  ring,
  icon,
  iconBg,
  latency,
  onInfo,
  children,
}: {
  number: string;
  title: string;
  subtitle: string;
  color: string;
  ring: string;
  icon: string;
  iconBg: string;
  latency?: number;
  onInfo: () => void;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`rounded-2xl bg-gradient-to-br ${color} p-3 ring-1 ${ring}`}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 text-left"
        aria-expanded={open}
      >
        <div className={`step-icon !h-10 !w-10 !text-base ${iconBg}`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-subt">
            {number}
            {latency !== undefined && (
              <span className="font-mono text-ink">{latency.toFixed(1)} ms</span>
            )}
          </div>
          <div className="text-sm font-semibold leading-tight">{title}</div>
          <div className="text-[11px] leading-snug text-subt">{subtitle}</div>
        </div>
        <span
          onClick={(e) => {
            e.stopPropagation();
            onInfo();
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.stopPropagation();
              onInfo();
            }
          }}
          className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full bg-white text-xs font-semibold text-subt shadow-soft hover:text-violet-700"
          title="Algorithm details"
        >
          ⓘ
        </span>
        <span
          className="flex h-7 w-7 items-center justify-center rounded-full bg-white text-xs font-bold text-subt shadow-soft"
          aria-label="Toggle"
        >
          {open ? "−" : "+"}
        </span>
      </button>
      {open && <div className="mt-3">{children}</div>}
    </div>
  );
}
