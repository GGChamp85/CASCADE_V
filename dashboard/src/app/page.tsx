"use client";

import Link from "next/link";
import { useState } from "react";

import KpiTile from "@/components/KpiTile";
import PipelineStrip from "@/components/PipelineStrip";
import StatusModal from "@/components/StatusModal";
import { useHealth, useReceipts, useResults, useTraining } from "@/lib/api";

export default function DashboardHome() {
  const { data: health } = useHealth();
  const { data: receipts } = useReceipts();
  const { data: results } = useResults();
  const { data: training } = useTraining();

  const cv = (results || []).filter((r) => r.method === "cascade_v");
  const axiomPass = cv.length
    ? cv.reduce((a, r) => a + r.axioms_proven, 0) /
      Math.max(1, cv.reduce((a, r) => a + r.axioms_total, 0))
    : NaN;
  const avgInstanceMae = cv.length
    ? cv.reduce((a, r) => a + r.instance_mae, 0) / cv.length
    : NaN;
  const avgLatency = receipts?.length
    ? receipts.reduce((a, r) => a + (r.total_latency_ms || 0), 0) /
      receipts.length
    : NaN;
  const finalLoss = training?.length ? training[training.length - 1].loss : NaN;
  const provenCount = receipts?.filter((r) => r.overall_status === "PROVEN").length ?? 0;
  const [modalId, setModalId] = useState<string | null>(null);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-3xl font-bold text-ink">CASCADE-V Dashboard</h1>
        <p className="mt-1 text-sm text-subt">
          <span className="font-semibold text-violet-700">C</span>oalition-aware{" "}
          <span className="font-semibold text-rose-600">S</span>ource{" "}
          <span className="font-semibold text-emerald-600">C</span>rediting{" "}
          And{" "}
          <span className="font-semibold text-sky-600">D</span>ecomposed{" "}
          <span className="font-semibold text-amber-600">E</span>ngine,{" "}
          Verified — multi-source attribution with Z3-verified Shapley fairness.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          icon="≡"
          label="Catalog"
          value={health ? health.catalog_size : "—"}
          sub={health ? `${health.n_creators} creators · 30 test outputs` : null}
          tone="white"
        />
        <KpiTile
          icon="✓"
          label="Receipts proven"
          value={
            receipts?.length
              ? `${((provenCount / receipts.length) * 100).toFixed(1)}%`
              : "—"
          }
          sub={
            receipts?.length
              ? `${provenCount} of ${receipts.length} Z3-verified`
              : null
          }
          tone="mint"
        />
        <KpiTile
          icon="⚖"
          label="Axiom pass-rate"
          value={isNaN(axiomPass) ? "—" : `${(axiomPass * 100).toFixed(1)}%`}
          sub={
            isNaN(axiomPass)
              ? null
              : `efficiency · symmetry · dummy across ${cv.length} receipts`
          }
          tone="coral"
        />
        <KpiTile
          icon="⏱"
          label="Avg latency"
          value={isNaN(avgLatency) ? "—" : `${avgLatency.toFixed(0)}ms`}
          sub="end-to-end attribution"
          tone="violet"
        />
      </div>

      <PipelineStrip
        statuses={{
          stage1: "done",
          stage2: "done",
          stage3: "done",
          compose: "done",
          stageV: "done",
        }}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-base font-semibold">Recent receipts</div>
              <div className="text-xs text-subt">
                Sorted by attribution time, click to view detail
              </div>
            </div>
            <Link
              href="/receipts"
              className="rounded-lg border border-border bg-white px-3 py-1 text-xs font-medium text-ink hover:bg-slate-50"
            >
              See all →
            </Link>
          </div>
          <table className="w-full text-sm">
            <thead className="border-b border-border text-subt">
              <tr>
                <th className="py-2 text-left font-medium">Receipt</th>
                <th className="py-2 text-left font-medium">Top creator</th>
                <th className="py-2 text-right font-medium">Weight</th>
                <th className="py-2 text-right font-medium">Latency</th>
                <th className="py-2 text-left font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {(receipts || []).slice(0, 8).map((r) => (
                <tr
                  key={r.receipt_id}
                  className="border-b border-border/60 last:border-0 hover:bg-slate-50"
                >
                  <td className="py-2 font-mono text-xs">
                    <Link
                      href={`/receipts/${r.receipt_id}`}
                      className="text-violet-700 hover:underline"
                    >
                      {r.receipt_id}
                    </Link>
                  </td>
                  <td className="py-2">{r.top_creator || "—"}</td>
                  <td className="py-2 text-right font-mono">
                    {r.top_creator_weight != null
                      ? `${(r.top_creator_weight * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td className="py-2 text-right font-mono">
                    {r.total_latency_ms != null
                      ? `${r.total_latency_ms.toFixed(1)}ms`
                      : "—"}
                  </td>
                  <td className="py-2">
                    <button
                      onClick={() => setModalId(r.receipt_id)}
                      className="cursor-pointer transition-transform hover:scale-105"
                      title="Click to see why this verdict was reached"
                    >
                      {r.overall_status === "PROVEN" && (
                        <span className="pill pill-ok">PROVEN ⓘ</span>
                      )}
                      {r.overall_status === "VIOLATED" && (
                        <span className="pill pill-bad">VIOLATED ⓘ</span>
                      )}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex flex-col gap-3">
          <div className="card flex flex-col gap-3 p-5">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-blue-400 to-sky-600 text-sm text-white">
                △
              </div>
              <div className="text-base font-semibold">Training</div>
            </div>
            <div className="space-y-1 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-subt">Epochs</span>
                <span className="font-mono">{training?.length ?? "—"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-subt">Final loss</span>
                <span className="font-mono">
                  {isNaN(finalLoss) ? "—" : finalLoss.toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-subt">Instance MAE</span>
                <span className="font-mono">
                  {isNaN(avgInstanceMae) ? "—" : avgInstanceMae.toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-subt">Device</span>
                <span className="font-mono">{health?.device ?? "—"}</span>
              </div>
            </div>
          </div>

          <Link
            href="/attribute"
            className="card flex items-center gap-3 p-5 transition-colors hover:bg-violet-50"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 text-lg text-white shadow-soft">
              ⏵
            </div>
            <div className="flex-1">
              <div className="text-sm font-semibold">Run live attribution</div>
              <div className="text-xs text-subt">
                Pick a test output, watch stages stream live
              </div>
            </div>
            <span className="text-violet-500">→</span>
          </Link>

          <Link
            href="/eval"
            className="card flex items-center gap-3 p-5 transition-colors hover:bg-amber-50"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 text-lg text-white shadow-soft">
              ▮
            </div>
            <div className="flex-1">
              <div className="text-sm font-semibold">Method comparison</div>
              <div className="text-xs text-subt">
                CASCADE-V vs Shapley / LOO / TRAK
              </div>
            </div>
            <span className="text-amber-500">→</span>
          </Link>
        </div>
      </div>
      {modalId && (
        <StatusModal receiptId={modalId} onClose={() => setModalId(null)} />
      )}
    </div>
  );
}
