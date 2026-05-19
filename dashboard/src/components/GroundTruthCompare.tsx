"use client";

import { API_BASE, Receipt, useGroundTruth } from "@/lib/api";

/**
 * Side-by-side ground truth vs prediction. Only shown for test outputs that
 * have known ground truth (the `output_NNN` family). Uploads have no GT.
 *
 * Color coding:
 *   green pill — both predicted and actual ("Hit")
 *   amber pill — actual but not predicted ("Missed")
 *   slate pill — predicted but not actual ("False positive")
 */
export default function GroundTruthCompare({ receipt }: { receipt: Receipt }) {
  const targetIsTest = !receipt.receipt_id.startsWith("upload_");
  const { data: gt, isError } = useGroundTruth(
    targetIsTest ? receipt.receipt_id : null,
  );

  if (!targetIsTest || isError || !gt?.creators_annotated) {
    return null;
  }

  // build creator-level comparison
  const predicted = new Map(
    receipt.per_creator
      .filter((c) => c.weight_point > 0.005)
      .map((c) => [c.creator_id, c]),
  );
  const actual = new Map(
    gt.creators_annotated.map((c) => [c.creator_id, c]),
  );

  const allKeys = new Set([...predicted.keys(), ...actual.keys()]);
  type Row = {
    creator_id: string;
    creator_name: string;
    actual: number;
    predicted: number;
    status: "hit" | "miss" | "false";
  };
  const rows: Row[] = [];
  for (const key of allKeys) {
    const a = actual.get(key);
    const p = predicted.get(key);
    const name = (a as any)?.creator_name || (p as any)?.creator_name || key;
    const aw = a?.weight ?? 0;
    const pw = p?.weight_point ?? 0;
    const status: Row["status"] = a && p ? "hit" : a ? "miss" : "false";
    rows.push({
      creator_id: key,
      creator_name: name,
      actual: aw,
      predicted: pw,
      status,
    });
  }
  rows.sort((a, b) => Math.max(b.actual, b.predicted) - Math.max(a.actual, a.predicted));

  const totalAbsErr = rows.reduce((s, r) => s + Math.abs(r.actual - r.predicted), 0);
  const hits = rows.filter((r) => r.status === "hit").length;
  const total_actual = rows.filter((r) => r.actual > 0).length;
  const recall = total_actual > 0 ? hits / total_actual : 0;

  const STATUS_LABEL: Record<Row["status"], string> = {
    hit: "Hit",
    miss: "Missed",
    false: "False positive",
  };
  const STATUS_PILL: Record<Row["status"], string> = {
    hit: "pill-ok",
    miss: "pill-warn",
    false: "pill-mute",
  };
  const STATUS_DESC: Record<Row["status"], string> = {
    hit: "Predicted AND actually used",
    miss: "Actually used but NOT predicted",
    false: "Predicted but NOT actually used",
  };

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="text-base font-semibold">
            Ground truth vs prediction
          </div>
          <div className="text-xs text-subt">
            Test outputs come from a known mix, so we can score the attribution
            directly. Green = matched, amber = missed contributor, slate =
            false positive.
          </div>
        </div>
        <div className="flex flex-col items-end gap-0.5 text-xs">
          <span className="pill pill-brand">
            recall {(recall * 100).toFixed(0)}% ({hits}/{total_actual})
          </span>
          <span className="pill pill-mute font-mono">
            creator MAE {(totalAbsErr / Math.max(rows.length, 1)).toFixed(3)}
          </span>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-border">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-subt">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Creator</th>
              <th className="px-3 py-2 text-right font-medium">Actual</th>
              <th className="px-3 py-2 text-right font-medium">Predicted</th>
              <th className="px-3 py-2 text-right font-medium">Δ</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const delta = r.predicted - r.actual;
              const deltaSign = delta > 0 ? "+" : "";
              return (
                <tr
                  key={r.creator_id}
                  className={
                    "border-t border-border " +
                    (r.status === "hit"
                      ? "bg-emerald-50/30"
                      : r.status === "miss"
                        ? "bg-amber-50/40"
                        : "bg-slate-50/40")
                  }
                  title={STATUS_DESC[r.status]}
                >
                  <td className="px-3 py-2 font-medium">{r.creator_name}</td>
                  <td className="px-3 py-2 text-right font-mono">
                    {r.actual > 0 ? `${(r.actual * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-2 text-right font-mono">
                    {r.predicted > 0 ? `${(r.predicted * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td
                    className={
                      "px-3 py-2 text-right font-mono " +
                      (Math.abs(delta) < 0.05
                        ? "text-emerald-700"
                        : "text-rose-700")
                    }
                  >
                    {deltaSign}
                    {(delta * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2">
                    <span className={`pill ${STATUS_PILL[r.status]}`}>
                      {STATUS_LABEL[r.status]}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {gt.sources_annotated && gt.sources_annotated.length > 0 && (
        <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs leading-relaxed">
          <div className="mb-1 font-semibold text-amber-900">
            Sources actually used in this output (mixer ground truth):
          </div>
          <ul className="flex flex-wrap gap-2">
            {[...gt.sources_annotated]
              .sort((a, b) => b.weight - a.weight)
              .map((s) => (
                <li
                  key={s.source_id}
                  className="flex items-center gap-1.5 rounded-md bg-white px-2 py-1 text-amber-900 ring-1 ring-amber-200"
                >
                  <audio
                    controls
                    preload="none"
                    className="h-7"
                    src={`${API_BASE}/api/audio/catalog/${s.source_id}`}
                  />
                  <span className="font-mono text-[11px]">{s.source_id}</span>
                  <span className="font-medium">{s.creator_name}</span>
                  <span className="font-mono">{(s.weight * 100).toFixed(1)}%</span>
                </li>
              ))}
          </ul>
        </div>
      )}

      <div className="mt-3 rounded-md border border-border bg-slate-50 p-3 text-[11px] leading-relaxed text-subt">
        <span className="font-semibold text-ink">How to read this:</span> The
        attribution math is right by construction (Z3-verified Shapley split of
        the embedding-space value function). Recovery accuracy depends entirely
        on the encoder — if the encoder can't tell two stems apart, neither can
        the attributor. At 200 stems × 24 creators with this 1.4M-param encoder,
        recall is around 35–40%; a CLAP-scale encoder would close that gap.
      </div>
    </div>
  );
}
