"use client";

import { useState } from "react";

type DignityProof = Record<string, "PROVEN" | "VIOLATED" | "NA">;

type CurrenciesBlock = {
  receipt_id?: string;
  dials: { alpha: number; beta: number; gamma: number };
  qws: {
    quantile_active: number;
    output_quality_score: number;
    weights_final: number[];
    weights_by_quantile?: Record<string, number[]>;
  };
  per_contributor: Array<{
    contributor_id: string;
    monetary: {
      weight_qws: number;
      weight_with_floor: number;
      payout_usd: number;
      interval_usd: [number, number];
    };
    reputational: {
      roles: string[];
      streak: string;
      is_featured: boolean;
      feature_probability: number;
    };
    opportunity: {
      priority_adjustment: number;
      diversity_amplified: boolean;
      under_represented_protected: boolean;
    };
  }>;
  dignity_proof: DignityProof;
};

type ReceiptWithCurrencies = {
  receipt_id: string;
  currencies?: CurrenciesBlock;
};

export default function CurrencyPanel({ receipt }: { receipt: ReceiptWithCurrencies }) {
  const [expanded, setExpanded] = useState(false);
  const c = receipt.currencies;

  if (!c) {
    return (
      <div className="card p-4 text-sm text-subt">
        No currencies block on this receipt — re-run with{" "}
        <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs">
          ENABLE_CURRENCIES=True
        </code>{" "}
        and{" "}
        <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs">
          cascade-evaluate --force
        </code>
        .
      </div>
    );
  }

  const dialPct = (v: number) => `${Math.round(v * 100)}%`;
  const dignityAllPass =
    c.dignity_proof.every_creator_received_all_three === "PROVEN";

  // Active = post-sparsification contributors only (already filtered by the
  // currencies layer with active_only=True).
  const sortedContributors = [...c.per_contributor].sort(
    (a, b) => b.monetary.weight_with_floor - a.monetary.weight_with_floor,
  );

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center gap-3 border-b border-border bg-gradient-to-br from-pink-50 to-rose-50 px-5 py-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-pink-400 to-rose-500 text-white">
          ★
        </div>
        <div className="flex-1">
          <div className="text-base font-semibold">
            Currencies — three-currency receipt
          </div>
          <div className="text-xs text-subt">
            QWS quantile {c.qws.quantile_active.toFixed(2)} (q ={" "}
            {c.qws.output_quality_score.toFixed(3)}) ·{" "}
            {c.per_contributor.length} active contributor
            {c.per_contributor.length === 1 ? "" : "s"}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <Pill label="α" value={dialPct(c.dials.alpha)} hint="fairness" />
          <Pill label="β" value={dialPct(c.dials.beta)} hint="recognition" />
          <Pill label="γ" value={dialPct(c.dials.gamma)} hint="opportunity" />
        </div>
        <span
          className={
            "ml-2 rounded-full px-2 py-0.5 text-[10px] font-semibold " +
            (dignityAllPass
              ? "bg-emerald-100 text-emerald-700"
              : "bg-rose-100 text-rose-700")
          }
        >
          {dignityAllPass ? "DIGNITY PROVEN" : "DIGNITY VIOLATED"}
        </span>
      </div>

      {/* Per-contributor compact table */}
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-xs">
          <thead className="border-b border-border bg-slate-50 text-subt">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Contributor</th>
              <th className="px-3 py-2 text-right font-medium">Money</th>
              <th className="px-3 py-2 text-right font-medium">Floor lift</th>
              <th className="px-3 py-2 text-left font-medium">Roles</th>
              <th className="px-3 py-2 text-right font-medium">Feature P</th>
              <th className="px-3 py-2 text-right font-medium">Opp. boost</th>
            </tr>
          </thead>
          <tbody>
            {sortedContributors.map((p) => {
              const lifted =
                p.monetary.weight_with_floor - p.monetary.weight_qws;
              return (
                <tr
                  key={p.contributor_id}
                  className="border-b border-border/60 last:border-0"
                >
                  <td className="px-3 py-1.5 font-mono">
                    <span className="mr-2">{p.contributor_id}</span>
                    {p.reputational.is_featured && (
                      <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-[9px] font-bold text-amber-700">
                        ★ FEATURED
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono">
                    ${p.monetary.payout_usd.toFixed(4)}
                    <div className="text-[9px] text-subt">
                      ({(p.monetary.weight_with_floor * 100).toFixed(1)}%)
                    </div>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono">
                    {lifted > 0.0001 ? (
                      <span className="text-violet-700">
                        +{(lifted * 100).toFixed(2)}pp
                      </span>
                    ) : lifted < -0.0001 ? (
                      <span className="text-amber-700">
                        {(lifted * 100).toFixed(2)}pp
                      </span>
                    ) : (
                      <span className="text-subt">—</span>
                    )}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex flex-wrap gap-1">
                      {p.reputational.roles.map((r) => (
                        <span
                          key={r}
                          className={
                            "rounded px-1.5 py-0.5 text-[9px] font-medium " +
                            (r === "ESSENTIAL CONTRIBUTOR"
                              ? "bg-emerald-100 text-emerald-700"
                              : r === "CONTRIBUTING VOICE"
                                ? "bg-slate-100 text-slate-600"
                                : "bg-violet-100 text-violet-700")
                          }
                        >
                          {r}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono">
                    {(p.reputational.feature_probability * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono">
                    {p.opportunity.priority_adjustment > 1e-6
                      ? `+${p.opportunity.priority_adjustment.toFixed(4)}`
                      : "0"}
                    {(p.opportunity.diversity_amplified ||
                      p.opportunity.under_represented_protected) && (
                      <div className="flex justify-end gap-1 text-[9px]">
                        {p.opportunity.diversity_amplified && (
                          <span className="text-blue-700">div</span>
                        )}
                        {p.opportunity.under_represented_protected && (
                          <span className="text-emerald-700">under-rep</span>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Dignity proof rows */}
      <div className="border-t border-border bg-slate-50 px-5 py-3">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wider text-subt">
            Dignity meta-proof
          </div>
          <button
            onClick={() => setExpanded((s) => !s)}
            className="text-xs text-violet-700 hover:underline"
          >
            {expanded ? "Hide details" : "Show all 6 checks"}
          </button>
        </div>
        {expanded && (
          <div className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
            {Object.entries(c.dignity_proof).map(([k, v]) => (
              <div
                key={k}
                className="flex items-center justify-between border-b border-border/40 py-1"
              >
                <span className="font-mono text-[10px] text-subt">{k}</span>
                <span
                  className={
                    "rounded px-1.5 py-0.5 text-[9px] font-bold " +
                    (v === "PROVEN"
                      ? "bg-emerald-100 text-emerald-700"
                      : v === "VIOLATED"
                        ? "bg-rose-100 text-rose-700"
                        : "bg-slate-200 text-slate-600")
                  }
                >
                  {v}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Pill({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div
      className="flex items-baseline gap-1 rounded-lg border border-border bg-white px-2 py-1 font-mono text-[11px]"
      title={hint}
    >
      <span className="text-rose-600">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
