"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Receipt } from "@/lib/api";

const STACK_COLORS = [
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

export default function PerSourceBars({ receipt }: { receipt: Receipt }) {
  const ranked = [...receipt.per_source]
    .sort((a, b) => b.weight_point - a.weight_point)
    .filter((s) => s.weight_point > 1e-4);
  const visible = ranked.slice(0, 12);

  const data = visible.map((s) => ({
    source: s.source_id,
    creator: s.creator_name,
    weight: s.weight_point * 100,
    err: [
      Math.max(0, (s.weight_point - s.weight_lower) * 100),
      Math.max(0, (s.weight_upper - s.weight_point) * 100),
    ] as [number, number],
    isExact:
      Math.abs(s.weight_upper - s.weight_lower) < 1e-6,
  }));

  const totalShown = visible.reduce((a, s) => a + s.weight_point, 0);
  const tail = ranked.length - visible.length;
  const tailWeight = ranked
    .slice(12)
    .reduce((a, s) => a + s.weight_point, 0);
  const stackData = [
    ...visible.map((s, i) => ({
      source_id: s.source_id,
      creator: s.creator_name,
      pct: s.weight_point * 100,
      color: STACK_COLORS[i % STACK_COLORS.length],
    })),
  ];
  if (tailWeight > 0) {
    stackData.push({
      source_id: `+ ${tail} smaller`,
      creator: "tail",
      pct: tailWeight * 100,
      color: "#cbd5e1",
    });
  }
  const allExact = visible.every((s) => Math.abs(s.weight_upper - s.weight_lower) < 1e-6);

  // Round-trip the visible weights so the stacked bar is a clean 100%
  const stackTotal = stackData.reduce((a, s) => a + s.pct, 0);

  return (
    <div className="card p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="text-sm font-semibold">Per-source attribution</div>
          <div className="text-xs text-subt">
            Each source's individual share of the 100% payout. Hover for confidence intervals.
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="pill pill-ok font-mono">
            Σ visible {totalShown.toFixed(3) === "1.000" ? "100.0" : (totalShown * 100).toFixed(1)}%
          </span>
          {allExact && (
            <span
              className="pill pill-mute"
              title="Intervals collapsed to a point because all clusters used exact 2^n Shapley (n ≤ 10). Monte Carlo + Hoeffding intervals widen above n = 10."
            >
              exact ε = 0
            </span>
          )}
        </div>
      </div>

      {/* 100%-stacked single bar — makes the 'they sum to 100%' fact visible */}
      <div className="mb-3">
        <div className="mb-1 flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-subt">
          <span>100% payout split</span>
          <span className="font-mono text-ink">
            Σ {stackTotal.toFixed(1)}%
          </span>
        </div>
        <div className="flex h-7 w-full overflow-hidden rounded-md border border-border bg-slate-50">
          {stackData.map((s, i) => (
            <div
              key={s.source_id + i}
              title={`${s.creator} · ${s.source_id} · ${s.pct.toFixed(2)}%`}
              className="flex h-full items-center justify-end overflow-hidden text-[10px] font-medium text-white"
              style={{ width: `${s.pct}%`, background: s.color }}
            >
              {s.pct > 5 && (
                <span className="px-1.5 truncate">
                  {s.pct.toFixed(0)}%
                </span>
              )}
            </div>
          ))}
        </div>
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-subt">
          {stackData.map((s, i) => (
            <span key={i} className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: s.color }}
              />
              {s.creator !== "tail" && (
                <>
                  <span className="font-mono">{s.source_id}</span>
                  <span className="text-ink">{s.creator}</span>
                </>
              )}
              {s.creator === "tail" && (
                <span className="italic">{s.source_id}</span>
              )}
              <span className="font-mono">{s.pct.toFixed(1)}%</span>
            </span>
          ))}
        </div>
      </div>

      {/* Detail per-source bar chart */}
      <div className="h-64">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="source"
              tick={{ fontSize: 10 }}
              angle={-30}
              textAnchor="end"
              interval={0}
            />
            <YAxis tick={{ fontSize: 10 }} unit="%" />
            <Tooltip
              contentStyle={{
                fontSize: "12px",
                borderRadius: 6,
                border: "1px solid #e2e8f0",
              }}
              formatter={(v: any) => `${(v as number).toFixed(2)}%`}
              labelFormatter={(_, payload) => {
                const d: any = payload?.[0]?.payload;
                if (!d) return "";
                return `${d.source} · ${d.creator}${d.isExact ? "  (exact ε=0)" : ""}`;
              }}
            />
            <Bar dataKey="weight" fill="#0d9488" radius={[2, 2, 0, 0]}>
              <ErrorBar dataKey="err" width={4} stroke="#475569" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 rounded-md border border-border bg-slate-50 p-2.5 text-[11px] leading-snug text-subt">
        <div>
          <span className="font-semibold text-ink">Why each bar caps below 100%:</span>{" "}
          each bar is one source's individual share of the total payout — they
          aren't stacked. The 100%-stacked bar at the top shows how all shares
          add up to exactly 100% (Z3-verified efficiency axiom).
        </div>
        <div className="mt-1">
          <span className="font-semibold text-ink">Why error bars look invisible:</span>{" "}
          {allExact ? (
            <>
              all clusters here ran <b>exact 2ⁿ Shapley</b> (n ≤ 10), so the
              Shapley value is computed by full enumeration with no
              statistical uncertainty — interval{" "}
              <code className="font-mono">[lower, upper]</code> collapses to a
              point.
            </>
          ) : (
            <>
              clusters with n ≤ 10 use exact Shapley (zero uncertainty);
              clusters with n &gt; 10 fall back to Monte Carlo with{" "}
              <b>Hoeffding-bounded intervals</b>{" "}
              ε = R · √(ln(2/α)/(2T)). Hover any bar to see its bound.
            </>
          )}
        </div>
      </div>
    </div>
  );
}
