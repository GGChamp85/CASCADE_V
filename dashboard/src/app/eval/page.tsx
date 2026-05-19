"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import KpiTile from "@/components/KpiTile";
import { useResults } from "@/lib/api";

const METHODS = ["cascade_v", "shapley_alone", "loo_alone", "trak_alone"];
const COLORS: Record<string, string> = {
  cascade_v: "#0d9488",
  shapley_alone: "#d97706",
  loo_alone: "#3b82f6",
  trak_alone: "#ef4444",
};

export default function EvalPage() {
  const { data } = useResults();
  if (!data) return <div className="text-sm text-subt">loading…</div>;

  const byMethod: Record<string, typeof data> = {};
  for (const r of data) {
    (byMethod[r.method] = byMethod[r.method] || []).push(r);
  }

  const stat = (rows: typeof data, key: keyof (typeof data)[number]) => {
    if (!rows.length) return 0;
    const xs = rows.map((r) =>
      typeof r[key] === "boolean" ? (r[key] ? 1 : 0) : (r[key] as number),
    );
    return xs.reduce((a, b) => a + b, 0) / xs.length;
  };

  const summary = METHODS.filter((m) => byMethod[m]).map((m) => {
    const rows = byMethod[m];
    const dna = rows.filter((r) => r.is_dna_case);
    return {
      method: m,
      n: rows.length,
      instance_mae: stat(rows, "instance_mae"),
      creator_mae: stat(rows, "creator_mae"),
      dna_creator_mae: dna.length ? stat(dna, "creator_mae") : 0,
      top1: stat(rows, "top1_hit") * 100,
      coverage: stat(rows, "coverage_at_k") * 100,
      axiom: rows.reduce((a, r) => a + r.axioms_proven, 0) /
        Math.max(1, rows.reduce((a, r) => a + r.axioms_total, 0)) * 100,
    };
  });

  const cv = summary.find((s) => s.method === "cascade_v");
  const shp = summary.find((s) => s.method === "shapley_alone");

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          icon="⚖"
          label="CASCADE-V axiom pass-rate"
          value={cv ? `${cv.axiom.toFixed(1)}%` : "—"}
          tone="mint"
        />
        <KpiTile
          icon="◎"
          label="Top-1 hit (CASCADE-V)"
          value={cv ? `${cv.top1.toFixed(1)}%` : "—"}
          sub={
            cv && shp
              ? `+${(cv.top1 - shp.top1).toFixed(1)}% vs Shapley alone`
              : null
          }
          tone="violet"
        />
        <KpiTile
          icon="≡"
          label="Creator MAE"
          value={cv ? cv.creator_mae.toFixed(4) : "—"}
          sub={
            cv && shp
              ? `${(((shp.creator_mae - cv.creator_mae) / shp.creator_mae) * 100).toFixed(0)}% better than Shapley`
              : null
          }
          tone="coral"
        />
        <KpiTile
          icon="◐"
          label="DNA Creator MAE"
          value={cv ? cv.dna_creator_mae.toFixed(4) : "—"}
          sub="creator-DNA test cases"
          tone="amber"
        />
      </div>

      <ChartCard title="Instance-level MAE (lower = better)">
        <BarChart data={summary} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="method" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="instance_mae" name="Instance MAE">
            {summary.map((s) => (
              <Cell key={s.method} fill={COLORS[s.method]} />
            ))}
          </Bar>
        </BarChart>
      </ChartCard>

      <ChartCard title="Creator-level MAE (lower = better)">
        <BarChart data={summary} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="method" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="creator_mae" name="Creator MAE">
            {summary.map((s) => (
              <Cell key={s.method} fill={COLORS[s.method]} />
            ))}
          </Bar>
          <Bar dataKey="dna_creator_mae" name="DNA Creator MAE" fill="#94a3b8" />
        </BarChart>
      </ChartCard>

      <ChartCard title="Coverage @ K (higher = better)">
        <BarChart data={summary} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="method" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} unit="%" />
          <Tooltip />
          <Legend />
          <Bar dataKey="coverage" name="Coverage@K">
            {summary.map((s) => (
              <Cell key={s.method} fill={COLORS[s.method]} />
            ))}
          </Bar>
          <Bar dataKey="top1" name="Top-1 hit" fill="#94a3b8" />
        </BarChart>
      </ChartCard>

      <div className="card overflow-hidden">
        <div className="border-b border-border px-4 py-2 text-sm font-semibold">
          Summary table
        </div>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-subt">
            <tr>
              <th className="px-3 py-1.5 text-left font-medium">Method</th>
              <th className="px-3 py-1.5 text-right font-medium">Instance MAE</th>
              <th className="px-3 py-1.5 text-right font-medium">Creator MAE</th>
              <th className="px-3 py-1.5 text-right font-medium">DNA Creator MAE</th>
              <th className="px-3 py-1.5 text-right font-medium">Top-1</th>
              <th className="px-3 py-1.5 text-right font-medium">Coverage@K</th>
              <th className="px-3 py-1.5 text-right font-medium">Axioms</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((s) => (
              <tr key={s.method} className="border-t border-border">
                <td className="px-3 py-1.5 font-mono">{s.method}</td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {s.instance_mae.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {s.creator_mae.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {s.dna_creator_mae.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {s.top1.toFixed(1)}%
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {s.coverage.toFixed(1)}%
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {s.axiom > 0 ? `${s.axiom.toFixed(1)}%` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactElement;
}) {
  return (
    <div className="card p-4">
      <div className="mb-2 text-sm font-semibold">{title}</div>
      <div className="h-64">
        <ResponsiveContainer>{children}</ResponsiveContainer>
      </div>
    </div>
  );
}
