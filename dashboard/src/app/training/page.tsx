"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import KpiTile from "@/components/KpiTile";
import { useResults, useTraining } from "@/lib/api";

export default function TrainingPage() {
  const { data } = useTraining();
  const { data: results } = useResults();
  if (!data) return <div className="text-sm text-subt">loading…</div>;

  const last = data[data.length - 1];
  const first = data[0];
  const totalTime = data.reduce((a, e) => a + e.elapsed_sec, 0);
  const cv = (results || []).filter((r) => r.method === "cascade_v");
  const axiomPass = cv.length
    ? (cv.reduce((a, r) => a + r.axioms_proven, 0) /
        Math.max(1, cv.reduce((a, r) => a + r.axioms_total, 0))) *
      100
    : NaN;

  // Has the richer metrics?
  const hasRich = data.some((d) => d.contrastive_loss !== undefined);

  // Smoothed loss (EMA, 0.9)
  const smoothed = data.reduce<{ epoch: number; loss: number; smooth: number }[]>(
    (acc, d, i) => {
      const prev = acc[i - 1]?.smooth ?? d.loss;
      acc.push({ epoch: d.epoch, loss: d.loss, smooth: 0.9 * prev + 0.1 * d.loss });
      return acc;
    },
    [],
  );

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <KpiTile icon="△" label="Epochs" value={data.length} tone="violet" />
        <KpiTile
          icon="↘"
          label="Final loss"
          value={last.loss.toFixed(4)}
          sub={`from ${first.loss.toFixed(4)}  ·  ${(((first.loss - last.loss) / first.loss) * 100).toFixed(1)}% drop`}
          tone="mint"
        />
        <KpiTile
          icon="⏱"
          label="Wall time"
          value={`${(totalTime / 60).toFixed(1)} min`}
          sub={`${(totalTime / data.length).toFixed(2)}s / epoch`}
          tone="sky"
        />
        <KpiTile
          icon="⚖"
          label="Axiom pass-rate"
          value={isNaN(axiomPass) ? "—" : `${axiomPass.toFixed(1)}%`}
          sub="downstream eval signal"
          tone="amber"
        />
      </div>

      {/* Loss curve with smoothed overlay */}
      <ChartCard
        title="Total loss"
        subtitle="NT-Xent + mix-consistency, with EMA-smoothed overlay (β = 0.9)"
      >
        <LineChart data={smoothed} margin={M}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v: any) => (v as number).toFixed(4)} />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            name="raw"
            stroke="#a7f3d0"
            dot={false}
            strokeWidth={1}
          />
          <Line
            type="monotone"
            dataKey="smooth"
            name="EMA"
            stroke="#0d9488"
            dot={false}
            strokeWidth={2.5}
          />
        </LineChart>
      </ChartCard>

      {hasRich && (
        <>
          <ChartCard
            title="Loss components"
            subtitle="NT-Xent contrastive vs mix-consistency cosine — both should decrease in tandem"
          >
            <LineChart data={data} margin={M}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: any) => (v as number).toFixed(4)} />
              <Legend />
              <Line
                type="monotone"
                dataKey="contrastive_loss"
                name="NT-Xent (contrastive)"
                stroke="#7c3aed"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="mix_loss"
                name="Mix-consistency"
                stroke="#f59e0b"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ChartCard>

          <ChartCard
            title="Mix consistency cosine"
            subtitle="cos(emb(mix), normalize(Σ wᵢ emb(sᵢ))). 1.0 = encoder is perfectly additive — exactly the property our value function assumes."
          >
            <LineChart data={data} margin={M}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                domain={[0, 1]}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip formatter={(v: any) => (v as number).toFixed(4)} />
              <Legend />
              <Line
                type="monotone"
                dataKey="mix_cosine"
                name="cos(mix, Σ contributors)"
                stroke="#0ea5e9"
                dot={false}
                strokeWidth={2.5}
              />
            </LineChart>
          </ChartCard>

          <ChartCard
            title="Embedding alignment & uniformity"
            subtitle="Wang & Isola 2020 — alignment: ‖emb_aug₁ − emb_aug₂‖² for positive pairs (lower is better). Uniformity: spread of negatives on the hypersphere (lower is better)."
          >
            <LineChart data={data} margin={M}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: any) => (v as number).toFixed(4)} />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="align"
                name="alignment"
                stroke="#10b981"
                dot={false}
                strokeWidth={2}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="uniformity"
                name="uniformity"
                stroke="#ef4444"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ChartCard>
        </>
      )}

      <ChartCard
        title="Learning rate schedule"
        subtitle="Cosine annealing from 1e-3 → 0"
      >
        <LineChart data={data} margin={M}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v: any) => (v as number).toExponential(2)} />
          <Legend />
          <Line
            type="monotone"
            dataKey="lr"
            stroke="#3b82f6"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ChartCard>

      <ChartCard
        title="Time per epoch"
        subtitle="Wall-clock seconds — sanity check for hardware utilization"
      >
        <LineChart data={data} margin={M}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} unit="s" />
          <Tooltip formatter={(v: any) => `${(v as number).toFixed(2)}s`} />
          <Legend />
          <Line
            type="monotone"
            dataKey="elapsed_sec"
            name="seconds/epoch"
            stroke="#a855f7"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ChartCard>

      {!hasRich && (
        <div className="card p-3 text-xs text-subt">
          The loss-component, alignment, uniformity, and mix-cosine charts
          appear after the next training run with v0.2.0+ logging. Re-run{" "}
          <code className="kbd">cascade-train --force</code>.
        </div>
      )}
    </div>
  );
}

const M = { top: 10, right: 10, left: 0, bottom: 0 };

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactElement;
}) {
  return (
    <div className="card p-4">
      <div className="mb-2">
        <div className="text-sm font-semibold">{title}</div>
        <div className="text-[11px] text-subt">{subtitle}</div>
      </div>
      <div className="h-64">
        <ResponsiveContainer>{children}</ResponsiveContainer>
      </div>
    </div>
  );
}
