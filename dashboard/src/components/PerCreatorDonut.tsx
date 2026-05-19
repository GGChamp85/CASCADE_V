"use client";

import { useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import CreatorTestimony from "./CreatorTestimony";
import { Receipt } from "@/lib/api";

const COLORS = [
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
  "#06b6d4",
  "#84cc16",
  "#64748b",
];

export default function PerCreatorDonut({ receipt }: { receipt: Receipt }) {
  const data = receipt.per_creator
    .filter((c) => c.weight_point > 0)
    .map((c, i) => ({
      creator_id: c.creator_id,
      name: c.creator_name,
      value: c.weight_point,
      color: COLORS[i % COLORS.length],
    }));
  const total = data.reduce((a, d) => a + d.value, 0);
  const [openCreator, setOpenCreator] = useState<string | null>(null);

  return (
    <div className="card flex flex-col p-4">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold">Per-creator payout</div>
          <div className="text-[10px] text-subt">
            Click any creator for their "Why was I paid this?" testimony
          </div>
        </div>
        <span className="pill pill-ok font-mono">
          Σ {(total * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex items-start gap-4">
        <div className="h-44 w-44 shrink-0">
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                cx="50%"
                cy="50%"
                innerRadius={45}
                outerRadius={70}
                stroke="#fff"
                strokeWidth={1}
                onClick={(d: any) => setOpenCreator(d?.creator_id || null)}
                style={{ cursor: "pointer" }}
              >
                {data.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v: any) => `${(v * 100).toFixed(2)}%`}
                contentStyle={{
                  fontSize: "12px",
                  borderRadius: 6,
                  border: "1px solid #e2e8f0",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <ul className="flex max-h-44 flex-1 flex-col gap-1 overflow-auto text-sm">
          {data.map((d) => (
            <li key={d.creator_id}>
              <button
                onClick={() => setOpenCreator(d.creator_id)}
                className="flex w-full items-center justify-between gap-2 rounded-md px-1 py-0.5 text-left hover:bg-slate-50"
                title="Open creator's testimony"
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span
                    className="inline-block h-2 w-2 shrink-0 rounded-sm"
                    style={{ background: d.color }}
                  />
                  <span className="truncate">{d.name}</span>
                </span>
                <span className="font-mono text-subt">
                  {(d.value * 100).toFixed(1)}%
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="mt-3 border-t border-border pt-2 text-[11px] text-subt">
        {data.length} creators · sums to 100% by Z3-verified efficiency axiom ·
        <span className="ml-1 text-violet-700">click a name → creator testimony</span>
      </div>
      {openCreator && (
        <CreatorTestimony
          receipt={receipt}
          creatorId={openCreator}
          onClose={() => setOpenCreator(null)}
        />
      )}
    </div>
  );
}
