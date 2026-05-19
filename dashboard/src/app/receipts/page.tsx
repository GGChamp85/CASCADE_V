"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import StatusModal from "@/components/StatusModal";
import { useReceipts } from "@/lib/api";

export default function ReceiptsList() {
  const { data, isLoading, error } = useReceipts();
  const [filter, setFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "PROVEN" | "VIOLATED">("all");
  const [modalId, setModalId] = useState<string | null>(null);

  const rows = useMemo(() => {
    return (data || [])
      .filter((r) =>
        filter
          ? r.receipt_id.toLowerCase().includes(filter.toLowerCase()) ||
            (r.top_creator || "").toLowerCase().includes(filter.toLowerCase())
          : true,
      )
      .filter((r) =>
        statusFilter === "all" ? true : r.overall_status === statusFilter,
      );
  }, [data, filter, statusFilter]);

  return (
    <div className="flex flex-col gap-4">
      <div className="card flex flex-wrap items-center gap-3 p-3">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter receipts or creators…"
          className="rounded-md border border-border bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as any)}
          className="rounded-md border border-border bg-white px-3 py-1.5 text-sm"
        >
          <option value="all">All statuses</option>
          <option value="PROVEN">Proven</option>
          <option value="VIOLATED">Violated</option>
        </select>
        <span className="text-xs text-subt">
          {rows.length} of {data?.length ?? 0} receipts · click any status pill
          for the verdict reasoning
        </span>
      </div>
      {isLoading && <div className="text-sm text-subt">loading…</div>}
      {error && (
        <div className="card border-red-200 bg-red-50 p-3 text-sm text-red-700">
          ✗ {(error as Error).message}
        </div>
      )}
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-subt">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Receipt</th>
              <th className="px-3 py-2 text-left font-medium">Top creator</th>
              <th className="px-3 py-2 text-right font-medium">Weight</th>
              <th className="px-3 py-2 text-right font-medium">Triaged</th>
              <th className="px-3 py-2 text-right font-medium">Clusters</th>
              <th className="px-3 py-2 text-right font-medium">Latency</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.receipt_id}
                className="border-t border-border hover:bg-slate-50"
              >
                <td className="px-3 py-1.5 font-mono">
                  <Link
                    href={`/receipts/${r.receipt_id}`}
                    className="text-violet-700 hover:underline"
                  >
                    {r.receipt_id}
                  </Link>
                </td>
                <td className="px-3 py-1.5">{r.top_creator || "—"}</td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {r.top_creator_weight != null
                    ? `${(r.top_creator_weight * 100).toFixed(1)}%`
                    : "—"}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {r.candidates_from_triage ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {r.n_clusters ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {r.total_latency_ms != null
                    ? `${r.total_latency_ms.toFixed(1)}ms`
                    : "—"}
                </td>
                <td className="px-3 py-1.5">
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
      {modalId && (
        <StatusModal receiptId={modalId} onClose={() => setModalId(null)} />
      )}
    </div>
  );
}
