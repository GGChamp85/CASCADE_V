"use client";

import { useState } from "react";

import { useProof } from "@/lib/api";

export default function SmtViewer({ receiptId }: { receiptId: string }) {
  const { data, isLoading, error } = useProof(receiptId);
  const [expanded, setExpanded] = useState(false);
  const [view, setView] = useState<"annotated" | "raw">("annotated");

  return (
    <div className="card flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold">Z3 SMT proof certificate</div>
          <div className="text-xs text-subt">
            Z3 verdict:{" "}
            {data ? (
              <code className="rounded bg-slate-100 px-1 font-mono">
                {data.z3_verdict}
              </code>
            ) : (
              "…"
            )}{" "}
            · independent auditors can re-verify with{" "}
            <code className="rounded bg-slate-100 px-1 font-mono">
              z3 {receiptId}.smt2
            </code>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {expanded && data?.annotated && (
            <div className="flex rounded-md border border-border bg-white p-0.5 text-xs">
              <button
                onClick={() => setView("annotated")}
                className={
                  "rounded px-2.5 py-1 font-medium " +
                  (view === "annotated"
                    ? "bg-violet-100 text-violet-800"
                    : "text-subt hover:text-ink")
                }
              >
                Plain English
              </button>
              <button
                onClick={() => setView("raw")}
                className={
                  "rounded px-2.5 py-1 font-medium " +
                  (view === "raw"
                    ? "bg-violet-100 text-violet-800"
                    : "text-subt hover:text-ink")
                }
              >
                Raw SMT-LIB
              </button>
            </div>
          )}
          <button
            onClick={() => setExpanded((v) => !v)}
            className="btn-secondary !py-1.5"
          >
            {expanded ? "Hide" : "Show"} proof
          </button>
        </div>
      </div>
      {error && (
        <div className="text-xs text-red-600">
          failed to load proof: {(error as Error).message}
        </div>
      )}
      {isLoading && <div className="text-xs text-subt">loading proof…</div>}
      {expanded && data && (
        <pre className="max-h-[28rem] overflow-auto rounded-md border border-border bg-slate-50 p-3 font-mono text-[11px] leading-relaxed">
          {view === "annotated" && data.annotated
            ? data.annotated
            : data.smt_lib}
        </pre>
      )}
    </div>
  );
}
