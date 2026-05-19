"use client";

import { useState } from "react";

import StatusModal from "./StatusModal";
import { Receipt } from "@/lib/api";

const AXIOM_LABELS: Record<string, { title: string; tagline: string }> = {
  efficiency: {
    title: "Efficiency",
    tagline: "Per-source payouts sum to 100% — no money left on the table.",
  },
  symmetry: {
    title: "Symmetry",
    tagline:
      "Equivalent contributors receive equal credit — no favoritism between near-identical sources.",
  },
  dummy: {
    title: "Null player",
    tagline:
      "Sources that don't contribute are paid 0 — no credit for free riders.",
  },
};

export default function VerdictBanner({ receipt }: { receipt: Receipt }) {
  const proven = receipt.verification.overall_status === "PROVEN";
  const axioms = receipt.verification.axioms;
  const passed = axioms.filter((a) => a.status === "PROVEN").length;
  const checked = axioms.filter((a) => a.status !== "NA").length;
  const lat = receipt.latency_ms?.total;
  const top = receipt.per_creator?.[0];

  return (
    <div
      className={
        "rounded-lg border-2 p-5 " +
        (proven
          ? "border-emerald-300 bg-emerald-50"
          : "border-red-300 bg-red-50")
      }
    >
      <div className="flex items-start gap-4">
        <div
          className={
            "flex h-14 w-14 shrink-0 items-center justify-center rounded-full text-3xl text-white " +
            (proven ? "bg-emerald-500" : "bg-red-500")
          }
        >
          {proven ? "✓" : "✗"}
        </div>
        <div className="flex-1">
          <div
            className={
              "text-2xl font-bold " +
              (proven ? "text-emerald-800" : "text-red-800")
            }
          >
            {proven ? "PROVEN" : "VIOLATED"}
          </div>
          <div className="text-sm text-subt">
            {proven ? (
              <>
                Z3 verified all {checked} fairness axioms (efficiency, symmetry,
                dummy). The receipt is auditable.
              </>
            ) : (
              <>
                Z3 found {checked - passed} of {checked} axioms violated. The
                receipt should NOT be paid out as-is.
              </>
            )}
          </div>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="rounded-full border border-border bg-white px-2.5 py-1">
              <span className="text-subt">SMT hash</span>{" "}
              <span className="font-mono">
                {receipt.verification.smt_lib_hash.slice(0, 16)}…
              </span>
            </span>
            {lat != null && (
              <span className="rounded-full border border-border bg-white px-2.5 py-1">
                <span className="text-subt">latency</span>{" "}
                <span className="font-mono">{lat.toFixed(1)}ms</span>
              </span>
            )}
            {top && (
              <span className="rounded-full border border-border bg-white px-2.5 py-1">
                <span className="text-subt">top creator</span>{" "}
                <span className="font-medium">{top.creator_name}</span>{" "}
                <span className="font-mono">
                  ({(top.weight_point * 100).toFixed(1)}%)
                </span>
              </span>
            )}
            <span className="rounded-full border border-border bg-white px-2.5 py-1">
              <span className="text-subt">verify locally</span>{" "}
              <code className="font-mono">
                z3 {receipt.verification.smt_lib_file.split("/").pop()}
              </code>
            </span>
          </div>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
        {axioms.map((a) => {
          const info = AXIOM_LABELS[a.name] || {
            title: a.name,
            tagline: a.detail,
          };
          return (
            <div
              key={a.name}
              className={
                "flex items-start gap-3 rounded-md border bg-white p-3 " +
                (a.status === "PROVEN"
                  ? "border-emerald-200"
                  : a.status === "VIOLATED"
                    ? "border-red-200"
                    : "border-border")
              }
            >
              <div
                className={
                  "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm text-white " +
                  (a.status === "PROVEN"
                    ? "bg-emerald-500"
                    : a.status === "VIOLATED"
                      ? "bg-red-500"
                      : "bg-slate-400")
                }
              >
                {a.status === "PROVEN" ? "✓" : a.status === "VIOLATED" ? "✗" : "—"}
              </div>
              <div>
                <div className="text-sm font-medium">{info.title}</div>
                <div className="text-[11px] leading-snug text-subt">
                  {info.tagline}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
