"use client";

import Link from "next/link";
import { useEffect } from "react";

import { useReceipt } from "@/lib/api";

const AXIOM_INFO: Record<
  string,
  { title: string; what: string; why_proven: string; why_violated: string }
> = {
  efficiency: {
    title: "Efficiency",
    what: "All payouts together must equal exactly 100% — no money left on the table.",
    why_proven: "Σ wᵢ is within tolerance of 1.0; every dollar is accounted for.",
    why_violated:
      "Σ wᵢ deviates from 1.0 beyond tolerance — the receipt allocates more or less than 100% of the total payout.",
  },
  symmetry: {
    title: "Symmetry",
    what: "Sources whose embeddings are near-identical (cosine ≥ 0.99) must receive equal payouts — no favoritism between equivalent contributors.",
    why_proven:
      "Every pair of near-identical sources has equal weight (within 0.001).",
    why_violated:
      "At least one near-identical pair received different weights — the attribution is treating equivalent sources unfairly.",
  },
  dummy: {
    title: "Null player",
    what: "Sources with zero marginal contribution to the target must receive 0% — no credit for free riders.",
    why_proven:
      "Every flagged null source has zero weight; no free riders got paid.",
    why_violated:
      "At least one source with no marginal contribution still received a non-zero payout.",
  },
};

export default function StatusModal({
  receiptId,
  onClose,
}: {
  receiptId: string;
  onClose: () => void;
}) {
  const { data, isLoading } = useReceipt(receiptId);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-slate-900/50 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="card mt-12 w-full max-w-3xl"
        onClick={(e) => e.stopPropagation()}
      >
        {isLoading || !data ? (
          <div className="p-6 text-sm text-subt">loading verdict…</div>
        ) : (
          <ModalBody data={data} onClose={onClose} />
        )}
      </div>
    </div>
  );
}

function ModalBody({ data, onClose }: { data: any; onClose: () => void }) {
  const proven = data.verification.overall_status === "PROVEN";
  const axioms = data.verification.axioms;
  const passed = axioms.filter((a: any) => a.status === "PROVEN").length;
  const checked = axioms.filter((a: any) => a.status !== "NA").length;
  const violations = axioms.filter((a: any) => a.status === "VIOLATED");

  return (
    <div className="flex flex-col">
      <div
        className={
          "flex items-start gap-4 rounded-t-2xl border-b-2 p-5 " +
          (proven
            ? "border-emerald-300 bg-emerald-50"
            : "border-red-300 bg-red-50")
        }
      >
        <div
          className={
            "flex h-14 w-14 shrink-0 items-center justify-center rounded-full text-3xl text-white shadow-soft " +
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
          <div className="mt-1 text-sm text-subt">
            <span className="font-mono">{data.receipt_id}</span> ·{" "}
            {passed} of {checked} axioms verified by Z3
            {data.latency_ms?.total != null && (
              <>
                {" "}
                ·{" "}
                <span className="font-mono">
                  {data.latency_ms.total.toFixed(1)}ms
                </span>
              </>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="rounded-full p-1 text-slate-500 hover:bg-slate-200 hover:text-slate-800"
          aria-label="Close"
        >
          ✕
        </button>
      </div>

      <div className="p-5">
        <div className="mb-4 rounded-xl bg-slate-50 p-4 text-sm leading-relaxed text-ink">
          <div className="text-xs font-semibold uppercase tracking-wider text-subt">
            Why this verdict?
          </div>
          <p className="mt-1">
            {proven ? (
              <>
                Z3 (Microsoft Research's SMT solver) re-checked the actual
                numerical weights against the three Shapley fairness axioms and
                confirmed they all hold within tolerance.{" "}
                <span className="font-medium">
                  This receipt is safe to pay out.
                </span>{" "}
                Anyone with z3 installed can re-run the same check on the SMT
                file independently.
              </>
            ) : (
              <>
                Z3 found{" "}
                <span className="font-semibold text-red-700">
                  {violations.length} axiom violation
                  {violations.length === 1 ? "" : "s"}
                </span>
                .{" "}
                <span className="font-medium">
                  This receipt should NOT be paid out as-is.
                </span>{" "}
                See the per-axiom breakdown below for the specific failure(s).
                In production this would block the payout and flag the case for
                review.
              </>
            )}
          </p>
        </div>

        <div className="flex flex-col gap-3">
          {axioms.map((ax: any) => {
            const info = AXIOM_INFO[ax.name] || {
              title: ax.name,
              what: ax.detail,
              why_proven: ax.detail,
              why_violated: ax.detail,
            };
            const tone =
              ax.status === "PROVEN"
                ? "border-emerald-200 bg-emerald-50/50"
                : ax.status === "VIOLATED"
                  ? "border-red-200 bg-red-50/50"
                  : "border-border bg-slate-50/50";
            const dot =
              ax.status === "PROVEN"
                ? "bg-emerald-500"
                : ax.status === "VIOLATED"
                  ? "bg-red-500"
                  : "bg-slate-400";
            return (
              <div
                key={ax.name}
                className={`flex items-start gap-3 rounded-xl border p-4 ${tone}`}
              >
                <div
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm text-white ${dot}`}
                >
                  {ax.status === "PROVEN"
                    ? "✓"
                    : ax.status === "VIOLATED"
                      ? "✗"
                      : "—"}
                </div>
                <div className="flex flex-1 flex-col gap-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-semibold">{info.title}</div>
                    {ax.status === "PROVEN" && (
                      <span className="pill pill-ok">PROVEN</span>
                    )}
                    {ax.status === "VIOLATED" && (
                      <span className="pill pill-bad">VIOLATED</span>
                    )}
                    {ax.status === "NA" && (
                      <span className="pill pill-mute">N/A</span>
                    )}
                  </div>
                  <div className="text-xs text-subt">{info.what}</div>
                  <div className="rounded-md bg-white p-2 text-xs leading-relaxed">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-subt">
                      {ax.status === "PROVEN"
                        ? "Why it passed"
                        : ax.status === "VIOLATED"
                          ? "Why it failed"
                          : "Why N/A"}
                    </div>
                    <div className="mt-0.5">
                      {ax.status === "PROVEN"
                        ? info.why_proven
                        : ax.status === "VIOLATED"
                          ? info.why_violated
                          : "This axiom doesn't apply to this attribution (e.g. only one source, or no equivalent pairs)."}
                    </div>
                    <div className="mt-1.5 font-mono text-[10px] text-subt">
                      Z3 detail: {ax.detail}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-5 rounded-xl border border-border bg-slate-50 p-3 font-mono text-[11px] text-subt">
          <div className="font-semibold text-ink">Independent verification</div>
          <div className="mt-1">
            $ brew install z3 # one-time
            <br />
            $ z3 outputs/proofs/{data.receipt_id}.smt2
            <br />
            sat # ← Z3 confirms the constraints hold
          </div>
          <div className="mt-1.5">
            SMT-LIB sha256:{" "}
            <span className="text-ink">
              {data.verification.smt_lib_hash.slice(0, 24)}…
            </span>
          </div>
        </div>

        <div className="mt-5 flex items-center justify-end gap-2">
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
          <Link
            href={`/receipts/${data.receipt_id}`}
            className="btn-primary"
            onClick={onClose}
          >
            Open full receipt →
          </Link>
        </div>
      </div>
    </div>
  );
}
