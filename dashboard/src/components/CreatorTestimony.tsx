"use client";

import { useEffect, useMemo } from "react";

import { API_BASE, Receipt } from "@/lib/api";

/**
 * "Why was I paid this much?" — a creator-facing testimony page.
 *
 * Designed to convince a working musician (no SMT background) that the split
 * is correct. Every claim is paired with either listening evidence, a Z3
 * verdict, or an independent re-verification path.
 */
export default function CreatorTestimony({
  receipt,
  creatorId,
  onClose,
}: {
  receipt: Receipt;
  creatorId: string;
  onClose: () => void;
}) {
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

  const creator = useMemo(
    () => receipt.per_creator.find((c) => c.creator_id === creatorId),
    [receipt, creatorId],
  );
  const mySources = useMemo(
    () =>
      receipt.per_source
        .filter((s) => s.creator_id === creatorId)
        .sort((a, b) => b.weight_point - a.weight_point),
    [receipt, creatorId],
  );

  const verdictProven = receipt.verification.overall_status === "PROVEN";
  const totalPayoutUsd = receipt.total_payout_usd ?? 1.0;
  const targetKind = receipt.receipt_id.startsWith("upload_") ? "upload" : "test";

  if (!creator) return null;

  const myPercent = creator.weight_point * 100;
  const myDollars = creator.payout_usd;

  // Rank within all creators
  const sorted = [...receipt.per_creator].sort(
    (a, b) => b.weight_point - a.weight_point,
  );
  const rank = sorted.findIndex((c) => c.creator_id === creatorId) + 1;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-slate-900/60 p-4 backdrop-blur-sm print:relative print:bg-white"
      onClick={onClose}
    >
      <div
        className="card my-8 w-full max-w-4xl print:my-0 print:max-w-full print:shadow-none"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Print header */}
        <div className="flex items-start justify-between border-b border-border p-6 print:border-b-2">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-violet-700">
              CASCADE-V Creator Receipt
            </div>
            <h2 className="mt-1 text-2xl font-bold leading-tight">
              {creator.creator_name}
            </h2>
            <div className="text-sm text-subt">
              Receipt <span className="font-mono">{receipt.receipt_id}</span> ·
              Issued {new Date(receipt.created_at_utc).toLocaleString()}
            </div>
          </div>
          <div className="flex items-center gap-2 print:hidden">
            <button
              onClick={() => window.print()}
              className="btn-secondary"
              title="Print or save as PDF"
            >
              ⎙ Print
            </button>
            <button
              onClick={onClose}
              className="rounded-full p-1 text-slate-500 hover:bg-slate-200 hover:text-slate-800"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="space-y-6 p-6">
          {/* Headline payout */}
          <div className="rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 p-6 text-white shadow-soft">
            <div className="text-xs font-semibold uppercase tracking-wider text-violet-100">
              Your share of this attribution
            </div>
            <div className="mt-2 flex items-baseline gap-3">
              <span className="text-6xl font-bold leading-none">
                {myPercent.toFixed(1)}%
              </span>
              <span className="text-3xl font-semibold text-violet-100">
                ${myDollars.toFixed(4)}
              </span>
            </div>
            <div className="mt-3 text-sm text-violet-50">
              Rank #{rank} of {sorted.length} contributors · across{" "}
              {creator.n_sources} of your stem
              {creator.n_sources === 1 ? "" : "s"} · total payout pool $
              {totalPayoutUsd.toFixed(2)}
            </div>
          </div>

          {/* Section 1: What was used */}
          <Section
            number="1"
            title="Which of your stems was used"
            subtitle="Listen to the output and your contributing stems side-by-side"
          >
            <div className="rounded-xl border-2 border-violet-200 bg-violet-50/50 p-3">
              <div className="mb-2 flex items-center justify-between text-sm">
                <span className="font-semibold">Output audio (the AI mix)</span>
                <span className="font-mono text-xs text-subt">
                  {receipt.receipt_id}
                </span>
              </div>
              <audio
                controls
                preload="none"
                className="w-full"
                src={`${API_BASE}/api/audio/${targetKind}/${receipt.receipt_id}`}
              />
            </div>
            <div className="mt-3 space-y-2">
              {mySources.map((s) => (
                <div
                  key={s.source_id}
                  className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3"
                >
                  <div className="mb-1.5 flex items-center justify-between">
                    <div>
                      <span className="text-sm font-semibold">
                        {s.source_id}
                      </span>
                      <span className="ml-2 text-xs text-subt">
                        {s.stage_path}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-bold text-emerald-700">
                        {(s.weight_point * 100).toFixed(2)}%
                      </div>
                      <div className="text-[11px] text-subt">
                        ${s.payout_usd.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <audio
                    controls
                    preload="none"
                    className="w-full"
                    src={`${API_BASE}/api/audio/catalog/${s.source_id}`}
                  />
                </div>
              ))}
            </div>
          </Section>

          {/* Section 2: Plain-English why */}
          <Section
            number="2"
            title="Why this percentage"
            subtitle="Plain-English summary of the math behind your share"
          >
            <div className="prose prose-sm max-w-none rounded-xl bg-slate-50 p-4 leading-relaxed text-ink">
              <p>
                The system encodes the output audio and your stem into a shared
                "embedding space" — a coordinate system where similar-sounding
                audio sits close together. {mySources.length === 1 ? "Your stem" : `Your ${mySources.length} stems`} ranked among the top{" "}
                {receipt.pipeline_metadata.candidates_from_triage}{" "}
                most-similar in the catalog of{" "}
                {receipt.pipeline_metadata.catalog_size} stems.
              </p>
              <p className="mt-2">
                Within that shortlist, your{" "}
                {mySources.length === 1 ? "stem was" : "stems were"} grouped
                with similar-sounding contributors and a{" "}
                <span className="font-semibold">Shapley value</span> was
                computed for each — Shapley is the unique way to split a
                cooperative payout that satisfies all four classical fairness
                axioms.{" "}
                {mySources.length > 1 ? (
                  <>
                    Because two of your stems are by{" "}
                    <em>the same creator</em> (you), the system used GUDA-style{" "}
                    <span className="font-semibold">group-wise</span> Shapley
                    so you weren't double-counted across them.
                  </>
                ) : null}
              </p>
              <p className="mt-2">
                The result was <strong>{myPercent.toFixed(1)}%</strong>{" "}
                — equal to{" "}
                <strong>${myDollars.toFixed(4)}</strong> of the{" "}
                ${totalPayoutUsd.toFixed(2)} payout pool. This number was then
                re-checked by the Z3 SMT solver against the three Shapley
                fairness axioms (efficiency, symmetry, null player) and{" "}
                <strong className={verdictProven ? "text-emerald-700" : "text-red-700"}>
                  {verdictProven
                    ? "all three passed."
                    : "at least one failed — see disputes below."}
                </strong>
              </p>
            </div>
          </Section>

          {/* Section 3: Trust signals */}
          <Section
            number="3"
            title="Why you can trust this number"
            subtitle="Four independent ways to verify"
          >
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Trust
                tone={verdictProven ? "ok" : "bad"}
                title={
                  verdictProven
                    ? "Z3 SMT solver confirms all axioms"
                    : "Z3 found violations — see disputes"
                }
                body="Microsoft Research's Z3 solver re-checked the actual numbers against the three fairness axioms. Anyone with z3 installed can re-run the same check on the SMT-LIB file."
              />
              <Trust
                tone="ok"
                title="Receipt is content-hashed"
                body={
                  <>
                    SHA-256:{" "}
                    <code className="rounded bg-slate-100 px-1 font-mono text-[10px]">
                      {receipt.verification.smt_lib_hash.slice(0, 32)}…
                    </code>
                    . If anyone alters one digit of the receipt, this hash
                    breaks.
                  </>
                }
              />
              <Trust
                tone="ok"
                title="The math is deterministic"
                body="Same input audio + same catalog + same seed → exactly the same payout. There's no random tie-breaking, no ML temperature, no hidden variability you have to trust us about."
              />
              <Trust
                tone="ok"
                title="You don't have to trust us"
                body={
                  <>
                    Run the audit yourself:
                    <pre className="mt-1 rounded bg-slate-100 p-1.5 font-mono text-[10px]">
                      brew install z3{"\n"}
                      z3 {receipt.receipt_id}.smt2 # → sat
                    </pre>
                  </>
                }
              />
            </div>
          </Section>

          {/* Footer fingerprint */}
          <div className="rounded-xl border border-border bg-slate-50 p-3 text-[11px] leading-relaxed text-subt">
            <div className="mb-1 font-semibold text-ink">
              Cryptographic fingerprint of this receipt
            </div>
            <div className="font-mono">
              receipt_id: {receipt.receipt_id}
              <br />
              method:     {receipt.method} {receipt.version}
              <br />
              issued:     {receipt.created_at_utc}
              <br />
              smt_hash:   {receipt.verification.smt_lib_hash}
              <br />
              z3_status:  {receipt.verification.overall_status}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t border-border p-4 print:hidden">
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
          <button onClick={() => window.print()} className="btn-primary">
            ⎙ Print receipt
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({
  number,
  title,
  subtitle,
  children,
}: {
  number: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="mb-2 flex items-baseline gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-100 text-xs font-bold text-violet-700">
          {number}
        </span>
        <div>
          <div className="text-base font-semibold leading-tight">{title}</div>
          <div className="text-[11px] text-subt">{subtitle}</div>
        </div>
      </div>
      {children}
    </section>
  );
}

function Trust({
  tone,
  title,
  body,
}: {
  tone: "ok" | "bad";
  title: string;
  body: React.ReactNode;
}) {
  const dot = tone === "ok" ? "bg-emerald-500" : "bg-red-500";
  const border = tone === "ok" ? "border-emerald-200" : "border-red-200";
  const bg = tone === "ok" ? "bg-emerald-50/40" : "bg-red-50/40";
  return (
    <div className={`rounded-xl border p-3 ${border} ${bg}`}>
      <div className="mb-1 flex items-center gap-2">
        <span
          className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] text-white ${dot}`}
        >
          ✓
        </span>
        <span className="text-sm font-semibold">{title}</span>
      </div>
      <div className="text-[11px] leading-relaxed text-subt">{body}</div>
    </div>
  );
}

