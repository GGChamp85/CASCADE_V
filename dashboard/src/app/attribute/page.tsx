"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import AudioPlayer from "@/components/AudioPlayer";
import ContributorStack from "@/components/ContributorStack";
import GroundTruthCompare from "@/components/GroundTruthCompare";
import LiveAttributionStream, {
  StreamSource,
} from "@/components/LiveAttributionStream";
import PerCreatorDonut from "@/components/PerCreatorDonut";
import PerSourceBars from "@/components/PerSourceBars";
import StagePipelineView from "@/components/StagePipelineView";
import VerdictBanner from "@/components/VerdictBanner";
import { Receipt, useTestOutputs } from "@/lib/api";

export default function AttributePage() {
  const { data: outputs } = useTestOutputs();
  const [selected, setSelected] = useState<string>("output_001");
  const [upload, setUpload] = useState<File | null>(null);
  const [source, setSource] = useState<StreamSource | null>(null);
  const [runToken, setRunToken] = useState(0);
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [drag, setDrag] = useState(false);

  const selectedGT = useMemo(
    () => (outputs || []).find((o) => o.output_id === selected),
    [outputs, selected],
  );

  const runTest = () => {
    setReceipt(null);
    setSource({ kind: "test", output_id: selected });
    setRunToken((n) => n + 1);
  };

  const runUpload = () => {
    if (!upload) return;
    setReceipt(null);
    setSource({ kind: "upload", file: upload });
    setRunToken((n) => n + 1);
  };

  const onDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.name.toLowerCase().endsWith(".wav")) setUpload(f);
  };

  const inputBadge = source
    ? source.kind === "test"
      ? source.output_id
      : (source.file as File).name
    : null;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-ink">Live attribution</h1>
        <p className="text-sm text-subt">
          Pick a known test output OR drop in your own WAV. Watch the 5-stage
          pipeline run and see the Z3-verified payout receipt at the bottom.
        </p>
      </div>

      {/* Step 1: Input */}
      <section className="card p-5">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-rose-400 to-orange-500 text-base font-bold text-white shadow-soft">
            1
          </div>
          <div>
            <div className="text-base font-semibold">Choose input audio</div>
            <div className="text-xs text-subt">
              Either a generated test output (with known ground truth) or your
              own WAV — encoder will embed it.
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Test outputs */}
          <div className="rounded-xl border border-border bg-slate-50/60 p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-subt">
                Test outputs · 30 available
              </span>
              {selectedGT && (
                <span className="font-mono text-[11px] text-subt">
                  GT: {selectedGT.source_ids.length} contributors
                </span>
              )}
            </div>
            <div className="mb-3 flex max-h-40 flex-wrap gap-2 overflow-auto">
              {(outputs || []).map((o) => (
                <button
                  key={o.output_id}
                  onClick={() => setSelected(o.output_id)}
                  className={
                    "rounded-full border px-3 py-1 font-mono text-xs transition-colors " +
                    (selected === o.output_id
                      ? "border-violet-500 bg-violet-100 text-violet-800"
                      : "border-border bg-white hover:bg-slate-100")
                  }
                >
                  {o.output_id}
                </button>
              ))}
            </div>
            <button
              onClick={runTest}
              className="btn-primary w-full justify-center"
            >
              ⏵ Run attribution on {selected}
            </button>
            {selectedGT && (
              <div className="mt-3">
                <AudioPlayer
                  kind="test"
                  name={selected}
                  label={`${selected}.wav`}
                />
              </div>
            )}
          </div>

          {/* Upload */}
          <div className="flex flex-col gap-3">
            <label
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              className={
                "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors " +
                (drag
                  ? "border-violet-400 bg-violet-50"
                  : "border-border bg-slate-50/60 hover:bg-slate-100")
              }
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 text-2xl text-white shadow-soft">
                ↥
              </div>
              <div className="text-sm font-semibold">
                {upload ? upload.name : "Drop your own .wav here"}
              </div>
              <div className="text-xs text-subt">
                Encoder accepts any sample rate · auto-resampled to 22.05 kHz mono
              </div>
              <input
                type="file"
                accept=".wav,audio/wav"
                className="hidden"
                onChange={(e) => setUpload(e.target.files?.[0] || null)}
              />
            </label>
            <button
              onClick={runUpload}
              disabled={!upload}
              className="btn-primary w-full justify-center"
            >
              ⏵ Run attribution on uploaded WAV
            </button>
          </div>
        </div>
      </section>

      {/* Step 2: Pipeline */}
      <section className="card p-5">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 text-base font-bold text-white shadow-soft">
            2
          </div>
          <div className="flex-1">
            <div className="text-base font-semibold">Pipeline execution</div>
            <div className="text-xs text-subt">
              Streams stage events live via Server-Sent Events from FastAPI →
              Python pipeline.
            </div>
          </div>
          {inputBadge && (
            <span className="pill pill-brand">running on {inputBadge}</span>
          )}
        </div>

        {!source && (
          <div className="rounded-xl border-2 border-dashed border-border bg-slate-50/60 p-8 text-center text-sm text-subt">
            ▷ Pick an input above and click <span className="font-semibold">Run attribution</span> to start the pipeline.
          </div>
        )}
        {source && (
          <LiveAttributionStream
            source={source}
            runToken={runToken}
            onComplete={setReceipt}
          />
        )}
      </section>

      {/* Step 3: Output */}
      <section className="card p-5">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-teal-500 text-base font-bold text-white shadow-soft">
            3
          </div>
          <div className="flex-1">
            <div className="text-base font-semibold">Output: payout receipt</div>
            <div className="text-xs text-subt">
              Z3-verified fairness certificate with per-creator payouts and
              confidence intervals.
            </div>
          </div>
          {receipt && (
            <Link
              href={`/receipts/${receipt.receipt_id}`}
              className="btn-secondary"
            >
              Open full receipt →
            </Link>
          )}
        </div>

        {!receipt && (
          <div className="rounded-xl border-2 border-dashed border-border bg-slate-50/60 p-8 text-center text-sm text-subt">
            ☼ Receipt will appear here once the pipeline completes.
          </div>
        )}
        {receipt && (
          <div className="flex flex-col gap-4">
            <VerdictBanner receipt={receipt} />
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <PerCreatorDonut receipt={receipt} />
              <div className="card flex flex-col gap-3 p-5 ring-1 ring-border">
                <div className="text-sm font-semibold">Target audio</div>
                <AudioPlayer
                  kind={
                    receipt.receipt_id.startsWith("upload_") ? "upload" : "test"
                  }
                  name={receipt.receipt_id}
                  label={receipt.receipt_id}
                />
                <div className="text-xs text-subt">
                  {receipt.pipeline_metadata.candidates_from_triage} candidates
                  → {receipt.pipeline_metadata.n_clusters} clusters · proof{" "}
                  <span className="font-mono">
                    {receipt.verification.smt_lib_hash.slice(0, 12)}…
                  </span>
                </div>
              </div>
            </div>
            <StagePipelineView receipt={receipt as any} />
            <ContributorStack receipt={receipt} />
            <GroundTruthCompare receipt={receipt} />
            <PerSourceBars receipt={receipt} />
          </div>
        )}
      </section>
    </div>
  );
}
