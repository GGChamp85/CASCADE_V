"use client";

import { ReactNode, useState } from "react";

import StageInfoModal, { StageKey } from "./StageInfoModal";

export type StageStatus = "idle" | "running" | "done" | "fail";

const stages: Array<{
  key: StageKey;
  number: string;
  title: string;
  desc: string;
  icon: string;
  bg: string;
  ring: string;
  iconBg: string;
}> = [
  {
    key: "stage0",
    number: "STEP 0",
    title: "Separate",
    desc: "Demucs splits mix → drums/bass/other/vocals",
    icon: "✂",
    bg: "from-fuchsia-50 to-pink-50",
    ring: "ring-fuchsia-200",
    iconBg: "bg-gradient-to-br from-fuchsia-500 to-pink-600",
  },
  {
    key: "stage1",
    number: "STEP 1",
    title: "Triage",
    desc: "NNLS-first rank + cosine tie-break, optional BPM/key gate",
    icon: "◎",
    bg: "from-rose-50 to-orange-50",
    ring: "ring-rose-200",
    iconBg: "bg-gradient-to-br from-rose-400 to-orange-500",
  },
  {
    key: "stage2",
    number: "STEP 2",
    title: "Group",
    desc: "Ward / HDBSCAN + group counterfactual + sparsify",
    icon: "◫",
    bg: "from-violet-50 to-indigo-50",
    ring: "ring-violet-200",
    iconBg: "bg-gradient-to-br from-violet-500 to-indigo-600",
  },
  {
    key: "stage3",
    number: "STEP 3",
    title: "Shapley",
    desc: "Exact 2ⁿ or antithetic + stratified MC w/ Hoeffding",
    icon: "Σ",
    bg: "from-emerald-50 to-teal-50",
    ring: "ring-emerald-200",
    iconBg: "bg-gradient-to-br from-emerald-400 to-teal-500",
  },
  {
    key: "compose",
    number: "STEP 4",
    title: "Compose",
    desc: "mpmath intervals + final sparsify (4% floor)",
    icon: "⏚",
    bg: "from-sky-50 to-blue-50",
    ring: "ring-sky-200",
    iconBg: "bg-gradient-to-br from-sky-400 to-blue-600",
  },
  {
    key: "stageV",
    number: "STEP 5",
    title: "Verify",
    desc: "Z3 SMT proof of fairness axioms",
    icon: "✓",
    bg: "from-amber-50 to-orange-50",
    ring: "ring-amber-200",
    iconBg: "bg-gradient-to-br from-amber-400 to-orange-500",
  },
  {
    key: "currencies",
    number: "STEP 6",
    title: "Currencies",
    desc: "QWS + Rawlsian floor + roles + opportunity (α/β/γ)",
    icon: "★",
    bg: "from-rose-50 to-pink-50",
    ring: "ring-rose-200",
    iconBg: "bg-gradient-to-br from-pink-400 to-rose-500",
  },
];

const LATENCY_KEY: Record<string, string> = {
  stage0: "stage0_separate",
  stage1: "stage1_triage",
  stage2: "stage2_grouping",
  stage3: "stage3_shapley_total",
  compose: "compose",
  stageV: "stageV_proof",
  currencies: "currencies",
};

export default function PipelineStrip({
  statuses = {},
  latency = {},
  title = "How CASCADE-V works — 6-stage attribution pipeline",
}: {
  statuses?: Record<string, StageStatus>;
  latency?: Record<string, number>;
  title?: string;
}) {
  const [openStage, setOpenStage] = useState<StageKey | null>(null);

  return (
    <>
      <div className="card p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div>
              <div className="text-base font-semibold">{title}</div>
              <div className="text-xs text-subt">
                target audio → embed → triage → group → Shapley → compose →
                SMT proof → currencies (QWS + α/β/γ)
              </div>
            </div>
            <button
              onClick={() => setOpenStage("overview")}
              className="ml-1 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-white text-xs font-semibold text-subt hover:border-violet-300 hover:bg-violet-50 hover:text-violet-700"
              title="What is CASCADE-V?"
            >
              ⓘ
            </button>
          </div>
          <div className="hidden items-center gap-1.5 text-xs text-subt md:flex">
            <span className="pill pill-ok">PASS</span>
            <span className="pill pill-warn">REVIEW</span>
            <span className="pill pill-bad">VIOLATED</span>
          </div>
        </div>
        <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
          {stages.map((s, i) => {
            const st = statuses[s.key] || "idle";
            const ms = latency[LATENCY_KEY[s.key]];
            return (
              <div key={s.key} className="flex flex-1 items-stretch gap-2">
                <div
                  className={`relative flex flex-1 flex-col gap-2 rounded-2xl bg-gradient-to-br p-4 ring-1 ${s.bg} ${s.ring} ${st === "running" ? "animate-pulse" : ""}`}
                >
                  <button
                    onClick={() => setOpenStage(s.key)}
                    className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-white/70 text-xs font-semibold text-subt backdrop-blur transition-colors hover:bg-white hover:text-violet-700 hover:shadow-soft"
                    title={`Learn more about ${s.title}`}
                  >
                    ⓘ
                  </button>
                  <div className="flex items-center justify-between pr-7">
                    <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-subt">
                      {s.number}
                    </span>
                    <StatusPill status={st} />
                  </div>
                  <div className={`step-icon ${s.iconBg}`}>{s.icon}</div>
                  <div>
                    <div className="text-base font-semibold leading-tight text-ink">
                      {s.title}
                    </div>
                    <div className="text-[11px] leading-snug text-subt">
                      {s.desc}
                    </div>
                  </div>
                  {ms !== undefined && (
                    <div className="font-mono text-[11px] font-medium text-ink">
                      {ms.toFixed(1)} ms
                    </div>
                  )}
                </div>
                {i < stages.length - 1 && (
                  <div className="flex items-center justify-center self-center">
                    <div
                      className={
                        "flex h-7 w-7 items-center justify-center rounded-full text-sm font-bold text-white shadow-soft " +
                        (st === "done"
                          ? "bg-violet-500"
                          : st === "running"
                            ? "bg-violet-300"
                            : "bg-slate-300")
                      }
                    >
                      →
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
      {openStage && (
        <StageInfoModal stage={openStage} onClose={() => setOpenStage(null)} />
      )}
    </>
  );
}

function StatusPill({ status }: { status: StageStatus }): ReactNode {
  if (status === "done") return <span className="pill pill-ok">done</span>;
  if (status === "running")
    return (
      <span className="pill pill-brand">
        <span className="inline-block h-1.5 w-1.5 animate-ping rounded-full bg-violet-500" />
        running
      </span>
    );
  if (status === "fail") return <span className="pill pill-bad">fail</span>;
  return <span className="pill pill-mute">idle</span>;
}
