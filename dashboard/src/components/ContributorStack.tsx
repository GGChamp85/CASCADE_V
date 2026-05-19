"use client";

import { useEffect, useRef, useState } from "react";

import { API_BASE, Receipt } from "@/lib/api";

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
];

export default function ContributorStack({ receipt }: { receipt: Receipt }) {
  const targetKind = receipt.receipt_id.startsWith("upload_")
    ? "upload"
    : "test";
  const targetUrl = `${API_BASE}/api/audio/${targetKind}/${encodeURIComponent(receipt.receipt_id)}`;

  // top contributors with non-trivial weight, sorted descending
  const sources = [...receipt.per_source]
    .filter((s) => s.weight_point > 0.01)
    .sort((a, b) => b.weight_point - a.weight_point)
    .slice(0, 8);

  const [playingId, setPlayingId] = useState<string | null>(null);
  const audios = useRef<Record<string, HTMLAudioElement>>({});

  useEffect(() => {
    return () => {
      // cleanup on unmount
      Object.values(audios.current).forEach((a) => a.pause());
    };
  }, []);

  const ensureAudio = (id: string, url: string): HTMLAudioElement => {
    if (!audios.current[id]) {
      const a = new Audio(url);
      a.preload = "metadata";
      a.addEventListener("ended", () => setPlayingId((p) => (p === id ? null : p)));
      audios.current[id] = a;
    }
    return audios.current[id];
  };

  const togglePlay = (id: string, url: string) => {
    // stop everything else
    Object.entries(audios.current).forEach(([k, a]) => {
      if (k !== id) {
        a.pause();
        a.currentTime = 0;
      }
    });
    const a = ensureAudio(id, url);
    if (playingId === id) {
      a.pause();
      a.currentTime = 0;
      setPlayingId(null);
    } else {
      a.currentTime = 0;
      a.play().catch(() => {});
      setPlayingId(id);
    }
  };

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="text-base font-semibold">A/B listen — why this split?</div>
          <div className="text-xs text-subt">
            Play the output, then play each contributing source. The bars show
            the attributed share. A musician can audibly verify the split makes
            sense.
          </div>
        </div>
        <span className="pill pill-brand">{sources.length} contributors</span>
      </div>

      {/* Output / target row */}
      <div className="mb-4 flex items-center gap-3 rounded-xl border-2 border-violet-200 bg-gradient-to-r from-violet-50 to-indigo-50 p-3">
        <button
          onClick={() => togglePlay("__target__", targetUrl)}
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 text-xl text-white shadow-soft transition-transform hover:scale-105"
          aria-label="Play target"
        >
          {playingId === "__target__" ? "❚❚" : "▶"}
        </button>
        <div className="flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <div className="text-sm font-semibold">Output (target audio)</div>
            <span className="font-mono text-xs text-subt">
              {receipt.receipt_id}
            </span>
          </div>
          <div className="text-xs text-subt">
            This is the AI-generated mix being attributed. {sources.length}{" "}
            contributing sources discovered.
          </div>
        </div>
        <span className="pill pill-brand">100%</span>
      </div>

      <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-subt">
        <span className="h-px flex-1 bg-border" />
        contributing sources (Z3-verified Shapley split)
        <span className="h-px flex-1 bg-border" />
      </div>

      <ul className="flex flex-col gap-2">
        {sources.map((s, i) => {
          const url = `${API_BASE}/api/audio/catalog/${encodeURIComponent(s.source_id)}`;
          const playing = playingId === s.source_id;
          const pct = s.weight_point * 100;
          const lo = s.weight_lower * 100;
          const hi = s.weight_upper * 100;
          const color = COLORS[i % COLORS.length];
          return (
            <li
              key={s.source_id}
              className={
                "flex items-center gap-3 rounded-xl border p-3 transition-colors " +
                (playing
                  ? "border-violet-400 bg-violet-50/50"
                  : "border-border bg-white hover:bg-slate-50")
              }
            >
              <button
                onClick={() => togglePlay(s.source_id, url)}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-base text-white shadow-soft transition-transform hover:scale-105"
                style={{ background: color }}
                aria-label={`Play ${s.source_id}`}
              >
                {playing ? "❚❚" : "▶"}
              </button>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between gap-2">
                  <div className="truncate text-sm font-semibold">
                    {s.creator_name}
                  </div>
                  <div className="font-mono text-xs text-subt">
                    {s.source_id}
                  </div>
                </div>
                <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${Math.max(pct, 1.5)}%`,
                      background: color,
                    }}
                  />
                </div>
                <div className="mt-1 flex items-center justify-between text-[11px] text-subt">
                  <span>
                    Range:{" "}
                    <span className="font-mono">
                      [{lo.toFixed(1)}, {hi.toFixed(1)}]%
                    </span>
                  </span>
                  <span className="font-mono">${(pct / 100).toFixed(4)} / $1.00 payout</span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-0.5">
                <span
                  className="font-mono text-base font-bold"
                  style={{ color }}
                >
                  {pct.toFixed(1)}%
                </span>
                <span className="text-[10px] text-subt">
                  rank #{i + 1}
                </span>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="mt-4 rounded-md border border-border bg-slate-50 p-3 text-[11px] leading-relaxed text-subt">
        <span className="font-semibold text-ink">How to read this:</span> Each
        bar's length is the source's Shapley-fair share of the output. The
        intervals come from Hoeffding-bounded Monte Carlo Shapley (or are exact
        for small clusters). The same numbers were re-checked by Z3 against the
        three fairness axioms.
      </div>
    </div>
  );
}
