"use client";

import { useEffect, useRef, useState } from "react";

import { Receipt } from "@/lib/api";
import { streamAttribution, StreamEvent } from "@/lib/sse";
import PipelineStrip, { StageStatus } from "./PipelineStrip";

export type StreamSource =
  | { kind: "test"; output_id: string }
  | { kind: "upload"; file: File };

export default function LiveAttributionStream({
  source,
  runToken,
  onComplete,
}: {
  source: StreamSource | null;
  runToken: number;
  onComplete?: (r: Receipt) => void;
}) {
  const [statuses, setStatuses] = useState<Record<string, StageStatus>>({});
  const [latency, setLatency] = useState<Record<string, number>>({});
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!source || runToken === 0) return;
    setStatuses({
      stage1: "running",
      stage2: "idle",
      stage3: "idle",
      compose: "idle",
      stageV: "idle",
    });
    setLatency({});
    setEvents([]);
    setError(null);

    const handle = (e: StreamEvent) => {
      setEvents((prev) => [...prev, e]);
      const name = e.name;
      if (name === "pipeline.start") {
        setStatuses({
          stage1: "running",
          stage2: "idle",
          stage3: "idle",
          compose: "idle",
          stageV: "idle",
        });
      } else if (name === "stage1.done") {
        setStatuses((s) => ({ ...s, stage1: "done", stage2: "running" }));
        setLatency((l) => ({ ...l, stage1_triage: e.data.latency_ms }));
      } else if (name === "stage2.done") {
        setStatuses((s) => ({ ...s, stage2: "done", stage3: "running" }));
        setLatency((l) => ({ ...l, stage2_grouping: e.data.latency_ms }));
      } else if (name === "stage3.done") {
        setStatuses((s) => ({ ...s, stage3: "done", compose: "running" }));
        setLatency((l) => ({
          ...l,
          stage3_shapley_total: e.data.latency_ms,
        }));
      } else if (name === "stageV.done") {
        setStatuses((s) => ({ ...s, compose: "done", stageV: "done" }));
        setLatency((l) => ({ ...l, stageV_proof: e.data.latency_ms }));
      } else if (name === "pipeline.end") {
        setLatency((l) => ({ ...l, total: e.data.total_ms }));
      } else if (name === "receipt.ready") {
        onComplete?.(e.data as Receipt);
      } else if (name === "pipeline.error") {
        setError(e.data?.error || "unknown error");
        setStatuses((s) => {
          const x = { ...s };
          for (const k of Object.keys(x))
            if (x[k] === "running") x[k] = "fail";
          return x;
        });
      }
    };

    if (abortRef.current) abortRef.current();
    abortRef.current = streamAttribution({
      output_id: source.kind === "test" ? source.output_id : undefined,
      upload: source.kind === "upload" ? source.file : undefined,
      onEvent: handle,
      onError: (err) => setError(String(err)),
    });

    return () => {
      abortRef.current?.();
    };
    // runToken forces re-trigger even if `source` reference is unchanged
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runToken]);

  return (
    <div className="flex flex-col gap-4">
      <PipelineStrip statuses={statuses} latency={latency} />
      {error && (
        <div className="card border-red-200 bg-red-50 p-3 text-sm text-red-700">
          ✗ {error}
        </div>
      )}
      <details className="card p-3">
        <summary className="cursor-pointer text-xs text-subt">
          Event log ({events.length})
        </summary>
        <ol className="mt-2 max-h-48 overflow-auto font-mono text-[11px] text-subt">
          {events.map((e, i) => (
            <li key={i} className="border-b border-border/50 py-0.5">
              <span className="text-teal-700">{e.name}</span>{" "}
              <span className="text-slate-400">
                {typeof e.data === "object"
                  ? JSON.stringify(e.data).slice(0, 140)
                  : String(e.data)}
              </span>
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}
