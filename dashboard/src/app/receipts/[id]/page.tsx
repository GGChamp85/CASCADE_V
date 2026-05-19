"use client";

import { useParams } from "next/navigation";
import { useState } from "react";

import AudioPlayer from "@/components/AudioPlayer";
import AxiomTable from "@/components/AxiomTable";
import ContributorStack from "@/components/ContributorStack";
import CreatorTestimony from "@/components/CreatorTestimony";
import CurrencyPanel from "@/components/CurrencyPanel";
import GroundTruthCompare from "@/components/GroundTruthCompare";
import KpiTile from "@/components/KpiTile";
import PerCreatorDonut from "@/components/PerCreatorDonut";
import PerSourceBars from "@/components/PerSourceBars";
import PipelineStrip, { StageStatus } from "@/components/PipelineStrip";
import SmtViewer from "@/components/SmtViewer";
import StagePipelineView from "@/components/StagePipelineView";
import VerdictBanner from "@/components/VerdictBanner";
import { useReceipt } from "@/lib/api";

export default function ReceiptDetail() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading, error } = useReceipt(id || null);
  const [openCreator, setOpenCreator] = useState<string | null>(null);

  if (isLoading) return <div className="text-sm text-subt">loading…</div>;
  if (error)
    return (
      <div className="card border-red-200 bg-red-50 p-3 text-sm text-red-700">
        ✗ {(error as Error).message}
      </div>
    );
  if (!data) return null;

  const lat = data.latency_ms || {};
  const statuses: Record<string, StageStatus> = {
    stage1: "done",
    stage2: "done",
    stage3: "done",
    compose: "done",
    stageV: data.verification.overall_status === "PROVEN" ? "done" : "fail",
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          icon="ⓘ"
          label="Receipt"
          value={<span className="font-mono text-base">{data.receipt_id}</span>}
          sub={new Date(data.created_at_utc).toLocaleString()}
          tone="white"
        />
        <KpiTile
          icon={data.verification.overall_status === "PROVEN" ? "✓" : "✗"}
          label="Verification"
          value={data.verification.overall_status}
          sub={data.verification.smt_lib_hash.slice(0, 12) + "…"}
          tone={
            data.verification.overall_status === "PROVEN" ? "mint" : "coral"
          }
        />
        <KpiTile
          icon="⏱"
          label="Total latency"
          value={
            lat.total != null ? `${lat.total.toFixed(1)}ms` : "—"
          }
          sub={`stage1 ${lat.stage1_triage?.toFixed(1)}ms · proof ${lat.stageV_proof?.toFixed(1)}ms`}
          tone="violet"
        />
        <KpiTile
          icon="◌"
          label="Pipeline"
          value={`${data.pipeline_metadata.candidates_from_triage} → ${data.pipeline_metadata.n_clusters}`}
          sub="triaged → clusters"
          tone="sky"
        />
      </div>

      <VerdictBanner receipt={data} />

      <PipelineStrip statuses={statuses} latency={lat} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <PerCreatorDonut receipt={data} />
        <div className="card flex flex-col gap-3 p-4">
          <div className="text-sm font-semibold">Target audio</div>
          <AudioPlayer
            kind={
              data.receipt_id.startsWith("upload_") ? "upload" : "test"
            }
            name={data.receipt_id}
            label={data.receipt_id}
          />
          <div className="text-xs text-subt">
            Catalog: {data.pipeline_metadata.catalog_size} stems · Method:
            {" "}
            <span className="font-mono">{data.method}</span>
          </div>
        </div>
      </div>

      <StagePipelineView receipt={data as any} />
      <ContributorStack receipt={data} />
      <GroundTruthCompare receipt={data} />
      <PerSourceBars receipt={data} />
      <AxiomTable axioms={data.verification.axioms} />
      <CurrencyPanel receipt={data as any} />
      <SmtViewer receiptId={data.receipt_id} />

      <div className="card overflow-hidden">
        <div className="border-b border-border px-4 py-2 text-sm font-semibold">
          Per-source detail
        </div>
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-subt">
              <tr>
                <th className="px-3 py-1.5 text-left font-medium">Source</th>
                <th className="px-3 py-1.5 text-left font-medium">Creator</th>
                <th className="px-3 py-1.5 text-right font-medium">Weight</th>
                <th className="px-3 py-1.5 text-right font-medium">Interval</th>
                <th className="px-3 py-1.5 text-right font-medium">Payout</th>
                <th className="px-3 py-1.5 text-left font-medium">Stage path</th>
              </tr>
            </thead>
            <tbody>
              {[...data.per_source]
                .sort((a, b) => b.weight_point - a.weight_point)
                .map((s) => (
                  <tr key={s.source_id} className="border-t border-border">
                    <td className="px-3 py-1.5 font-mono">{s.source_id}</td>
                    <td className="px-3 py-1.5">
                      <button
                        onClick={() => setOpenCreator(s.creator_id)}
                        className="text-violet-700 hover:underline"
                        title="Open creator's testimony"
                      >
                        {s.creator_name} ↗
                      </button>
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {(s.weight_point * 100).toFixed(2)}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-subt">
                      [{(s.weight_lower * 100).toFixed(1)},{" "}
                      {(s.weight_upper * 100).toFixed(1)}]
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      ${s.payout_usd.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 font-mono text-[11px] text-subt">
                      {s.stage_path}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
      {openCreator && (
        <CreatorTestimony
          receipt={data}
          creatorId={openCreator}
          onClose={() => setOpenCreator(null)}
        />
      )}
    </div>
  );
}
