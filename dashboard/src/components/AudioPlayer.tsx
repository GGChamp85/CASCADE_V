"use client";

import { API_BASE } from "@/lib/api";

export default function AudioPlayer({
  kind,
  name,
  label,
}: {
  kind: "catalog" | "test" | "upload";
  name: string;
  label?: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-border bg-white p-2">
      <span className="font-mono text-[11px] text-subt">{label || name}</span>
      <audio
        controls
        preload="none"
        src={`${API_BASE}/api/audio/${kind}/${encodeURIComponent(name)}`}
        className="h-7 flex-1"
      />
    </div>
  );
}
