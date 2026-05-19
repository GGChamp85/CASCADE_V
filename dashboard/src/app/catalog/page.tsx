"use client";

import { useMemo, useState } from "react";

import AudioPlayer from "@/components/AudioPlayer";
import { useCatalog } from "@/lib/api";

export default function CatalogPage() {
  const { data, isLoading } = useCatalog();
  const [creator, setCreator] = useState<string>("");
  const [category, setCategory] = useState<string>("");
  const [filter, setFilter] = useState("");

  const sources = useMemo(() => {
    return (data?.sources || [])
      .filter((s) => (creator ? s.creator_id === creator : true))
      .filter((s) => (category ? s.category === category : true))
      .filter((s) =>
        filter
          ? s.source_id.includes(filter) ||
            s.creator_name.toLowerCase().includes(filter.toLowerCase())
          : true,
      );
  }, [data, creator, category, filter]);

  const categories = useMemo(
    () =>
      Array.from(new Set((data?.sources || []).map((s) => s.category))).sort(),
    [data],
  );

  if (isLoading) return <div className="text-sm text-subt">loading…</div>;
  if (!data) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="card flex flex-wrap items-center gap-3 p-3">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter source or creator…"
          className="rounded-md border border-border bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
        />
        <select
          value={creator}
          onChange={(e) => setCreator(e.target.value)}
          className="rounded-md border border-border bg-white px-3 py-1.5 text-sm"
        >
          <option value="">All creators ({data.creators.length})</option>
          {data.creators.map((c) => (
            <option key={c.creator_id} value={c.creator_id}>
              {c.name}
            </option>
          ))}
        </select>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="rounded-md border border-border bg-white px-3 py-1.5 text-sm"
        >
          <option value="">All categories ({categories.length})</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <span className="text-xs text-subt">
          {sources.length} of {data.sources.length} stems
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
        {sources.map((s) => (
          <div key={s.source_id} className="card flex flex-col gap-2 p-3">
            <div className="flex items-center justify-between text-sm">
              <span className="font-mono">{s.source_id}</span>
              <span className="pill pill-mute">{s.category}</span>
            </div>
            <div className="text-sm font-medium">{s.creator_name}</div>
            <div className="text-xs text-subt">
              {s.bpm} BPM · {s.key}
            </div>
            <AudioPlayer kind="catalog" name={s.source_id} />
          </div>
        ))}
      </div>
    </div>
  );
}
