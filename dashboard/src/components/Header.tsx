"use client";

import { usePathname } from "next/navigation";
import { useHealth } from "@/lib/api";

const titles: Record<string, string> = {
  "/": "Dashboard",
  "/receipts": "Receipts",
  "/attribute": "Attribute",
  "/catalog": "Catalog",
  "/eval": "Evaluation",
  "/training": "Training",
};

export default function Header() {
  const path = usePathname() || "/";
  const root = "/" + (path.split("/")[1] || "");
  const title = titles[root] || "Dashboard";
  const sub = path.split("/").slice(2).join(" / ");
  const { data: h } = useHealth();

  return (
    <header className="flex items-center justify-between border-b border-border bg-white px-6 py-3">
      <div>
        <div className="text-[11px] uppercase tracking-wider text-subt">
          ☖ &gt; {title}
          {sub && <span className="ml-1">/ {sub}</span>}
        </div>
        <h1 className="text-lg font-semibold leading-tight">{title}</h1>
      </div>
      <div className="flex items-center gap-3 text-sm">
        {h ? (
          <>
            <span className="pill pill-ok">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-500" />
              api ok
            </span>
            <span className="pill pill-mute font-mono">{h.device}</span>
            <span className="pill pill-mute">
              {h.catalog_size} stems · {h.n_creators} creators
            </span>
            <span className="pill pill-accent">{h.n_receipts} receipts</span>
          </>
        ) : (
          <span className="pill pill-warn">connecting…</span>
        )}
      </div>
    </header>
  );
}
