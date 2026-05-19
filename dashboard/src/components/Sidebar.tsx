"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type Item = { href: string; label: string; icon: string; tone?: string };
const groups: Array<{ title: string; items: Item[] }> = [
  {
    title: "Main",
    items: [
      { href: "/", label: "Dashboard", icon: "▦", tone: "text-violet-500" },
      { href: "/attribute", label: "Attribute", icon: "⏵", tone: "text-rose-500" },
    ],
  },
  {
    title: "Data",
    items: [
      { href: "/catalog", label: "Catalog", icon: "≡", tone: "text-sky-500" },
      { href: "/receipts", label: "Receipts", icon: "✓", tone: "text-emerald-500" },
    ],
  },
  {
    title: "Insights",
    items: [
      { href: "/eval", label: "Evaluation", icon: "▮", tone: "text-amber-500" },
      { href: "/training", label: "Training", icon: "△", tone: "text-blue-500" },
    ],
  },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="hidden w-60 shrink-0 border-r border-border bg-white md:flex md:flex-col">
      <div className="flex items-center gap-3 px-5 py-5">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 text-lg font-bold text-white shadow-soft">
          CV
        </div>
        <div>
          <div className="text-base font-bold leading-tight tracking-tight">
            CASCADE-V
          </div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-subt">
            Coalition · Verified
          </div>
        </div>
      </div>
      <div className="px-5 pb-3 text-[11px] leading-snug text-subt">
        <span className="font-semibold text-violet-700">C</span>oalition-aware{" "}
        <span className="font-semibold text-rose-600">S</span>ource{" "}
        <span className="font-semibold text-emerald-600">C</span>rediting{" "}
        And{" "}
        <span className="font-semibold text-sky-600">D</span>ecomposed{" "}
        <span className="font-semibold text-amber-600">E</span>ngine,{" "}
        Verified.
      </div>
      <nav className="flex flex-1 flex-col gap-4 px-2 pb-6">
        {groups.map((g) => (
          <div key={g.title}>
            <div className="px-3 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-subt">
              {g.title}
            </div>
            <ul className="flex flex-col gap-0.5">
              {g.items.map((it) => {
                const active =
                  path === it.href ||
                  (it.href !== "/" && path?.startsWith(it.href));
                return (
                  <li key={it.href}>
                    <Link
                      href={it.href}
                      className={
                        "flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium transition-colors " +
                        (active
                          ? "bg-brand-soft text-brand"
                          : "text-ink hover:bg-slate-50")
                      }
                    >
                      <span className={"w-4 text-center text-base " + (it.tone || "text-subt")}>
                        {it.icon}
                      </span>
                      {it.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
      <div className="m-3 rounded-2xl bg-gradient-to-br from-violet-100 to-indigo-100 p-3 text-[11px] leading-snug text-violet-900">
        <div className="font-semibold">Z3 audit-grade</div>
        <div className="text-violet-700/80">
          Every receipt ships with an SMT-LIB proof verifiable independently.
        </div>
      </div>
      <div className="border-t border-border px-5 py-3 text-[11px] text-subt">
        v0.2.0 <span className="ml-2 font-mono">M4 · MPS</span>
      </div>
    </aside>
  );
}
