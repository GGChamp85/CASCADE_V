import { ReactNode } from "react";

type Tone = "white" | "mint" | "coral" | "violet" | "sky" | "amber";

const TONES: Record<
  Tone,
  { bg: string; fg: string; sub: string; icon: string }
> = {
  white: {
    bg: "bg-white",
    fg: "text-ink",
    sub: "text-subt",
    icon: "bg-slate-100 text-slate-500",
  },
  mint: {
    bg: "bg-gradient-to-br from-emerald-400 to-teal-500",
    fg: "text-white",
    sub: "text-emerald-50/90",
    icon: "bg-white/25 text-white",
  },
  coral: {
    bg: "bg-gradient-to-br from-rose-400 to-orange-500",
    fg: "text-white",
    sub: "text-rose-50/90",
    icon: "bg-white/25 text-white",
  },
  violet: {
    bg: "bg-gradient-to-br from-violet-500 to-indigo-600",
    fg: "text-white",
    sub: "text-violet-100/90",
    icon: "bg-white/25 text-white",
  },
  sky: {
    bg: "bg-gradient-to-br from-sky-400 to-blue-600",
    fg: "text-white",
    sub: "text-sky-50/90",
    icon: "bg-white/25 text-white",
  },
  amber: {
    bg: "bg-gradient-to-br from-amber-400 to-orange-500",
    fg: "text-white",
    sub: "text-amber-50/90",
    icon: "bg-white/25 text-white",
  },
};

export default function KpiTile({
  icon,
  label,
  value,
  sub,
  tone = "white",
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
}) {
  const t = TONES[tone];
  return (
    <div className={`${t.bg} kpi-card`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span
            className={`text-xs font-medium uppercase tracking-wider ${t.sub}`}
          >
            {label}
          </span>
          <div className={`text-4xl font-bold leading-none ${t.fg}`}>
            {value}
          </div>
        </div>
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-base shadow-soft ${t.icon}`}>
          {icon}
        </div>
      </div>
      {sub && (
        <div className={`mt-3 text-xs font-medium ${t.sub}`}>{sub}</div>
      )}
    </div>
  );
}
