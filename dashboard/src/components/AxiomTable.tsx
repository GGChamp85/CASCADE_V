import { AxiomCheck } from "@/lib/api";

const LABELS: Record<string, string> = {
  efficiency: "Efficiency",
  symmetry: "Symmetry",
  dummy: "Null player",
};

export default function AxiomTable({ axioms }: { axioms: AxiomCheck[] }) {
  return (
    <div className="card overflow-hidden">
      <div className="border-b border-border px-4 py-2">
        <div className="text-sm font-semibold">Fairness axioms</div>
        <div className="text-xs text-subt">
          Z3 SMT certificate — Shapley's three core fairness conditions, checked
          per-receipt.
        </div>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-subt">
          <tr>
            <th className="px-4 py-2 text-left font-medium">Axiom</th>
            <th className="px-4 py-2 text-left font-medium">Status</th>
            <th className="px-4 py-2 text-left font-medium">Detail</th>
          </tr>
        </thead>
        <tbody>
          {axioms.map((ax) => (
            <tr key={ax.name} className="border-t border-border">
              <td className="px-4 py-2 font-medium">
                {LABELS[ax.name] || ax.name}
              </td>
              <td className="px-4 py-2">
                {ax.status === "PROVEN" && (
                  <span className="pill pill-ok">PROVEN</span>
                )}
                {ax.status === "VIOLATED" && (
                  <span className="pill pill-bad">VIOLATED</span>
                )}
                {ax.status === "NA" && (
                  <span className="pill pill-mute">N/A</span>
                )}
              </td>
              <td className="px-4 py-2 text-subt">{ax.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
