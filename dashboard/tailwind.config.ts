import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  safelist: [
    {
      pattern:
        /(bg|text|ring|from|to|via|border)-(emerald|rose|amber|sky|violet|indigo|blue|teal|orange|slate)-(50|100|200|300|400|500|600|700|800|900)/,
    },
    { pattern: /bg-gradient-to-(br|r|tr|t|tl|l|bl|b)/ },
  ],
  theme: {
    extend: {
      colors: {
        bg: "#f1f3f7",
        card: "#ffffff",
        border: "#e2e8f0",
        ink: "#0f172a",
        subt: "#64748b",
        brand: "#7c3aed",
        "brand-soft": "#ede9fe",
        accent: "#0d9488",
        "accent-soft": "#ccfbf1",
      },
      fontFamily: {
        sans: [
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        soft: "0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 16px rgba(15, 23, 42, 0.06)",
      },
    },
  },
  plugins: [],
};

export default config;
