import type { Metadata } from "next";
import "./globals.css";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { Providers } from "@/lib/providers";

export const metadata: Metadata = {
  title: "CASCADE-V",
  description:
    "Coalition-Aware Source Crediting And Decomposed Engine, Verified.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-bg">
        <Providers>
          <div className="flex min-h-screen">
            <Sidebar />
            <div className="flex flex-1 flex-col">
              <Header />
              <main className="flex-1 overflow-auto p-6">{children}</main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
