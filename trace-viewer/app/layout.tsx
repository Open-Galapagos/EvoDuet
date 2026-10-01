import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "SkyDiscover Trace Viewer",
  description:
    "Analyst-friendly inspection of scientific discovery, web search, and LLM traces.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
