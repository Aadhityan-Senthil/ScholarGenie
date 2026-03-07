import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "ScholarGenie - AI Research Assistant",
  description: "Your autonomous AI-powered research assistant for discovering, analyzing, and synthesizing scientific papers",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body style={{ fontFamily: "Inter, system-ui, sans-serif" }}>{children}</body>
    </html>
  )
}
