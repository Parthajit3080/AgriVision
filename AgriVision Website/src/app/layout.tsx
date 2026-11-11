import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgriVision | Field intelligence for better farm decisions",
  description: "Plan your season with crop yield estimates, soil-informed fertilizer guidance, and climate context for farms across India.",
  applicationName: "AgriVision",
  referrer: "no-referrer",
  keywords: ["farm planning", "crop yield", "fertilizer planning", "soil health", "climate risk", "India agriculture"],
  openGraph: {
    title: "AgriVision | Field intelligence for better farm decisions",
    description: "Explore crop yield estimates, nutrient plans, and climate context for the season ahead.",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "AgriVision | Field intelligence for better farm decisions",
    description: "Explore crop yield estimates, nutrient plans, and climate context for the season ahead.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
