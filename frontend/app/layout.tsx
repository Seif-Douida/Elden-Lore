// app/layout.tsx
import type { Metadata } from "next";
import { Marcellus, Spectral } from "next/font/google";
import { Providers } from "./providers";
import "./globals.css";

const marcellus = Marcellus({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const spectral = Spectral({
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Elden Path",
  description: "A loremaster's guide to the Lands Between.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${marcellus.variable} ${spectral.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}