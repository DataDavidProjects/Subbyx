import type { Metadata } from "next";
import Link from "next/link";
import { Poppins } from "next/font/google";
import { Logo } from "@/components/ui/logo";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-poppins",
});

export const metadata: Metadata = {
  title: "Subbyx - Fraud Detection",
  description: "Checkout Fraud Prediction System",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${poppins.variable} min-h-screen py-12 px-4`}>
        <nav className="max-w-6xl mx-auto mb-8">
          <div className="flex items-center justify-center">
            <Link href="/">
              <Logo />
            </Link>
          </div>
        </nav>
        {children}
        <Toaster richColors position="top-right" />
      </body>
    </html>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="text-gray-600 hover:text-gray-900 font-medium transition"
    >
      {children}
    </Link>
  );
}
