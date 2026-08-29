import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000'),
  title: 'APEX-1 · Frontier AI research platform',
  description:
    'A technically credible foundation for local-first AI research, model development, and agentic applications.',
  openGraph: {
    title: 'APEX-1',
    description: 'Build intelligence from first principles.',
    images: [{ url: '/og.png', width: 1672, height: 941, alt: 'APEX-1 orbital research system' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'APEX-1',
    description: 'Build intelligence from first principles.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
