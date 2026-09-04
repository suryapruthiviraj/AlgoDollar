import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // API rewrites to backend
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/v1/:path*',
        destination: `${apiUrl}/api/v1/:path*`,
      },
    ];
  },

  // TypeScript strict mode enforced via tsconfig
  typescript: {
    ignoreBuildErrors: false,
  },

  // The `eslint` key is gone in Next 16, along with the `next lint` command it
  // configured — `next build` no longer runs ESLint at all. Linting is now a
  // separate `eslint .` step (`npm run lint`), which CI runs in its own job, so
  // nothing is skipped; it just no longer rides along with the build.

  // WebSocket proxy note:
  // Next.js does not natively proxy WebSocket connections via rewrites.
  // WebSocket connections should connect directly to NEXT_PUBLIC_WS_URL (e.g. ws://localhost:8000/ws).
  // In production, configure your reverse proxy (nginx/traefik) to forward /ws/* to the backend.

  // Experimental features
  experimental: {
    // Enable turbopack for dev
  },
};

export default nextConfig;
