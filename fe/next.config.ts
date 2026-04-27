import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://141.223.140.32:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/proxy/:path*",
        destination: `${BACKEND_URL}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
