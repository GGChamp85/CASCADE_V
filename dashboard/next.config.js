/** @type {import('next').NextConfig} */
const API_TARGET = process.env.API_TARGET || "http://localhost:8766";

const nextConfig = {
  reactStrictMode: false,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_TARGET}/api/:path*` },
    ];
  },
};

module.exports = nextConfig;
