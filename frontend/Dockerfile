# syntax=docker/dockerfile:1.7
#
# AlgoDollar frontend (Next.js 15) — production image.
#
# Build context is ./frontend (see .github/workflows/docker.yml and
# docker-compose.yml). Target platform is linux/amd64.
#
#   docker build -t algodollar/frontend:local ./frontend
#
# Design notes:
#   * Four stages. The dev toolchain (typescript, eslint, jest, tailwind) is
#     used only to produce .next/ and never reaches the runtime image, which
#     gets a separately resolved production-only dependency tree.
#   * Runs as an unprivileged user (uid 10001).
#   * Debian slim, not Alpine: @next/swc and @tailwindcss/oxide ship prebuilt
#     glibc (linux-x64-gnu) native binaries. musl would force a source build.
#   * `output: "standalone"` is NOT set in next.config.ts, so the runtime stage
#     ships .next + a pruned node_modules and runs `next start`.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — deps: full dependency tree (dev + prod), needed to compile
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS deps

WORKDIR /app

# `npm ci`, not `npm install` — it installs exactly the tree recorded in the
# lockfile and fails if package.json and the lockfile disagree. That is what
# makes a build of a given commit reproducible.
#
# This was previously `npm install --legacy-peer-deps` for two reasons, both
# now resolved: the repository had no lockfile at all, and next@15.0.0 declared
# a peer range (`react: "^18.2.0 || 19.0.0-rc-..."`) that excluded the stable
# react@19 which `^19.0.0` resolves to, so a plain install died with ERESOLVE.
# A lockfile is now committed and next is 16.3.4, so the peer conflict and the
# `--legacy-peer-deps` escape hatch are both unnecessary.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — builder: produce the optimized production build
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS builder

WORKDIR /app

# NEXT_PUBLIC_* values are inlined into the client bundle at BUILD time, so this
# has to be a build arg. It is a service URL, not a credential — never pass a
# secret this way, as build args are visible in `docker history`.
#
# The default is a BROWSER-reachable URL on purpose. src/lib/api.ts reads
# process.env.NEXT_PUBLIC_API_URL in client-side code, so this value ends up in
# the bundle the user's browser runs — and a browser cannot resolve the
# compose-internal hostname `backend`. Only server-side code (the rewrites() in
# next.config.ts) can use `http://backend:8000`.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# `next start` never reads .next/cache, so dropping it saves significant image
# weight. `mkdir -p public` guarantees the directory exists (this repo has no
# frontend/public), so the runtime COPY below is unconditional and cannot fail.
RUN npm run build \
 && rm -rf .next/cache \
 && mkdir -p public

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — prod-deps: production-only dependency tree for the runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS prod-deps

WORKDIR /app

# ~105 packages here versus ~609 with devDependencies. --ignore-scripts keeps
# third-party postinstall hooks out of the tree that actually ships.
#
# `npm ci` here too, for the same reproducibility reason as the deps stage: the
# runtime tree is pinned by the lockfile rather than re-resolved.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund --ignore-scripts \
 && npm cache clean --force

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — runner
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS runner

ARG BUILD_SHA=unknown

LABEL org.opencontainers.image.title="AlgoDollar Frontend" \
      org.opencontainers.image.description="AlgoDollar quantitative trading platform UI" \
      org.opencontainers.image.source="https://github.com/algodollar/algodollar" \
      org.opencontainers.image.revision="${BUILD_SHA}"

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Unprivileged service account. The node images ship a `node` user at uid 1000;
# a dedicated high uid is used instead so it cannot collide with a host user.
RUN groupadd --system --gid 10001 appuser \
 && useradd --system --uid 10001 --gid appuser \
            --create-home --home-dir /home/appuser \
            --shell /usr/sbin/nologin appuser

WORKDIR /app

# Application files are root-owned and read-only to the service account: the
# runtime has no reason to rewrite its own code or dependencies.
COPY --from=prod-deps --chown=root:root /app/node_modules ./node_modules
COPY --from=builder   --chown=root:root /app/.next        ./.next
COPY --from=builder   --chown=root:root /app/public       ./public
COPY --from=builder   --chown=root:root /app/package.json /app/next.config.ts ./

# Every route in this app builds as static and a local run of `next start`
# wrote nothing under .next, so the build output can stay read-only. The one
# writable exception is the cache directory, which the App Router uses for
# server-side fetch caching if a route ever becomes dynamic.
RUN mkdir -p /app/.next/cache && chown -R appuser:appuser /app/.next/cache

USER appuser

EXPOSE 3000

# `/` is a redirect (app/page.tsx calls redirect('/dashboard')) and answers 307,
# so the probe must not treat a redirect as failure — it only needs to prove the
# server is up. redirect:"manual" avoids following it; anything below 500 counts
# as alive. Uses node's built-in fetch so the image needs no curl/wget package.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD node --no-warnings -e "fetch('http://127.0.0.1:'+(process.env.PORT||3000)+'/',{redirect:'manual'}).then(r=>process.exit(r.status<500?0:1)).catch(()=>process.exit(1))"

# The next binary directly, not `npm run start`: npm would sit between init and
# the server as PID 1 and swallow SIGTERM, breaking graceful shutdown.
# Absolute path because exec form does not run through a shell.
CMD ["/app/node_modules/.bin/next", "start", "-H", "0.0.0.0", "-p", "3000"]
