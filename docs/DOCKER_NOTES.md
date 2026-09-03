# Docker notes

Covers `backend/Dockerfile`, `backend/.dockerignore`, `frontend/Dockerfile`,
`frontend/.dockerignore` and the `docker-compose.yml` audit.

## Verification status

**Docker is blocked by corporate policy on the machine these files were written
on** (`/Applications/Docker.app` → "App Blocked at YEXT"; the CLI is SIGKILLed,
no daemon socket exists). Therefore:

| Check | Status |
|---|---|
| `docker build` backend / frontend | **NOT VERIFIED** — no daemon |
| Image sizes | **NOT VERIFIED** — nothing was built |
| Container starts / serves | **NOT VERIFIED** in-container |
| `docker run … id -u` ≠ 0 | **NOT VERIFIED** in-container |
| `.env` absent from image | **NOT VERIFIED** in-container |
| `TRADING_MODE=paper` in container | **NOT VERIFIED** in-container |

No build output or image size in this repo has been fabricated. Real
verification must come from the Docker Build GitHub Actions workflow.

What *was* verified without Docker, by running the actual commands:

- **Every pinned backend dependency has a prebuilt cp311 linux/amd64 wheel.**
  `pip download -r requirements.txt --platform manylinux2014_x86_64
  --python-version 3.11 --abi cp311 --only-binary=:all:` resolved all 80
  packages, exit 0. No source compilation is required.
- **ASGI target is valid.** `app/main.py` ends with `app = create_app()`;
  importing `app.main` yields a `FastAPI` object, so `uvicorn app.main:app`
  is correct. uvicorn's CLI also does `sys.path.insert(0, app_dir)` (default
  `.`), and `PYTHONPATH=/app` is set as belt-and-braces.
- **Health path is `/api/v1/health`.** Probed on a locally running instance:
  `200`. `/health` (what compose used to probe) returns **404**.
- **Health works without a database.** With no Postgres it answers `200` with
  `{"status":"degraded","db":"disconnected"}`, so `curl -fsS` succeeds. The
  HEALTHCHECK reports liveness and does not deadlock at startup.
- **`TRADING_MODE=paper` resolves.** `settings.trading_mode == "paper"`,
  `is_live_trading_enabled == False`; the running app logs
  `trading_mode=paper live_enabled=False`.
- **Frontend builds.** `npm install --legacy-peer-deps` then `next build`
  succeeded (19 routes, all static, exit 0), types checked.
- **Prod-only runtime works.** `npm install --omit=dev` gives 105 packages
  (vs 609 with dev), and `next start` runs from `.next` + `package.json` +
  `next.config.ts` alone. `typescript` is **not** needed at runtime.
- **Frontend HEALTHCHECK verified both directions.** Server up → `307`, exit 0;
  nothing listening → exit 1.
- **`.dockerignore` simulated** against the real tree (Go `filepath.Match`
  semantics): backend context drops ~18,000 files → **72**; frontend → **42**.

Residual risk that only a real build can close: the images are linux/amd64 and
were reasoned about, not run. The frontend test ran on darwin/arm64 with Node 24,
so npm resolved `@next/swc-darwin-arm64` locally where CI will resolve
`@next/swc-linux-x64-gnu` (both publish glibc builds — this is why the images are
Debian slim, not Alpine).

## Blocking defects found in the repo (not fixable from the files I own)

1. **`npm install` fails outright.** `next@15.0.0` declares
   `peerDependencies.react: "^18.2.0 || 19.0.0-rc-65a56d0e-20241020"`, which
   excludes the stable `react@19.2.8` that `react: "^19.0.0"` resolves to. Plain
   `npm install` dies with `ERESOLVE`. The Dockerfile uses
   `--legacy-peer-deps` as the only available workaround. **Real fix: bump
   `next` to `>=15.0.3` in `frontend/package.json`.**
2. **No `frontend/package-lock.json`.** `npm ci` is impossible, so the image uses
   `npm install`. Caret ranges are re-resolved on every build, so two builds of
   the same commit can produce different dependency trees. **This is a
   reproducibility gap and a supply-chain exposure.** Commit a lockfile and
   switch the Dockerfile to `npm ci`.
3. **`next@15.0.0` has a published security vulnerability** (npm marks it
   deprecated for CVE-2025-66478). The image ships a vulnerable Next.js.
4. **`aioredis==2.0.1` cannot import on Python 3.11+.** Its
   `exceptions.py` does `class TimeoutError(asyncio.TimeoutError,
   builtins.TimeoutError, RedisError)`, and since 3.11 those two bases are the
   same object → `TypeError: duplicate base class TimeoutError` (reproduced
   directly). `main.py` and `health.py` wrap the import in `try/except`, so the
   app still starts, but **Redis will always report `disconnected`.** Fix:
   drop `aioredis` and use `redis.asyncio` (`redis` is already installed
   transitively via `celery[redis]`).
5. **The `worker` service cannot run.** `backend/app/worker.py` does not exist
   and no Celery app is defined anywhere (`grep -rn "Celery(" backend/app` is
   empty). It is commented out in compose rather than left to crash-loop.
6. **`backend/pyproject.toml` has an invalid build backend**
   (`setuptools.backends.legacy:build`), so `pip install .` would fail. The
   image installs from the pinned `requirements.txt` instead and never needs it.
7. **`next/font/google` is used**, so `next build` requires network access to
   `fonts.googleapis.com` / `fonts.gstatic.com`. Air-gapped builds will fail.

## docker-compose.yml — security audit

Fixed in this change:

| Issue | Fix |
|---|---|
| Postgres password `algodollar_password` hardcoded in git, duplicated into two `DATABASE_URL`s | `${POSTGRES_PASSWORD:?…}` — required, no default, fails closed |
| Redis had **no authentication at all** | `--requirepass ${REDIS_PASSWORD:?…}` and credentialed `REDIS_URL` |
| `5432:5432` and `6379:6379` published on **all host interfaces** | bound to `127.0.0.1` only |
| Backend `8000:8000` public (API can place orders; `/docs` is unauthenticated) | `${BACKEND_BIND:-127.0.0.1}:8000:8000` |
| Backend healthcheck probed `/health`, which **404s** — backend could never become healthy, so frontend's `depends_on: service_healthy` **deadlocked forever** | override removed; the correct `/api/v1/health` probe lives in the Dockerfile and Compose inherits it |
| `env_file: .env` mandatory, but `.env` does not exist in a fresh clone → `compose up` failed immediately; also injected *every* variable into the container | removed; each variable the app reads is passed by name (explicit allowlist) |
| `worker` would crash-loop forever under `restart: unless-stopped` | disabled with an explanation |
| `NEXT_PUBLIC_API_URL` set as a **runtime** env var, but Next inlines `NEXT_PUBLIC_*` at **build** time — it had no effect on the browser bundle; `http://backend:8000` is also unresolvable from a browser | moved to `build.args` with a browser-reachable default |
| No `no-new-privileges`, no capability dropping | `no-new-privileges:true` on all services; `cap_drop: [ALL]` on backend and frontend |
| Unbounded container logs | `json-file` with `max-size: 10m`, `max-file: 5` |
| Obsolete `version: "3.9"` | removed |
| Floating `postgres:15-alpine` / `redis:7-alpine` | pinned to `15.8-alpine` / `7.4-alpine` |
| `TRADING_MODE` not pinned | `${TRADING_MODE:-paper}` |

Checked and found **clean**: no service runs `privileged`, and every volume is a
named volume — there are no host bind mounts, so no host filesystem is exposed.

Not done, deliberately, because it could not be tested: `read_only: true` root
filesystems and `deploy.resources.limits`. Both are recommended follow-ups.

Note `frontend.depends_on.backend` was changed from `service_healthy` to
`service_started` on purpose: the backend's health depends on Postgres, so
gating the UI on it means a database outage also takes down the dashboard that
would report the outage.

## Notes on the images

- Base images are pinned by tag: `python:3.11-slim-bookworm`,
  `node:22-bookworm-slim`. Debian slim rather than Alpine because `@next/swc`,
  `@tailwindcss/oxide` and the scientific Python wheels all ship glibc builds.
- `libgomp1` is installed in the backend runtime stage. LightGBM's manylinux
  wheel links OpenMP; without it the build still succeeds but `import lightgbm`
  fails at container start.
- Both images run as `appuser`, uid/gid **10001**. Application code and
  dependencies are root-owned and read-only to that user; only `/app/logs`
  (backend) and `/app/.next/cache` (frontend) are writable.
- The backend runs `--workers 1` deliberately: the lifespan handler builds one
  execution stack and an in-process WebSocket registry on `app.state`, so extra
  workers would create several independent order-management stacks in one
  deployment. Scale with replicas behind a load balancer.
- `/app/logs` is pre-created owned by uid 10001 so the `backend-logs` named
  volume is seeded writable for the non-root user.
- The only build args are `BUILD_SHA` and `NEXT_PUBLIC_API_URL`, both
  non-secret. Never pass a credential as `ARG`/`ENV` — both are readable via
  `docker history`.
- The test suite is neither copied into nor run from the runtime images.

### The workflow passes a browser-unreachable API URL

`.github/workflows/docker.yml` builds the frontend with
`NEXT_PUBLIC_API_URL=http://backend:8000`. Since `src/lib/api.ts` reads that
value in **client-side** code, it is inlined into the browser bundle — and a
browser cannot resolve the compose-internal hostname `backend`. The CI image is
fine as a build check but must not be deployed. The Dockerfile's own default is
`http://localhost:8000`. That workflow is owned by another engineer.
