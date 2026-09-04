# AlgoDollar — CI/CD and Security Audit

**Scope:** CI/CD pipeline, security posture, secret management, dependency hygiene, container and deployment safety. No strategy, model or backtest work.

**Starting state:** all four running workflows were **RED on every commit**, and had been since the repository was created. No backend test had ever executed in CI.

---

## 1. Initial findings

`gh run list` on commit `ff0ab7d`:

| Workflow | Status | Actual error |
|---|---|---|
| Backend CI | FAIL | `BackendUnavailable: Cannot import 'setuptools.backends.legacy'` |
| Docker Build | FAIL | `failed to read dockerfile: open Dockerfile: no such file or directory` |
| Security Scanning | FAIL | TruffleHog `BASE and HEAD commits are the same`; setup-node cache path unresolved; Bandit exit 1 |
| Safety Gates | FAIL | `ModuleNotFoundError: No module named 'aiosqlite'` (7 errors, 667 passed) |
| Frontend CI | never ran | path filter — no frontend change had ever been pushed |

The pattern is consistent and worth naming: **the pipeline was decorative.** Backend CI died at dependency install, so the 1106-test suite it was supposed to guard had never run there once.

---

## 2. Defects found, root causes, and fixes

### 2.1 Integrity — fabricated financial data served to the UI

**The most serious finding of this phase**, and not one any scanner reports.

| | |
|---|---|
| **Defect** | `/api/v1/research/backtest` invented its results. A seeded RNG produced a Sharpe between 0.8 and 2.2, an annualized return between 8–22%, a win rate, a profit factor and a monthly equity curve, returned in a `BacktestResult` indistinguishable from a real one. `/api/v1/markets/sectors` likewise generated sector changes, advance/decline counts and invented tickers (`REASTOCK`). |
| **Root cause** | Placeholder scaffolding that shipped as live endpoints. The docstring said "plausible placeholder metrics so the API layer is fully functional" — but nothing in the response marked it synthetic. |
| **Reach** | `frontend/src/app/research/page.tsx` calls the backtest endpoint from its "Run Backtest" button; the dashboard renders sector data as a heatmap. A user was shown invented performance as measured fact. |
| **Fix** | Both now return **501** with a pointer to the real engine (`app.backtesting.engine.EventDrivenBacktester`, `app.research.pipeline`). |
| **Security implication** | Every report in `docs/` states no strategy is validated and results must never be fabricated. An endpoint quietly contradicting that undermines the entire audit trail. Returning nothing is strictly better than returning fiction. |

### 2.2 Pickle deserialization → RCE (Bandit B301, MEDIUM)

| | |
|---|---|
| **Defect** | `HistoricalDataProvider` round-tripped DataFrames through Redis with `pickle.dumps`/`pickle.loads`. |
| **Root cause** | Convenience serialization for a DataFrame cache. |
| **Security implication** | `pickle.loads` executes arbitrary code in its payload, so **write access to the cache is code execution inside the trading process** — and the *same Redis instance holds the kill switch*. One attacker-written key could disable the halt control and execute code, from a cache of daily price bars that needs no such capability. |
| **Fix** | Replaced with Parquet, a data-only format. A hostile payload can now at worst be malformed, and malformed is caught and treated as a cache miss. |

### 2.3 Weak MD5 (Bandit B324, HIGH)

Used to derive the RNG seed for the fabricated metrics in 2.1. **Eliminated at the source** by removing the fabrication — no suppression needed.

### 2.4 `aioredis` is dead code on Python 3.11+

| | |
|---|---|
| **Defect** | `aioredis==2.0.1` cannot be imported on Python 3.11+ (`duplicate base class TimeoutError`). Both call sites wrapped the import in `try/except`. |
| **Consequence** | Redis silently reported `disconnected` on **every** startup and **every** health check, against a perfectly healthy Redis. The Redis-backed kill-switch store would never have worked in production. |
| **Fix** | Migrated to `redis.asyncio` (aioredis was merged into redis-py and is unmaintained). |

### 2.5 Undefined name — `BrokerConnectionError` (ruff F821)

`app/broker/zerodha.py` raised `BrokerConnectionError` in three places without importing it. Introduced during the previous phase's hardening. If Kite ever returned an unexpected `positions()` shape, the raise would have died with `NameError` instead of surfacing the broker-contract violation. **Verified fixed** by driving a malformed response through it.

### 2.6 Invalid PEP 517 build backend

`build-backend = "setuptools.backends.legacy:build"` does not exist. pip raised `BackendUnavailable`, Backend CI died at install, and **no backend test had ever run in CI**. Corrected to `setuptools.build_meta`.

### 2.7 Missing dependencies

`aiosqlite` and `pytz` were imported but never declared. Both are the same failure mode: green on a developer machine that happened to have them, red on a clean runner. Added to `[project.optional-dependencies].test` and the runtime deps respectively.

### 2.8 No Dockerfiles existed

`docker.yml` built `./backend/Dockerfile` and `./frontend/Dockerfile`; `find . -name "Dockerfile*"` returned nothing. Created both — multi-stage, pinned bases, non-root uid 10001, `libgomp1` for the LightGBM wheel, healthcheck on the real `/api/v1/health` path.

### 2.9 docker-compose exposures

| Defect | Fix |
|---|---|
| Postgres password `algodollar_password` hardcoded in git | `${POSTGRES_PASSWORD:?}` — required, fails closed |
| **Redis had no authentication at all** — the store holding the kill switch | `--requirepass ${REDIS_PASSWORD:?}` |
| Postgres, Redis and backend published on all interfaces | bound to `127.0.0.1` |
| Healthcheck probed `/health`, which 404s (real path `/api/v1/health`) | corrected — the service could never become healthy, and the frontend's `depends_on: service_healthy` would have deadlocked forever |
| `env_file: .env` mandatory but no `.env` exists | explicit variable allowlist |
| `NEXT_PUBLIC_API_URL` set at runtime, but Next inlines at build time | moved to `build.args` |
| — | added `no-new-privileges`, `cap_drop: [ALL]`, bounded logging, pinned tags |

Confirmed clean: nothing privileged, and every volume is named — **no host bind mounts**, so no host data exposure.

### 2.10 Frontend dependency vulnerabilities

| | |
|---|---|
| **Defect** | No `package-lock.json` existed, so `npm ci` was impossible and builds were not reproducible. Worse, plain `npm install` **failed**: `next@15.0.0`'s peer range excluded the stable `react@19` that `^19.0.0` resolves to (ERESOLVE). |
| **Fix** | Lockfile committed. `next` 15.0.0 → 15.5.25. |
| **Result** | `npm audit`: **1 critical + 2 high → 1 moderate + 1 high**. The critical bundle included **Authorization Bypass in Next.js Middleware** and **SSRF** — both material for a trading dashboard. |

### 2.11 Lint had never run

111 ruff errors, and the config used the deprecated top-level `select`/`ignore` (warning on every run). Moved under `[tool.ruff.lint]`; 56 issues auto-fixed; the full suite was re-run after each pass with no behaviour change.

### 2.12 Password hashing was completely broken (HIGH)

The most serious defect found in this phase, and it was found only because `app/core/security.py` — the module guarding every authenticated route — had **zero tests**. The first test written against it failed immediately:

```
hash_password("correct horse battery staple")
ValueError: password cannot be longer than 72 bytes
```

on a **28-character** password.

| | |
|---|---|
| **Root cause** | `passlib` 1.7.4 (last released 2020, effectively unmaintained) reads `bcrypt.__about__.__version__`, an attribute `bcrypt` removed in 4.1. With `bcrypt` 5.x the failed version probe puts passlib on a code path where *every* `hash()` call raises, regardless of the actual input length. |
| **Impact** | Any user registration or password change would have raised `ValueError` → HTTP 500. Raw `bcrypt` was verified working; passlib alone was at fault. |
| **Why it was invisible** | Nothing tested it, and `hash_password` is not yet called by any application code, so no route exercised it either. |
| **Fix** | `passlib` removed; `bcrypt` called directly. |

bcrypt's **72-byte input limit is real**, not a passlib artifact, and the two obvious ways to handle it are both wrong:

- **truncating to 72 bytes** makes every password sharing a 72-byte prefix equivalent — a silent authentication weakness;
- **rejecting long input** turns a strong passphrase into an error.

Input is therefore SHA-256'd and base64'd first, mapping any length to a fixed 44 bytes — what Django and passlib's own `bcrypt_sha256` do. base64 rather than the raw digest because a raw digest can contain `NUL`, which truncates the C string bcrypt hashes. `test_long_passphrases_are_not_truncated_to_72_bytes` asserts the property directly.

**No stored hashes were affected**: there are no migrations, no seed data, and no caller — verified before changing the format.

### 2.13 Two dependency advisories that no version bump could ever fix

`python-jose` carries **PYSEC-2025-185** and requires `ecdsa` (`Required-by: python-jose`, nothing else), which carries **PYSEC-2026-1325**. Neither has a published fix, so no pin could clear them — pinning would have meant carrying two permanently-unpatched advisories in the authentication path.

`app/core/security.py` used exactly three symbols from it (`jwt.encode`, `jwt.decode`, `JWTError`), all of which PyJWT provides. **The dependency was removed rather than suppressed.** Removing it dropped `ecdsa` with it: 4 advisories cleared by deleting a dependency.

`decode()` still passes `algorithms` explicitly and never reads the token header's own `alg`; `test_alg_none_is_rejected` pins the classic forgery.

### 2.14 `requirements.txt` was auditing a dependency set nothing installed

`pyproject.toml` declares `>=` ranges and is what both CI (`pip install -e ".[test]"`) and local development install. `requirements.txt` is the pinned file `pip-audit` reads. The two had **drifted**: the audited file still named `starlette==0.41.3`, `lightgbm==4.5.0` and `python-multipart==0.0.12` while every real environment had long since resolved to current versions.

So the scanner was reporting on a set that nothing actually ran. Both files are now consistent, and the **pyproject floors were raised too** — otherwise a fresh resolve on a clean machine could fall back onto the vulnerable versions the pins had just removed.

| Package | Was | Now | Advisories cleared |
|---|---|---|---|
| `cryptography` | 43.0.3 | 50.0.1 | GHSA-537c-gmf6-5ccf, PYSEC-2026-3552/3553/3554 |
| `starlette` | 0.41.3 | 1.6.0 | PYSEC-2026-161, 1941, 1942, 2280, 2281, 248, 249 |
| `python-multipart` | 0.0.12 | 0.0.32 | PYSEC-2026-1851, 1852, 3036–3040 |
| `lightgbm` | 4.5.0 | 4.7.0 | PYSEC-2024-231 |
| `fastapi` | 0.115.5 | 0.141.1 | — (required for starlette 1.x) |
| `python-jose` + `ecdsa` | 3.3.0 / 0.19.2 | **removed** | PYSEC-2024-232/233, PYSEC-2025-185, PYSEC-2026-1325 |

### 2.15 CI linted the application but not the tests

`ruff check app` never looked at `tests/`, where **158 findings** had accumulated. Two of them were substantive: `test_07_a_broker_rejection_...` and `test_16e_zero_traded_volume_...` each computed an `ExecutionResult` and asserted nothing about it. Both now assert on the outcome.

CI runs `ruff check app tests`. Test code gates releases and is held to the same bar.

---

## 3. Suppressions — complete and honest list

**No security finding was suppressed.** No scanner was disabled, no `continue-on-error`, no `|| true`, no `xfail`, no skipped test.

Four **style** rules are ignored, each with in-file justification. None is a security or correctness rule:

| Rule | Scope | Why |
|---|---|---|
| `N803` / `N806` | all | Numerical code follows mathematical convention: `X` design matrix, `y` target, `Sigma` covariance. Renaming to `x_train`/`sigma` makes the code *harder* to check against the formulas it implements. |
| `N818` | all | `LiveTradingBlocked`, `IllegalStateTransition` read as conditions at the call site. An `-Error` suffix adds nothing and renaming is a breaking change across the execution and governance layers. |
| `E712` | **two lines** | Per-line `# noqa`. SQLAlchemy requires `== True` in a SQL expression; `is True` has no SQL form. Not a blanket ignore. |
| `N802` | `tests/**` only | Regression tests are deliberately named `test_BUG_<the-defect-it-pins>`; the uppercase marker is the point, since it shows at a glance which tests exist because something was once broken. Every other rule applies to test code exactly as to application code. |

Exactly **one dependency advisory** is ignored, by ID, and it is not a severity floor: `PYSEC-2020-25` (autobahn 19.11.2), because `kiteconnect` — Zerodha's official SDK and the only supported way to reach the broker — hard-pins the vulnerable version in every published release (verified against 4.2.0, 5.0.1, 5.1.0, 5.2.0, 5.2.1). See §7. Every other advisory, at every severity, still fails the build.

The counter-example that shows this is not a habit: `python-jose` and `ecdsa` each carry an advisory with **no published fix** — the exact situation where an `--ignore-vuln` would have been easiest. They were removed instead (§2.13).

---

## 4. Secret management

| Check | Result |
|---|---|
| `.env` gitignored | Yes — and `!.env.example` negation so the template is tracked |
| `.env.example` contains only placeholders | Yes |
| Any `.pem`/`.key`/`.p8`/`.p12`/`.pfx` ever committed | **No** — verified across all 27 commits |
| Secrets loaded from environment | Yes, via `pydantic-settings` |
| Credentials in logs or error messages | None found |
| CI secrets exposed to fork PRs | No — the safety gates require no credentials at all |

### One credential was committed

`POSTGRES_PASSWORD: algodollar_password` appeared in `docker-compose.yml` from the initial commit (`0b9f741`) until it was removed in `1ff376a`.

**Assessment — stated precisely:**

- It was a **local development default** for a containerized Postgres, never a production credential, and was never used against a real database.
- **Rotation is not required**, because it never protected anything real. Anyone who ran the old `docker-compose.yml` locally should change their local password.
- It **remains in git history**. Removing it would require a history rewrite and force-push, which is not justified for a dev-only default and would disrupt every clone. Deleting the current file does **not** remove the historical exposure, and this document exists so that is not assumed.

---

## 5. Dependency scanning design

| Ecosystem | Source of truth | Scanner |
|---|---|---|
| Python | `backend/pyproject.toml`, `backend/requirements.txt` | `pip-audit` |
| Node | `frontend/package.json`, `frontend/package-lock.json` | `npm audit` |
| Actions | `.github/workflows/*` | Dependabot |

`.github/dependabot.yml` covers pip, npm and github-actions.

---

## 6. Trading-safety regression protection

`safety-gates.yml` runs on every push and PR and **fails the build** if any of these regress:

- a required safety suite file is **missing** (a deleted safety test breaks CI rather than silently reducing coverage)
- the execution safety, order lifecycle, paper broker, reconciliation/recovery, safety invariant or eligibility suites fail
- session/timezone tests fail under `TZ=UTC` (CI runners are UTC — the natural place to catch the naive-datetime class of bug)
- `TRADING_MODE` is not `paper`
- **eligibility ever reports `LIVE_ELIGIBLE`**
- any secret or key material becomes tracked

The suites themselves carry named regressions for every defect fixed in the previous phase: PB-STALE, duplicate idempotency keys, duplicate submission, split exception classes, synchronous fill divergence, missing decline reasons, the PARTIAL vocabulary mismatch, and reconciliation divergence.

No workflow requires or accepts live broker credentials.

---

## 7. Remaining limitations

Stated rather than hidden.

1. **Docker builds are NOT VERIFIED locally.** Docker Desktop is blocked by corporate policy on the development machine. No image size or build output has been invented. The Docker Build workflow on GitHub runners is the only real verification.
2. **`autobahn 19.11.2` (PYSEC-2020-25)** is transitive from `kiteconnect` and is not fixed. Not silenced.
3. **2 npm vulnerabilities remain** (1 moderate `next`, 1 high `postcss`), both requiring a **Next 16 major upgrade**. The postcss issues are build-time and require attacker-controlled CSS, which this repository does not have. Deferred deliberately, not suppressed.
4. **CI tests only in-memory implementations.** `SqlAlchemyLocalStateStore` is exercised against SQLite, not Postgres. `RedisOrderStore` is not exercised against real Redis at all — `InMemoryOrderStore` is used. True crash recovery is simulated by replaying persisted paper state, not by killing a process. **This is a real coverage gap, not fake coverage.**
5. **No migration validation.** Alembic is a dependency but no migration is run in CI.
6. **`worker` service cannot run** — `docker-compose.yml` references it but there is no `app/worker.py` and no Celery app.
7. **`docker.yml` builds the frontend with `NEXT_PUBLIC_API_URL=http://backend:8000`**, which Next inlines into the *browser* bundle where `backend` is unresolvable. Valid as a CI build check, not deployable as-is.
8. **The live broker has never held a connection.** Everything in the Zerodha adapter remains verified by inspection and simulation only.

---

## 8. Production deployment safety

- `TRADING_MODE` defaults to `paper`; CI asserts it.
- Live requires **both** `trading_mode=live` **and** an explicit authorization flag, and `require_live_eligible()` is enforced at the order path keyed on the broker's declared mode.
- `LIVE_TRADING_ELIGIBILITY` remains **BLOCKED_INSUFFICIENT_DATA**, and CI fails if that ever becomes `LIVE_ELIGIBLE` without review.
- Containers run non-root with `no-new-privileges` and dropped capabilities.
- No secret is baked into any image; `.dockerignore` excludes `.env`.

---

## 9. Final status

See the run table in the phase report for the authoritative, actual GitHub Actions result. **Local green is not the acceptance criterion.**

**PRODUCTION MODEL = NONE. LIVE TRADING = BLOCKED.**
