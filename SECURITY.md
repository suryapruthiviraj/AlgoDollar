# Security Policy

AlgoDollar connects to a live brokerage account and can, in principle, move real money. The security posture below is written with that in mind.

**Current status: live trading is BLOCKED and no strategy is validated.** See [docs/CI_SECURITY_AUDIT.md](docs/CI_SECURITY_AUDIT.md) for the full audit.

---

## Reporting a vulnerability

Report privately — open a [GitHub security advisory](https://github.com/suryapruthiviraj/AlgoDollar/security/advisories/new) rather than a public issue.

Please include reproduction steps and the affected commit. If a report involves broker credentials or live order placement, say so in the first line so it is triaged first.

Do not test against a live brokerage account. Use `TRADING_MODE=paper`, which is the default.

---

## Secret management

**No credential is ever committed.** Configuration is loaded from the environment via `pydantic-settings`.

| | |
|---|---|
| `.env` | gitignored (`.env.example` is tracked via a `!` negation) |
| `.env.example` | placeholders only, never real values |
| Key material (`.pem`, `.key`, `.p8`, `.p12`, `.pfx`) | never committed — verified across full history |
| CI | requires **no** broker credential; the safety gates run on fork PRs with no secrets at all |

CI asserts on every push that no secret file has become tracked.

### Historical note

One credential was committed: a Postgres password in `docker-compose.yml`, present from the initial commit until it was removed. It was a **local development default** for a containerized database, never a production credential, and never used against a real database. **Rotation is not required**; anyone who ran the old compose file locally should change their local password.

It remains in git history. Deleting the current file does not remove that, and removing it would require a history rewrite affecting every clone — not justified for a dev-only default. Recorded here so it is not mistaken for a clean history.

---

## Required secrets

Set these in the environment or a secret manager — never in a file that gets committed:

```
KITE_API_KEY           Zerodha Kite Connect API key
KITE_API_SECRET        Zerodha Kite Connect API secret
KITE_ACCESS_TOKEN      Daily access token (expires each trading day)
DATABASE_URL           Includes the Postgres password
REDIS_URL              Includes the Redis password
SECRET_KEY             JWT signing key, >=32 random characters
```

Zerodha access tokens expire daily and can only be reissued through an interactive login. `ZerodhaBroker.disconnect()` deliberately does **not** invalidate the token by default — doing so on every restart would end trading for the day and require a human at a browser.

---

## Trading safety controls

These are security controls, not features. CI fails the build if any of them regress.

| Control | Behaviour |
|---|---|
| `TRADING_MODE` | Defaults to `paper`. CI asserts it. |
| Live trading | Requires **both** `trading_mode=live` **and** an explicit authorization flag |
| Eligibility gate | 31 fail-closed gates; `require_live_eligible()` is enforced on the order path, keyed on the broker's *declared mode* rather than a config flag |
| Kill switch | Any source saying stop means stop; a source that cannot be read counts as **ACTIVE** |
| Reconciliation | Only `RECONCILIATION_OK` permits trading; an unreachable broker is `UNAVAILABLE`, never OK |
| Ambiguous orders | Never automatically retried — a lost response may mean the order reached the exchange |

CI fails if eligibility ever reports `LIVE_ELIGIBLE` without explicit review.

---

## Automated scanning

| Scanner | Scope | Runs |
|---|---|---|
| TruffleHog | Secrets, full history | push, PR |
| pip-audit | Python dependencies | push, PR, weekly |
| npm audit | Node dependencies | push, PR, weekly |
| Bandit | Python static analysis | push, PR |
| Dependabot | pip, npm, github-actions | scheduled |

**No scanner is disabled and no severity floor is lowered.** Exactly one advisory is ignored by ID — `PYSEC-2020-25` (autobahn) — because `kiteconnect`, Zerodha's official SDK, hard-pins the vulnerable version in every published release. Documented in `docs/CI_SECURITY_AUDIT.md` §7 with a review trigger.

---

## Known limitations

Stated plainly rather than omitted:

- 2 npm advisories remain (1 moderate `next`, 1 high `postcss`); both require a Next 16 major upgrade. The postcss issues are build-time and require attacker-controlled CSS, which this repository does not have.
- CI exercises in-memory and SQLite implementations, not Postgres or real Redis. This is a coverage gap, not covered ground.
- The live broker adapter has never held a real connection.
- Docker images are built and verified in CI only; local Docker is unavailable in the development environment.

---

## Supported versions

Pre-1.0. Only `main` receives fixes.
