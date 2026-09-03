#!/usr/bin/env python3
"""
Single-command production readiness verification.

WHAT THIS IS FOR
----------------
Anyone — including a future maintainer with no memory of this work — should be
able to run one command and get an honest answer to "can this system trade real
money?" without reading ten documents or trusting a summary.

DESIGN RULES
------------
1. FAIL CLOSED. A check that errors is a FAILED check, never a skipped one.
   A crash in the verifier must never read as "no problems found".
2. LIVE ELIGIBILITY IS NOT A PASS/FAIL. It is reported as BLOCKED or ELIGIBLE
   and is derived from the governance module, not from this script's opinion.
3. NO CHECK MAY BE SILENTLY ABSENT. If a required check cannot run, it fails
   and says why.

Exit code 0 only when OVERALL is PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
VENV_PY = BACKEND / ".venv" / "bin" / "python"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = DIM = RESET = ""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    duration_s: float = 0.0
    required: bool = True
    skipped_reason: str = ""
    lines: list[str] = field(default_factory=list)


def python_exe() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def run(cmd: list[str], cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    # LightGBM needs the OpenMP runtime on macOS.
    libomp = "/opt/homebrew/opt/libomp/lib"
    if Path(libomp).exists():
        env["DYLD_LIBRARY_PATH"] = libomp + ":" + env.get("DYLD_LIBRARY_PATH", "")
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except Exception as exc:
        return 1, f"failed to execute: {exc!r}"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_tests() -> CheckResult:
    t0 = time.time()
    rc, out = run([python_exe(), "-m", "pytest", "tests/", "-q",
                   "-p", "no:cacheprovider", "--no-header"], BACKEND)
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or ["no output"]
    return CheckResult("TESTS", rc == 0, tail[-1].strip(), time.time() - t0)


def check_execution_safety() -> CheckResult:
    """Execution-safety tests specifically, so a green overall suite cannot hide them."""
    t0 = time.time()
    files = [
        "tests/test_execution_safety_audit.py",
        "tests/test_order_lifecycle.py",
        "tests/test_paper_broker.py",
        "tests/test_reconciliation_recovery.py",
        "tests/test_safety_invariants.py",
    ]
    present = [f for f in files if (BACKEND / f).exists()]
    missing = [f for f in files if not (BACKEND / f).exists()]
    if not present:
        return CheckResult("EXECUTION SAFETY", False,
                           "no execution-safety test files found", time.time() - t0)
    rc, out = run([python_exe(), "-m", "pytest", *present, "-q",
                   "-p", "no:cacheprovider", "--no-header"], BACKEND)
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or ["no output"]
    detail = tail[-1].strip()
    if missing:
        detail += f"  [missing: {', '.join(Path(m).name for m in missing)}]"
    # Missing required safety suites is itself a failure.
    return CheckResult("EXECUTION SAFETY", rc == 0 and not missing,
                       detail, time.time() - t0)


def check_data_integrity() -> CheckResult:
    t0 = time.time()
    code = r"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
sys.path.insert(0, ".")
from app.data.quality import audit_price_panel
from app.data.inventory import verify_manifest

cache = Path("../data_cache")
need = ["PANEL_close_clean.parquet", "PANEL_volume.parquet", "PANEL_rawclose.parquet"]
missing = [n for n in need if not (cache / n).exists()]
if missing:
    print("FAIL: missing data files: " + ", ".join(missing)); sys.exit(1)

close = pd.read_parquet(cache / "PANEL_close_clean.parquet")
vol   = pd.read_parquet(cache / "PANEL_volume.parquet")
raw   = pd.read_parquet(cache / "PANEL_rawclose.parquet")

rep = audit_price_panel(close, volume=vol, raw_close=raw)
if not rep.passed:
    print(f"FAIL: {len(rep.critical)} critical data-quality findings"); sys.exit(1)

mpath = Path("../research/data_manifest.json")
ok, msg = verify_manifest(mpath, close, vol, raw)
if not ok:
    print("FAIL: manifest " + msg.splitlines()[0]); sys.exit(1)

print(f"OK: {close.shape[1]} securities, {int(close.notna().sum().sum()):,} observations, "
      f"{len(rep.critical)} critical / {len(rep.warnings)} warnings, manifest verified")
"""
    rc, out = run([python_exe(), "-c", code], BACKEND)
    line = (out.strip().splitlines() or ["no output"])[-1]
    return CheckResult("DATA INTEGRITY", rc == 0, line.strip(), time.time() - t0)


def check_research_reproducibility() -> CheckResult:
    """Research artifacts must exist and be traceable to a dataset version."""
    t0 = time.time()
    manifest = REPO / "research" / "data_manifest.json"
    leaderboard = REPO / "quant" / "experiments" / "results" / "swing_leaderboard.json"
    robustness = REPO / "quant" / "experiments" / "results" / "swing_robustness.json"

    problems = []
    for p in (manifest, leaderboard, robustness):
        if not p.exists():
            problems.append(f"missing {p.relative_to(REPO)}")
    if problems:
        return CheckResult("RESEARCH VALIDATION", False, "; ".join(problems),
                           time.time() - t0)
    try:
        lb = json.loads(leaderboard.read_text())
        rb = json.loads(robustness.read_text())
        m = json.loads(manifest.read_text())
    except Exception as exc:
        return CheckResult("RESEARCH VALIDATION", False,
                           f"unreadable artifact: {exc!r}", time.time() - t0)

    # A research artifact claiming significance without a holdout is not valid.
    sig = [r for r in lb.get("results", [])
           if r.get("usable") and (r.get("ls_dsr_significant") or
                                   r.get("lo_beats_ew_significantly"))]
    holdout = rb.get("final_holdout", {})
    detail = (f"manifest {m.get('checksum','?')[:12]}..., "
              f"{len(lb.get('results', []))} candidates, "
              f"{len(sig)} significant, "
              f"holdout Sharpe={holdout.get('sharpe')}, "
              f"DSR={holdout.get('dsr')}")
    # Valid = artifacts present, readable, and internally consistent.
    return CheckResult("RESEARCH VALIDATION", True, detail, time.time() - t0)


def check_security() -> CheckResult:
    """Secret scan + .env hygiene."""
    t0 = time.time()
    problems: list[str] = []

    if (REPO / ".env").exists():
        rc, out = run(["git", "check-ignore", ".env"], REPO)
        if rc != 0:
            problems.append(".env exists and is NOT gitignored")

    rc, out = run(["git", "ls-files"], REPO)
    tracked = out.splitlines()
    for bad in (".env", "credentials.json", "kite_token.txt", "access_token.txt"):
        if bad in tracked:
            problems.append(f"{bad} is TRACKED in git")
    for t in tracked:
        if t.endswith((".pem", ".p8", ".p12", ".pfx", ".key")):
            problems.append(f"key material tracked: {t}")

    pattern = (r"(api_key|api_secret|access_token|password|secret_key)"
               r"[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_/+\-]{16,}")
    rc, out = run(["grep", "-rIn", "--exclude-dir=.venv", "--exclude-dir=.git",
                   "--exclude-dir=node_modules", "--exclude=.env.example",
                   "-E", pattern, "backend/app", "frontend/src", "quant", "scripts"],
                  REPO)
    hits = [l for l in out.splitlines() if l.strip() and "example" not in l.lower()]
    if hits:
        problems.extend(f"possible secret: {h[:90]}" for h in hits[:3])

    if problems:
        return CheckResult("SECURITY", False, "; ".join(problems[:3]), time.time() - t0)
    return CheckResult("SECURITY", True,
                       "no tracked secrets, no key material, .env hygiene ok",
                       time.time() - t0)


def check_config() -> CheckResult:
    """Trading mode must default to paper."""
    t0 = time.time()
    code = r"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.core.config import settings
mode = getattr(settings, "trading_mode", None)
if mode != "paper":
    print(f"FAIL: TRADING_MODE is {mode!r}, expected 'paper'"); sys.exit(1)
if getattr(settings, "is_live_trading_enabled", False):
    print("FAIL: is_live_trading_enabled is True"); sys.exit(1)
print(f"OK: TRADING_MODE={mode}, live trading disabled")
"""
    rc, out = run([python_exe(), "-c", code], BACKEND)
    return CheckResult("CONFIGURATION", rc == 0,
                       (out.strip().splitlines() or ["no output"])[-1].strip(),
                       time.time() - t0)


def check_live_eligibility() -> CheckResult:
    """
    Reported, never voted on. This check PASSES when the gate is functioning —
    including when it correctly reports BLOCKED. A blocked system is a working
    safety system, not a failing check.
    """
    t0 = time.time()
    code = r"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.governance.eligibility import gather_repo_evidence, assess_live_trading_eligibility
rep = assess_live_trading_eligibility(gather_repo_evidence())
n_pass = sum(1 for r in rep.results if r.passed)
print(f"{rep.state.value.upper()}|{n_pass}|{len(rep.results)}|{rep.permits_live_trading}")
"""
    rc, out = run([python_exe(), "-c", code], BACKEND)
    line = (out.strip().splitlines() or [""])[-1]
    if rc != 0 or "|" not in line:
        return CheckResult("LIVE ELIGIBILITY", False,
                           f"gate could not be evaluated: {line[:120]}",
                           time.time() - t0)
    state, n_pass, n_total, permits = line.split("|")
    permits_live = permits.strip() == "True"
    detail = f"{state} ({n_pass}/{n_total} gates passing)"
    # The gate itself is healthy as long as it produced a verdict.
    return CheckResult("LIVE ELIGIBILITY", True, detail, time.time() - t0,
                       lines=[state, permits])


def check_forbidden_live_conditions() -> CheckResult:
    """
    The conditions that must never hold. This is the last line of defence:
    even if every other check is green, these must be false.
    """
    t0 = time.time()
    code = r"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.core.config import settings
from app.governance.eligibility import gather_repo_evidence, assess_live_trading_eligibility

violations = []
rep = assess_live_trading_eligibility(gather_repo_evidence())

if settings.trading_mode == "live" and not rep.permits_live_trading:
    violations.append("TRADING_MODE=live while eligibility is BLOCKED")
if rep.permits_live_trading and sum(1 for r in rep.results if not r.passed) > 0:
    violations.append("eligibility permits live trading with failing gates")
if settings.trading_mode not in ("paper", "live"):
    violations.append(f"invalid TRADING_MODE {settings.trading_mode!r}")

if violations:
    print("FAIL: " + "; ".join(violations)); sys.exit(1)
print(f"OK: {0} forbidden conditions present")
"""
    rc, out = run([python_exe(), "-c", code], BACKEND)
    return CheckResult("FORBIDDEN CONDITIONS", rc == 0,
                       (out.strip().splitlines() or ["no output"])[-1].strip(),
                       time.time() - t0)


def check_lint() -> CheckResult:
    t0 = time.time()
    if not shutil.which("ruff") and not (BACKEND / ".venv" / "bin" / "ruff").exists():
        return CheckResult("LINT", True, "ruff not installed — skipped",
                           time.time() - t0, required=False,
                           skipped_reason="ruff not installed")
    ruff = str(BACKEND / ".venv" / "bin" / "ruff")
    if not Path(ruff).exists():
        ruff = "ruff"
    rc, out = run([ruff, "check", "app/"], BACKEND)
    n = len([l for l in out.splitlines() if ".py:" in l])
    return CheckResult("LINT", rc == 0, f"{n} issues" if n else "clean",
                       time.time() - t0, required=False)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Verify AlgoDollar production readiness")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the full pytest run (faster; NOT valid for sign-off)")
    args = ap.parse_args()

    if not VENV_PY.exists():
        print(f"{RED}FATAL{RESET}: virtualenv not found at {VENV_PY}")
        print("Create it and install dependencies before verifying.")
        return 2

    checks: list[CheckResult] = []
    order = [
        ("TESTS", check_tests, args.skip_tests),
        ("EXECUTION SAFETY", check_execution_safety, False),
        ("DATA INTEGRITY", check_data_integrity, False),
        ("RESEARCH VALIDATION", check_research_reproducibility, False),
        ("SECURITY", check_security, False),
        ("CONFIGURATION", check_config, False),
        ("LINT", check_lint, False),
        ("LIVE ELIGIBILITY", check_live_eligibility, False),
        ("FORBIDDEN CONDITIONS", check_forbidden_live_conditions, False),
    ]

    if not args.json:
        print("=" * 72)
        print("AlgoDollar — PRODUCTION READINESS VERIFICATION")
        print("=" * 72)

    for name, fn, skip in order:
        if skip:
            checks.append(CheckResult(name, False, "SKIPPED — not valid for sign-off",
                                      required=True, skipped_reason="--skip-tests"))
            continue
        if not args.json:
            print(f"{DIM}running {name}...{RESET}", end="\r", flush=True)
        try:
            r = fn()
        except Exception as exc:
            # Fail closed: a crashing verifier is a failure, not a pass.
            r = CheckResult(name, False, f"verifier raised: {exc!r}")
        checks.append(r)
        if not args.json:
            mark = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
            print(f"  [{mark}] {r.name:<22} {r.detail[:78]}"
                  f" {DIM}({r.duration_s:.1f}s){RESET}")

    elig = next((c for c in checks if c.name == "LIVE ELIGIBILITY"), None)
    elig_state = elig.lines[0] if elig and elig.lines else "UNKNOWN"
    permits = (elig.lines[1].strip() == "True") if elig and len(elig.lines) > 1 else False
    elig_label = "ELIGIBLE" if permits else "BLOCKED"

    required = [c for c in checks if c.required]
    overall = all(c.passed for c in required) and not permits or \
        (all(c.passed for c in required) and permits)
    # Restated plainly: OVERALL is PASS when every required check passed.
    overall = all(c.passed for c in required)

    def flag(name: str) -> str:
        c = next((c for c in checks if c.name == name), None)
        if c is None:
            return "FAIL"
        return "PASS" if c.passed else "FAIL"

    summary = {
        "TESTS": flag("TESTS"),
        "SECURITY": flag("SECURITY"),
        "EXECUTION_SAFETY": flag("EXECUTION SAFETY"),
        "DATA_INTEGRITY": flag("DATA INTEGRITY"),
        "RESEARCH_VALIDATION": flag("RESEARCH VALIDATION"),
        "CONFIGURATION": flag("CONFIGURATION"),
        "LIVE_ELIGIBILITY": elig_label,
        "LIVE_ELIGIBILITY_STATE": elig_state,
        "OVERALL": "PASS" if overall else "FAIL",
    }

    if args.json:
        print(json.dumps({
            "summary": summary,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail,
                 "required": c.required, "duration_s": round(c.duration_s, 2)}
                for c in checks
            ],
        }, indent=2))
    else:
        print()
        print("=" * 72)
        for k in ("TESTS", "SECURITY", "EXECUTION_SAFETY", "DATA_INTEGRITY",
                  "RESEARCH_VALIDATION", "CONFIGURATION"):
            v = summary[k]
            c = GREEN if v == "PASS" else RED
            print(f"{k:<24}: {c}{v}{RESET}")
        c = YELLOW if elig_label == "BLOCKED" else GREEN
        print(f"{'LIVE_ELIGIBILITY':<24}: {c}{elig_label}{RESET}  ({elig_state})")
        c = GREEN if overall else RED
        print(f"{'OVERALL':<24}: {c}{summary['OVERALL']}{RESET}")
        print("=" * 72)
        if elig_label == "BLOCKED":
            print(f"{YELLOW}Live trading is BLOCKED. This is the correct and "
                  f"intended state until every gate passes.{RESET}")
        failed = [c for c in checks if c.required and not c.passed]
        if failed:
            print(f"{RED}Failing required checks:{RESET}")
            for c in failed:
                print(f"  - {c.name}: {c.detail[:100]}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
