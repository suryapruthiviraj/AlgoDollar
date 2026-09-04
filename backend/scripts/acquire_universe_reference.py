#!/usr/bin/env python
"""
Acquire NSE's own reference data for point-in-time universe construction.

WHAT IS AVAILABLE, AND WHAT IS NOT
-----------------------------------
Probed directly against NSE's public archives:

  AVAILABLE
    EQUITY_L.csv        2,570 currently-listed equities WITH `DATE OF LISTING`
    symbolchange.csv    1,057 ticker changes with effective dates
    ind_nifty500list    today's NIFTY 500 membership (a SNAPSHOT)

  NOT AVAILABLE (404)
    DelistedCompanies.csv
    SUSPENSION.csv
    dated historical index constituent files

That gap decides the universe definition. Historical NIFTY 500 membership
cannot be established from any source reachable here, and Phase 2 of this work
forbids guessing it — so the universe is NOT defined by index membership. It is
defined by listing status and liquidity, both of which ARE derivable
point-in-time from real evidence:

  * entry  — NSE's published DATE OF LISTING, plus the first bar actually observed
  * exit   — the last bar actually observed (a series that stops is a delisting,
             a suspension, or a ticker change; all three end eligibility)
  * liquid — the trailing-window test the design already specifies

THE RESIDUAL BIAS, STATED
-------------------------
EQUITY_L.csv lists companies that are listed TODAY. Companies that delisted
before today are absent from it, so a pool built from it alone is still
survivor-biased — just far less so than a 108-name index snapshot. Known
delisted names are therefore added explicitly from the survivorship probe, and
the manifest records exactly how many came from each source so the remaining
gap is visible rather than implied.
"""

from __future__ import annotations

import csv
import io
import json
import ssl
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research.datasets import DEFAULT_ROOT  # noqa: E402

OUT = DEFAULT_ROOT / "universe_reference.json"

_UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

SOURCES = {
    "equity_list": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "symbol_changes": "https://nsearchives.nseindia.com/content/equities/symbolchange.csv",
    "nifty500_snapshot": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}


def _fetch(url: str) -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=40, context=ctx) as resp:
        return resp.read().decode("utf8", "replace")


def _parse_listing_dates(body: str) -> dict[str, dict]:
    """
    SYMBOL -> {listing_date, name, series, isin}.

    Only the EQ series is kept: that is the ordinary equity segment the system
    trades. Including SM/ST (SME) or BE (trade-for-trade) would put instruments
    with different settlement and liquidity behaviour into the same universe.
    """
    out: dict[str, dict] = {}
    reader = csv.DictReader(io.StringIO(body))
    for row in reader:
        clean = { (k or "").strip(): (v or "").strip() for k, v in row.items() }
        sym = clean.get("SYMBOL", "")
        series = clean.get("SERIES", "")
        raw = clean.get("DATE OF LISTING", "")
        if not sym or series != "EQ" or not raw:
            continue
        try:
            listed = datetime.strptime(raw, "%d-%b-%Y").date()
        except ValueError:
            continue
        out[sym] = {
            "listing_date": listed.isoformat(),
            "name": clean.get("NAME OF COMPANY", ""),
            "series": series,
            "isin": clean.get("ISIN NUMBER", ""),
        }
    return out


def _parse_symbol_changes(body: str) -> list[dict]:
    """
    Ticker changes: (company, old symbol, new symbol, effective date).

    The file has no header row, and it contains mutual-fund scheme renames as
    well as equities. It is stored whole and filtered by the consumer, because
    deciding here which rows are 'real' would be a judgement recorded nowhere.
    """
    out: list[dict] = []
    for row in csv.reader(io.StringIO(body)):
        if len(row) < 4:
            continue
        company, old, new, when = (c.strip() for c in row[:4])
        parsed = None
        for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
            try:
                parsed = datetime.strptime(when, fmt).date().isoformat()
                break
            except ValueError:
                continue
        out.append({
            "company": company, "old_symbol": old, "new_symbol": new,
            "effective_date": parsed, "raw_date": when,
        })
    return out


def _parse_index_members(body: str) -> list[str]:
    out = []
    for row in csv.DictReader(io.StringIO(body)):
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        sym = clean.get("Symbol", "")
        if sym:
            out.append(sym)
    return sorted(out)


def main() -> int:
    DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": SOURCES,
        "unavailable": {
            "historical_index_constituents": (
                "NSE serves no dated constituent files (404). Historical NIFTY "
                "500 membership therefore CANNOT be established from any source "
                "reachable here, and is not guessed."
            ),
            "delisted_register": (
                "DelistedCompanies.csv and SUSPENSION.csv both 404. Delisting is "
                "instead inferred from the price series ending, which is real "
                "evidence but carries no reason code."
            ),
        },
    }

    body = _fetch(SOURCES["equity_list"])
    listings = _parse_listing_dates(body)
    doc["listings"] = listings
    print(f"equity list      : {len(listings)} EQ-series symbols with listing dates")

    body = _fetch(SOURCES["symbol_changes"])
    changes = _parse_symbol_changes(body)
    doc["symbol_changes"] = changes
    print(f"symbol changes   : {len(changes)} rows")

    body = _fetch(SOURCES["nifty500_snapshot"])
    members = _parse_index_members(body)
    doc["nifty500_snapshot"] = {
        "symbols": members,
        "as_of": date.today().isoformat(),
        "warning": (
            "TODAY'S membership. Recorded for provenance only. It must NEVER be "
            "applied to a historical date — doing so is the survivorship bias "
            "this whole exercise exists to remove."
        ),
    }
    print(f"nifty500 snapshot: {len(members)} symbols (today only)")

    OUT.write_text(json.dumps(doc, indent=2))
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
