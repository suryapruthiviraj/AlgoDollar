"""
Dataset inventory and reproducibility manifest.

WHY THIS EXISTS
---------------
A previous report described the research dataset as "99 NSE symbols, 4,262
daily bars" without stating whether 4,262 was per instrument or in total.
That ambiguity is exactly the kind of imprecision that makes a research result
impossible to reproduce or to argue with. This module computes the inventory
from the data itself and emits a machine-readable manifest, so that a claim
about a result is always traceable to a specific dataset version.

The manifest records not only what the data IS but what is WRONG with it:
survivorship treatment, point-in-time treatment, and corporate-action policy
are fields, not footnotes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------

@dataclass
class DatasetInventory:
    """Exact, unambiguous dimensions of a price panel."""
    n_securities: int
    total_observations: int              # non-null (security, date) pairs
    observations_per_security_mean: float
    observations_per_security_min: int
    observations_per_security_max: int
    n_trading_dates: int                 # distinct dates in the index
    earliest_date: str
    latest_date: str
    currently_listed: int
    delisted_included: int
    incomplete_histories: int            # started late or ended early
    complete_histories: int
    n_sectors: int
    point_in_time_constituents: bool
    corporate_action_coverage: str
    fundamental_data_coverage: str
    intraday_data_coverage: str

    def summary(self) -> str:
        return "\n".join([
            f"Number of securities        : {self.n_securities}",
            f"Total observations          : {self.total_observations:,} "
            f"(non-null security-date pairs)",
            f"Observations per security   : mean {self.observations_per_security_mean:,.0f}, "
            f"min {self.observations_per_security_min:,}, "
            f"max {self.observations_per_security_max:,}",
            f"Distinct trading dates      : {self.n_trading_dates:,}",
            f"Earliest date               : {self.earliest_date}",
            f"Latest date                 : {self.latest_date}",
            f"Currently listed securities : {self.currently_listed}",
            f"Delisted securities         : {self.delisted_included}",
            f"Incomplete histories        : {self.incomplete_histories}",
            f"Complete histories          : {self.complete_histories}",
            f"Number of sectors           : {self.n_sectors}",
            f"Point-in-time constituents  : {self.point_in_time_constituents}",
            f"Corporate-action coverage   : {self.corporate_action_coverage}",
            f"Fundamental-data coverage   : {self.fundamental_data_coverage}",
            f"Intraday-data coverage      : {self.intraday_data_coverage}",
        ])


@dataclass
class DatasetManifest:
    """
    Machine-readable description of a research dataset.

    Two runs quoting the same manifest_checksum used byte-identical inputs.
    """
    name: str
    source: str
    version: str
    acquisition_date: str
    date_range: list[str]
    instrument_universe: str
    n_instruments: int
    frequency: str
    timezone: str
    adjustment_method: str
    corporate_action_policy: str
    survivorship_policy: str
    point_in_time_policy: str
    checksum: str
    license_notes: str
    known_limitations: list[str] = field(default_factory=list)
    preprocessing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------

def _content_checksum(*frames: pd.DataFrame) -> str:
    """
    Stable content hash of the actual numeric data.

    Hashing the file would change with irrelevant metadata; hashing the values
    means the checksum identifies the DATA, which is what reproducibility
    depends on.
    """
    h = hashlib.sha256()
    for df in frames:
        if df is None:
            continue
        h.update(str(sorted(map(str, df.columns))).encode())
        h.update(str(df.index.min()).encode())
        h.update(str(df.index.max()).encode())
        arr = np.ascontiguousarray(
            df.to_numpy(dtype="float64", na_value=np.nan)
        )
        h.update(np.nan_to_num(arr, nan=-9.87654321e300).tobytes())
    return h.hexdigest()


def build_inventory(
    close: pd.DataFrame,
    sector_map: Optional[dict[str, str]] = None,
    *,
    point_in_time_constituents: bool = False,
    corporate_action_coverage: str = "unknown",
    fundamental_data_coverage: str = "unknown",
    intraday_data_coverage: str = "unknown",
    delisted_symbols: Optional[set[str]] = None,
) -> DatasetInventory:
    """
    Compute exact dataset dimensions from a price panel.

    `delisted_symbols` is passed in rather than inferred: a symbol whose series
    ends early might be delisted, or the vendor might simply lack recent data.
    Guessing would be exactly the kind of unverified claim this module exists
    to prevent.
    """
    per_sec = close.notna().sum()
    last_date = close.index.max()
    first_date = close.index.min()

    # A history is "complete" if it spans essentially the whole panel.
    first_valid = close.apply(lambda c: c.first_valid_index())
    last_valid = close.apply(lambda c: c.last_valid_index())
    starts_late = first_valid > (first_date + pd.Timedelta(days=365))
    ends_early = last_valid < (last_date - pd.Timedelta(days=30))
    incomplete = int((starts_late | ends_early).sum())

    delisted = delisted_symbols or set()

    return DatasetInventory(
        n_securities=int(close.shape[1]),
        total_observations=int(per_sec.sum()),
        observations_per_security_mean=float(per_sec.mean()),
        observations_per_security_min=int(per_sec.min()),
        observations_per_security_max=int(per_sec.max()),
        n_trading_dates=int(close.shape[0]),
        earliest_date=str(first_date.date()),
        latest_date=str(last_date.date()),
        currently_listed=int(close.shape[1] - len(delisted)),
        delisted_included=len(delisted),
        incomplete_histories=incomplete,
        complete_histories=int(close.shape[1] - incomplete),
        n_sectors=len(set((sector_map or {}).values())) if sector_map else 0,
        point_in_time_constituents=point_in_time_constituents,
        corporate_action_coverage=corporate_action_coverage,
        fundamental_data_coverage=fundamental_data_coverage,
        intraday_data_coverage=intraday_data_coverage,
    )


def build_manifest(
    name: str,
    close: pd.DataFrame,
    *,
    source: str,
    universe_description: str,
    adjustment_method: str,
    corporate_action_policy: str,
    survivorship_policy: str,
    point_in_time_policy: str,
    known_limitations: list[str],
    preprocessing: list[str],
    extra_frames: tuple[pd.DataFrame, ...] = (),
    frequency: str = "daily",
    tz: str = "naive dates (Asia/Kolkata trading sessions)",
    license_notes: str = "unspecified — verify vendor terms before any redistribution or commercial use",
) -> DatasetManifest:
    """Build a reproducibility manifest for a dataset."""
    return DatasetManifest(
        name=name,
        source=source,
        version=datetime.now(timezone.utc).strftime("%Y%m%d"),
        acquisition_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        date_range=[str(close.index.min().date()), str(close.index.max().date())],
        instrument_universe=universe_description,
        n_instruments=int(close.shape[1]),
        frequency=frequency,
        timezone=tz,
        adjustment_method=adjustment_method,
        corporate_action_policy=corporate_action_policy,
        survivorship_policy=survivorship_policy,
        point_in_time_policy=point_in_time_policy,
        checksum=_content_checksum(close, *extra_frames),
        license_notes=license_notes,
        known_limitations=known_limitations,
        preprocessing=preprocessing,
    )


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2))


def verify_manifest(manifest_path: Path, *frames: pd.DataFrame) -> tuple[bool, str]:
    """
    Confirm that data on disk still matches a recorded manifest.

    Returns (matches, message). A research run that cannot verify its manifest
    is not reproducible and should say so rather than proceeding quietly.
    """
    if not manifest_path.exists():
        return False, f"manifest not found: {manifest_path}"
    recorded = json.loads(manifest_path.read_text())
    actual = _content_checksum(*frames)
    if actual == recorded.get("checksum"):
        return True, f"checksum matches ({actual[:16]}...)"
    return False, (
        f"CHECKSUM MISMATCH — data has changed since the manifest was written.\n"
        f"  manifest: {recorded.get('checksum', 'missing')[:32]}\n"
        f"  actual  : {actual[:32]}\n"
        f"Results produced now are not comparable to results quoting this manifest."
    )
