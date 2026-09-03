# Research data manifests

Every research result in this repository must be traceable to a specific
dataset version. That is what this directory is for.

## Why

A performance number without a dataset version is not a result, it is an
anecdote. If someone reports a Sharpe ratio and the data has since been
re-downloaded, revised by the vendor, or silently re-cleaned, there is no way
to tell whether a later disagreement is a genuine finding or a different input.

`data_manifest.json` records what the data was, where it came from, and — just
as importantly — what is wrong with it. Survivorship treatment, point-in-time
treatment and corporate-action policy are fields, not footnotes, because they
determine what the data can and cannot be used to claim.

## Format

JSON rather than YAML, for one reason: it is what
`app/data/inventory.py` reads and writes, so the manifest is produced by the
code that uses it rather than maintained by hand. A hand-maintained manifest
drifts from reality; a generated one cannot.

## Fields

| Field | Meaning |
|---|---|
| `source` | Vendor and access method |
| `version` | Manifest version stamp |
| `acquisition_date` | When the data was retrieved |
| `date_range` | First and last date present |
| `instrument_universe` | How the universe was chosen — including whether it is point-in-time |
| `n_instruments` | Security count |
| `frequency` | Bar frequency |
| `timezone` | Timezone convention of the index |
| `adjustment_method` | How prices were adjusted |
| `corporate_action_policy` | How splits, bonuses and demergers were handled |
| `survivorship_policy` | Whether delisted securities are present |
| `point_in_time_policy` | Whether historical membership was reconstructed |
| `checksum` | SHA-256 of the actual numeric content |
| `known_limitations` | What this dataset cannot support |
| `preprocessing` | Every transformation applied before research |

## The checksum is of the data, not the file

`_content_checksum` hashes the numeric values, the column set and the index
bounds — not the parquet bytes. Re-saving the same data with a different
library version produces the same checksum; changing a single price does not.

Verified behaviour:

```
unmodified data           -> matches
one value changed 0.00001% -> MISMATCH detected
one symbol removed         -> MISMATCH detected
```

A verification that passed on modified data would be worthless, so this was
tested against deliberately corrupted inputs rather than assumed.

## Usage

Regenerate after acquiring or re-cleaning data:

```python
from app.data.inventory import build_manifest, write_manifest
```

Verify before trusting a research run:

```python
from app.data.inventory import verify_manifest
ok, msg = verify_manifest(Path("research/data_manifest.json"), close, volume, raw)
```

`scripts/verify_production_readiness` runs this check as part of DATA
INTEGRITY. A research run that cannot verify its manifest is not reproducible,
and the verifier reports that rather than proceeding quietly.

## Current manifest

`data_manifest.json` describes the only dataset used for research so far: NSE
daily large/mid-cap prices from Yahoo Finance.

Its `survivorship_policy` field records that the dataset is **filtered** —
delisted securities are absent, verified directly. Its `point_in_time_policy`
records that no point-in-time membership exists. Both are the reason
`docs/DATA_INTEGRITY_REPORT.md` concludes the dataset is adequate for
engineering verification and inadequate for any performance claim.
