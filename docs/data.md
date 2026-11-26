# Data access

## Sources

| Source (WRDS library) | Used for | Coverage used |
|---|---|---|
| OptionMetrics IvyDB US (`optionm`) | option prices, implied vols, Greeks; underlying prices; zero curve; index dividend yields; distributions; forwards; standardized ATM options and volatility surfaces | 1996-01-04 .. 2025-08-29 |
| Cboe option end-of-day data (`cboe`) | SPX option prices and implied vols after OptionMetrics ends; cross-vendor check; VIX/VXO/VXN/VXD history; rate curve | 1998 .. 2026-11-03 |
| Thomson Reuters Datastream futures (`tr_ds_fut`) | Cboe VIX futures settlements (contract code 429) | 2005-12 .. 2026-11-19 |
| CRSP (`crsp`) | S&P 500 membership, monthly market caps and names | 1995 .. 2024-12 |
| WRDS link (`wrdsapps_link_crsp_optionm`) | CRSP permno → OptionMetrics secid | point in time |
| Federal Reserve (`frb`) | Treasury and policy rates (reference only) | 1990 .. 2025-02 |

All are licensed: you need your own subscriptions, and nothing pulled from them is published in this repository or
in the demo.

## Connecting

The code uses libpq's standard password file and never handles a password. Log in once with WRDS's own tooling (the
`wrds` Python package's first connection offers to save the credentials) so that `%APPDATA%\postgresql\pgpass.conf`
(Windows) or `~/.pgpass` contains the line for `wrds-pgdata.wharton.upenn.edu:9737:wrds`. The username is read from
that line or from `WRDS_USERNAME`. Then:

```bash
python scripts/wrds_probe.py          # what your account can see (metadata only)
python scripts/pull_data.py --stage all --dry-run    # print every query
python scripts/pull_data.py --stage all              # pull (resumable)
python scripts/manifest_summary.py                   # rows and bytes per stage
```

Every raw file's SQL, row count, columns, size, sha256 and retrieval time are in `data/raw/manifest.json`. Set
`VOLDESK_DATA` to keep the data somewhere other than `./data`.

## Without WRDS

The tests and CI run on `voldesk.data.synthetic`, which generates Heston-priced option chains in OptionMetrics'
column layout. Every frame it returns is marked `synthetic = True` / `source = "SYNTHETIC"`, and no research script
reads it: the studies need `data/store` and stop if it is missing.

Data quality, cleaning rules and the cross-vendor comparison are in [reports/01_data.md](../reports/01_data.md).
