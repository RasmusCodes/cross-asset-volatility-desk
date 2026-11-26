"""Pull the raw data from WRDS into data/raw/ (parquet + manifest.json). Resumable: files already recorded with the
same SQL are skipped, so a stopped run continues where it stopped.

    python scripts/pull_data.py --stage reference        # small tables (minutes)
    python scripts/pull_data.py --stage universe         # point-in-time top-50 S&P members -> data/store/universe_top50.parquet
    python scripts/pull_data.py --stage panel            # underlying prices, forwards, standardized surfaces
    python scripts/pull_data.py --stage cboe             # Cboe SPX 2025-07..2026-11 + cross-vendor sample
    python scripts/pull_data.py --stage spx              # every SPX option row 1996-2025 (largest)
    python scripts/pull_data.py --stage etf              # SPY/QQQ/IWM month-end chains + near-the-money monthlies
    python scripts/pull_data.py --stage all              # in that order
    python scripts/pull_data.py --stage spx --dry-run    # print the SQL only
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.data import catalog, universe  # noqa: E402
from voldesk.data.wrds import Manifest, copy_to_parquet, engine  # noqa: E402

STAGES = ["reference", "universe", "panel", "cboe", "spx", "etf"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def build_universe() -> pd.DataFrame:
    raw = config.RAW
    uni = universe.top_members(pd.read_parquet(raw / "crsp/msp500list.parquet"),
                               pd.read_parquet(raw / "crsp/msf_members.parquet"),
                               pd.read_parquet(raw / "wrdsapps/opcrsphist.parquet"),
                               pd.read_parquet(raw / "crsp/msenames_members.parquet"), n=50)
    config.ensure(config.STORE)
    uni.to_parquet(config.STORE / "universe_top50.parquet", index=False)
    miss = uni["secid"].isna().mean()
    log(f"universe: {uni['month_end'].nunique()} months, {uni['permno'].nunique()} permnos, "
        f"{uni['secid'].nunique()} secids, {miss:.1%} member-months without an OptionMetrics link")
    return uni


def single_name_secids() -> list[int]:
    p = config.STORE / "universe_top50.parquet"
    if not p.exists():
        raise SystemExit("Run --stage universe first.")
    u = pd.read_parquet(p)
    return sorted(int(s) for s in u["secid"].dropna().unique())


def specs_for(stage: str):
    if stage == "reference":
        return catalog.reference_specs()
    if stage == "panel":
        return catalog.panel_specs(single_name_secids())
    if stage == "cboe":
        return catalog.cboe_specs()
    if stage == "spx":
        return catalog.spx_chain_specs()
    if stage == "etf":
        return catalog.etf_chain_specs()
    raise ValueError(stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on the destination path")
    args = ap.parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]
    config.ensure(config.RAW)
    man = Manifest(config.RAW)
    eng = None if args.dry_run else engine()
    failures = []
    for st in stages:
        if st == "universe":
            if not args.dry_run:
                build_universe()
            continue
        specs = [s for s in specs_for(st) if args.only in s.rel]
        log(f"stage {st}: {len(specs)} files")
        for i, spec in enumerate(specs, 1):
            if args.dry_run:
                print(f"-- {spec.rel}\n{spec.sql}\n")
                continue
            if man.has(spec.rel, spec.sql):
                continue
            for attempt in range(1, 4):
                try:
                    rec = copy_to_parquet(eng, spec, config.RAW)
                    man.add(spec.rel, rec)
                    log(f"[{st} {i}/{len(specs)}] {spec.rel}: {rec['rows']:,} rows, {rec['bytes'] / 1e6:.1f} MB, "
                        f"{rec['seconds_total']:.0f}s")
                    break
                except Exception as e:  # noqa: BLE001 - network hiccups: retry, then record and continue
                    log(f"[{st} {i}/{len(specs)}] {spec.rel}: attempt {attempt} failed: {str(e).splitlines()[0][:200]}")
                    if attempt == 3:
                        failures.append(spec.rel)
                        traceback.print_exc()
                    time.sleep(10 * attempt)
    if failures:
        log(f"FAILED ({len(failures)}): {failures}")
        raise SystemExit(1)
    log("done")


if __name__ == "__main__":
    main()
