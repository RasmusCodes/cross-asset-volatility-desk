"""Read-only probe of what a WRDS account can see for this project: schemas, OptionMetrics tables and their year range,
security ids for the planned underlyings, and row counts. It pulls no data rows (counts and metadata only).

Authentication is libpq's own: the standard PostgreSQL password file (%APPDATA%\\postgresql\\pgpass.conf on Windows,
~/.pgpass elsewhere) holds the WRDS line. This script never reads or prints the password, and prints no username.

    python scripts/wrds_probe.py [--counts]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
TICKERS = ["SPX", "SPY", "QQQ", "IWM", "NDX", "RUT", "VIX", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP",
           "XLU", "XLB", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "TSLA", "AVGO", "BRK"]


def pgpass_path() -> Path:
    return Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf" if os.name == "nt" else Path.home() / ".pgpass"


def wrds_username() -> str | None:
    if os.environ.get("WRDS_USERNAME"):
        return os.environ["WRDS_USERNAME"]
    p = pgpass_path()
    if not p.exists():
        return None
    prefix = f"{HOST}:{PORT}:{DB}:"
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith(prefix):
            return ln[len(prefix):].split(":", 1)[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true", help="also count SPX/SPY option rows per year (slower)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    user = wrds_username()
    if not user:
        raise SystemExit("No WRDS login in the PostgreSQL password file.")
    eng = create_engine(f"postgresql+psycopg2://{user}@{HOST}:{PORT}/{DB}",
                        connect_args={"sslmode": "require", "connect_timeout": 120})
    out: dict = {}
    with eng.connect() as c:
        t0 = time.time()
        sch = pd.read_sql(text(
            "select table_schema, count(*) as n_tables from information_schema.tables "
            "where table_schema ilike any (array['%option%','%cboe%','%vix%','%fut%','%ivy%','%frb%','%fred%','%treas%']) "
            "group by 1 order by 1"), c)
        print(f"schemas visible ({time.time() - t0:.1f}s):")
        print(sch.to_string(index=False))
        out["schemas"] = sch.to_dict("records")

        tabs = pd.read_sql(text("select table_name from information_schema.tables where table_schema = 'optionm' "
                                "order by 1"), c)["table_name"].tolist()
        out["optionm_tables_n"] = len(tabs)
        base = sorted({t.rstrip("0123456789") for t in tabs})
        print(f"\noptionm: {len(tabs)} tables; families: {base}")
        years = {}
        for t in tabs:
            stem = t.rstrip("0123456789")
            suf = t[len(stem):]
            if suf.isdigit() and len(suf) == 4:
                years.setdefault(stem, []).append(int(suf))
        for k, v in sorted(years.items()):
            print(f"  {k}: {min(v)}..{max(v)} ({len(v)} years)")
        out["optionm_yearly"] = {k: [min(v), max(v), len(v)] for k, v in years.items()}

        if "secnmd" in tabs:
            q = text("select * from optionm.secnmd where ticker = any(:t) order by ticker, effect_date")
            sec = pd.read_sql(q, c, params={"t": TICKERS})
            print("\nsecnmd rows for planned tickers:")
            print(sec.to_string(index=False, max_colwidth=40))
            out["secnmd"] = sec.astype(str).to_dict("records")

        if "opprcd" in years:
            last = max(years["opprcd"])
            for tbl in (f"opprcd{last}", f"secprd{last}" if "secprd" in years else None):
                if tbl is None:
                    continue
                cols = pd.read_sql(text("select column_name, data_type from information_schema.columns "
                                        "where table_schema='optionm' and table_name=:t order by ordinal_position"),
                                   c, params={"t": tbl})
                print(f"\ncolumns of optionm.{tbl}: " + ", ".join(cols["column_name"]))
                out[f"columns_{tbl}"] = cols["column_name"].tolist()
            t0 = time.time()
            mx = pd.read_sql(text(f"select max(date) as d from optionm.secprd{last} where secid = 108105"), c)
            print(f"\nlast secprd date for secid 108105 in {last}: {mx['d'].iloc[0]} ({time.time() - t0:.1f}s)")
            out["last_date"] = str(mx["d"].iloc[0])

        for extra in ("zerocd", "idxdvd", "distrd", "stdopd", "vsurfd"):
            hits = [t for t in tabs if t.startswith(extra)]
            out[f"has_{extra}"] = hits[:3] + (["..."] if len(hits) > 3 else [])

        if args.counts and "opprcd" in years:
            rows = []
            for y in sorted(years["opprcd"]):
                for secid, name in ((108105, "SPX"), (109820, "SPY")):
                    t0 = time.time()
                    n = pd.read_sql(text(f"select count(*) as n from optionm.opprcd{y} where secid = :s"), c,
                                    params={"s": secid})["n"].iloc[0]
                    rows.append({"year": y, "secid": secid, "name": name, "rows": int(n), "sec": round(time.time() - t0, 1)})
                    print(f"  opprcd{y} {name}: {int(n):,} rows ({time.time() - t0:.1f}s)", flush=True)
            out["counts"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
