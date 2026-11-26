"""Row counts for the planned pulls (no data rows are fetched). Writes JSON lines to --out as it goes, so a stopped run
keeps what it counted.

    python scripts/wrds_count.py --out counts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrds_probe import DB, HOST, PORT, wrds_username  # noqa: E402

SECIDS = {108105: "SPX", 109820: "SPY", 107899: "QQQ", 106445: "IWM"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=1996)
    ap.add_argument("--last", type=int, default=2025)
    args = ap.parse_args()
    eng = create_engine(f"postgresql+psycopg2://{wrds_username()}@{HOST}:{PORT}/{DB}",
                        connect_args={"sslmode": "require", "connect_timeout": 120,
                                      "options": "-c statement_timeout=600000"})
    out = Path(args.out)

    def emit(rec: dict) -> None:
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(rec, flush=True)

    def run(label: str, sql: str) -> None:
        t0 = time.time()
        try:
            with eng.connect() as c:
                df = pd.read_sql(text(sql), c)
            emit({"label": label, "sec": round(time.time() - t0, 1), "rows": df.to_dict("records")})
        except Exception as e:  # noqa: BLE001 - keep counting the rest
            emit({"label": label, "sec": round(time.time() - t0, 1), "error": str(e).splitlines()[0][:300]})

    ids = ",".join(str(s) for s in SECIDS)
    run("ds_vix_futures", "select count(*) as n, min(v.date_) as d0, max(v.date_) as d1, count(distinct v.futcode) as k "
        "from tr_ds_fut.wrds_fut_contract v join tr_ds_fut.wrds_contract_info i on i.futcode = v.futcode "
        "where i.contrcode = 429")
    run("cboe_spx_contracts", "select count(*) as n, min(expdate) as e0, max(expdate) as e1 from cboe.optcontract "
        "where eqid = 20341")
    run("cboe_spx_2026", "select count(*) as n, min(p.date_) as d0, max(p.date_) as d1 from cboe.optprice_2026 p "
        "where p.optid in (select optid from cboe.optcontract where eqid = 20341)")
    for y in range(args.first, args.last + 1):
        run(f"opprcd{y}", f"select secid, count(*) as n_all, "
            f"count(*) filter (where exdate - date <= 400) as n_dte400, "
            f"count(*) filter (where exdate - date <= 400 and best_bid > 0) as n_dte400_bid, "
            f"count(distinct date) as days from optionm.opprcd{y} where secid in ({ids}) group by secid")


if __name__ == "__main__":
    main()
