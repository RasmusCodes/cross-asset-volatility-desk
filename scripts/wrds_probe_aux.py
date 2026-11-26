"""Second read-only probe: auxiliary WRDS libraries that could supply VIX futures, Cboe indices/options and interest
rates. Metadata, spans and a handful of identifying rows only (no bulk data)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrds_probe import DB, HOST, PORT, wrds_username  # noqa: E402

ENG = create_engine(f"postgresql+psycopg2://{wrds_username()}@{HOST}:{PORT}/{DB}",
                    connect_args={"sslmode": "require", "connect_timeout": 120})


def q(sql: str, **params) -> pd.DataFrame | str:
    """Each query on its own connection, so one failure does not abort the rest."""
    try:
        with ENG.connect() as c:
            return pd.read_sql(text(sql), c, params=params)
    except Exception as e:  # noqa: BLE001 - probe: report and continue
        return "FAILED: " + str(e).splitlines()[0][:300]


def cols(schema: str, table: str) -> list[str] | str:
    r = q("select column_name from information_schema.columns where table_schema=:s and table_name=:t "
          "order by ordinal_position", s=schema, t=table)
    return r if isinstance(r, str) else r["column_name"].tolist()


def show(title: str, r) -> None:
    print(f"\n{title}")
    print(r if isinstance(r, str) else r.to_string(index=False, max_colwidth=45))


def main() -> None:
    for t in ("cboe", "eqprice", "optprice_2026", "ivlisted_2026", "optcontract", "eqmaster", "ivrate", "ovscalendar"):
        print(f"cboe.{t}: {cols('cboe', t)}")
    show("cboe.cboe span", q("select min(date) as d0, max(date) as d1, count(*) as n from cboe.cboe"))
    show("cboe.optprice_2026 span", q("select min(date) as d0, max(date) as d1 from cboe.optprice_2026"))
    show("cboe.optprice_2025 span", q("select min(date) as d0, max(date) as d1 from cboe.optprice_2025"))
    for t in ("wrds_fut_contract", "wrds_fut_series", "wrds_contract_info", "wrds_cseries_info", "dsfutcontrval",
              "dsfutcontrinfo"):
        print(f"tr_ds_fut.{t}: {cols('tr_ds_fut', t)}")
    show("Datastream contracts like VIX", q(
        "select * from tr_ds_fut.wrds_contract_info where contrname ilike '%VIX%' or contrname ilike '%VOLATILITY%' "
        "limit 25"))
    show("optionm.futures like VIX", q(
        "select * from optionm.futures where description ilike '%VIX%' or description ilike '%VOLATILITY%' limit 25"))
    for t in ("zerocd", "idxdvd", "distrd", "stdopd2025", "vsurfd2025", "hvold2025", "fwdprd2025", "borrate2025"):
        print(f"optionm.{t}: {cols('optionm', t)}")
    show("zerocd span", q("select min(date) as d0, max(date) as d1, count(*) as n from optionm.zerocd"))
    show("frb.rates_daily span", q("select min(date) as d0, max(date) as d1, count(*) as n from frb.rates_daily"))


if __name__ == "__main__":
    main()
