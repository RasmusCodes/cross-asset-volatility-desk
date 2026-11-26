"""Third read-only probe: VIX futures in Datastream, the Cboe option tables' span and ids. Metadata/counts only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrds_probe_aux import q, show  # noqa: E402


def main() -> None:
    show("Datastream volatility futures contract families", q(
        "select contrcode, clscode, min(contrname) as name, min(substr(dsmnem,1,3)) as mnem, count(*) as n_contracts, "
        "min(startdate) as first_start, max(lasttrddate) as last_trade, min(exchtickersymb) as tick "
        "from tr_ds_fut.wrds_contract_info where contrname ilike '%volatil%' or contrname ilike '%VIX%' "
        "group by contrcode, clscode order by n_contracts desc limit 30"))
    show("cboe.optprice_2026 span", q("select min(date_) as d0, max(date_) as d1, count(distinct date_) as days "
                                      "from cboe.optprice_2026"))
    show("cboe.optprice_2025 span", q("select min(date_) as d0, max(date_) as d1, count(distinct date_) as days "
                                      "from cboe.optprice_2025"))
    show("cboe.ivlisted_2026 span", q("select min(date_) as d0, max(date_) as d1 from cboe.ivlisted_2026"))
    show("cboe.eqmaster planned tickers", q(
        "select eqid, ticker, issuer, issue, isstype, startdate, enddate from cboe.eqmaster "
        "where ticker in ('SPX','SPY','QQQ','IWM','VIX','NDX','RUT','AAPL','MSFT','NVDA') order by ticker, startdate"))
    show("cboe.eqprice span", q("select min(date_) as d0, max(date_) as d1 from cboe.eqprice"))
    show("cboe.ivrate sample", q("select rateid, min(date_) as d0, max(date_) as d1, count(*) as n from cboe.ivrate "
                                 "group by rateid order by rateid limit 20"))


if __name__ == "__main__":
    main()
