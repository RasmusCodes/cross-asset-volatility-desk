"""Fourth read-only probe: link table, CRSP S&P list span, expiry indicators, ivrate map. Metadata/counts only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrds_probe_aux import cols, q, show  # noqa: E402


def main() -> None:
    t = q("select table_name from information_schema.tables where table_schema='wrdsapps_link_crsp_optionm'")
    print("link tables:", t if isinstance(t, str) else t["table_name"].tolist())
    if not isinstance(t, str):
        for name in t["table_name"]:
            print(f"  {name}: {cols('wrdsapps_link_crsp_optionm', name)}")
    show("crsp.msp500list span", q("select min(start) as s0, max(ending) as e1, count(*) as n from crsp.msp500list"))
    print("crsp.dsp500list_v2 cols", cols("crsp", "dsp500list_v2"))
    show("SPX expiry_indicator 2024", q("select expiry_indicator, am_settlement, count(*) as n from optionm.opprcd2024 "
                                        "where secid = 108105 group by 1, 2 order by 3 desc"))
    show("SPX expiry_indicator 2005", q("select expiry_indicator, am_settlement, count(*) as n from optionm.opprcd2005 "
                                        "where secid = 108105 group by 1, 2 order by 3 desc"))
    show("SPX exdate weekday 2010", q("select extract(dow from exdate) as dow, am_settlement, count(*) as n from "
                                      "optionm.opprcd2010 where secid = 108105 group by 1, 2 order by 3 desc"))
    show("cboe.ivratemap", q("select * from cboe.ivratemap"))
    show("cboe.ovscalendar sample", q("select caltype, min(date_) as d0, max(date_) as d1, count(*) as n, "
                                      "min(details) as ex from cboe.ovscalendar group by caltype"))
    show("Cboe SPX roots", q("select opraroot, exercise, count(*) as n, min(expdate) as e0, max(expdate) as e1 "
                             "from cboe.optcontract where eqid = 20341 group by 1, 2 order by 3 desc limit 20"))
    show("vsurfd days/deltas", q("select days, count(distinct delta) as n_delta, min(delta) as dmin, max(delta) as dmax "
                                 "from optionm.vsurfd2024 where secid = 108105 and date = '2024-06-03' group by days "
                                 "order by days"))
    show("stdopd days", q("select days, count(*) as n from optionm.stdopd2024 where secid = 108105 and "
                          "date = '2024-06-03' group by days order by days"))


if __name__ == "__main__":
    main()
