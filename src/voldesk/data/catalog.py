"""What is pulled from WRDS, and how: security ids, column schemas and the SQL of every raw file.

Sources (all through the WRDS PostgreSQL service, under the user's own subscription):
  optionm    OptionMetrics IvyDB US, 1996-01-04 .. 2025-08-29: option prices/IV/Greeks (opprcdYYYY), underlying prices
             (secprdYYYY), zero curve (zerocd), index dividend yields (idxdvd), distributions (distrd), forwards
             (fwdprdYYYY), standardized ATM options (stdopdYYYY) and the standardized volatility surface (vsurfdYYYY)
  cboe       Cboe option end-of-day data, 1998 .. 2026-11-03: contracts (optcontract), prices (optprice_YYYY), IVs and
             Greeks (ivlisted_YYYY), underlying prices (eqprice), rate curve (ivrate), calendar (ovscalendar), and the
             VIX/VXO/VXN/VXD index history (cboe.cboe)
  tr_ds_fut  Datastream futures: Cboe VIX futures contracts (contract code 429) and daily settlements, 2005-12 .. 2026-11
  crsp       S&P 500 membership spells (msp500list, dsp500list_v2), monthly prices/shares (msf), names (msenames)
  wrdsapps   CRSP-OptionMetrics link (opcrsphist)
  frb        Federal Reserve daily rates (rates_daily, ends 2025-02-13 on WRDS)

Identifiers below were resolved from optionm.secnmd (scripts/wrds_probe*.py). Each ETF has extra "INTRADAY"/"NAV"
quote series under other secids (class I/N); those carry no options and are excluded.
"""
from __future__ import annotations

import datetime as dt

import pyarrow as pa

from .wrds import PullSpec

FIRST_YEAR, LAST_YEAR = 1996, 2025            # OptionMetrics yearly tables on WRDS
OM_LAST_DATE = dt.date(2025, 8, 29)

INDEX_SECIDS = {"SPX": 108105, "NDX": 102480, "RUT": 102434, "DJX": 102456, "OEX": 109764, "VIX": 117801}
ETF_SECIDS = {"SPY": 109820, "QQQ": 107899, "IWM": 106445, "DIA": 103823,
              "XLB": 110007, "XLV": 110008, "XLP": 110009, "XLY": 110010, "XLE": 110011, "XLF": 110012,
              "XLI": 110013, "XLK": 110014, "XLU": 110015,
              "TLT": 116070, "EFA": 116908, "EEM": 116959, "FXI": 122017, "GLD": 122392, "USO": 126681,
              "SLV": 126776, "HYG": 129322, "SMH": 151720}
PANEL_SECIDS = {**INDEX_SECIDS, **ETF_SECIDS}
SPX = INDEX_SECIDS["SPX"]
CHAIN_ETFS = {"SPY": 109820, "QQQ": 107899, "IWM": 106445}

CBOE_EQIDS = {"SPX": 20341, "SPY": 23112, "QQQ": 21808, "IWM": 11579, "VIX": 2551}
CBOE_EXTENSION_START = dt.date(2025, 7, 1)    # overlaps OptionMetrics for two months (cross-vendor check)
CBOE_QA_DATES = ["1998-06-17", "2000-03-15", "2002-07-24", "2005-06-15", "2008-10-15", "2010-05-12", "2013-06-19",
                 "2015-08-26", "2018-02-07", "2020-03-18", "2022-06-15", "2024-08-07", "2025-04-09"]
DS_VIX_CONTRACT = 429

# --------------------------------------------------------------------------------------------- schemas
I32, I64, F64, S, D = pa.int32(), pa.int64(), pa.float64(), pa.string(), pa.date32()

OPPRCD = {"secid": I32, "date": D, "exdate": D, "last_date": D, "cp_flag": S, "strike_price": F64, "best_bid": F64,
          "best_offer": F64, "volume": I64, "open_interest": I64, "impl_volatility": F64, "delta": F64, "gamma": F64,
          "vega": F64, "theta": F64, "optionid": I64, "cfadj": F64, "am_settlement": pa.int8(), "contract_size": F64,
          "ss_flag": S, "forward_price": F64, "expiry_indicator": S, "root": S, "suffix": S}
SECPRD = {"secid": I32, "date": D, "open": F64, "high": F64, "low": F64, "close": F64, "volume": F64, "return": F64,
          "cfadj": F64, "cfret": F64, "shrout": F64}
STDOPD = {"secid": I32, "date": D, "days": pa.int16(), "forward_price": F64, "strike_price": F64, "premium": F64,
          "impl_volatility": F64, "delta": F64, "gamma": F64, "theta": F64, "vega": F64, "cp_flag": S}
VSURFD = {"secid": I32, "date": D, "days": pa.int16(), "delta": pa.int16(), "impl_volatility": F64,
          "impl_strike": F64, "impl_premium": F64, "dispersion": F64, "cp_flag": S}
FWDPRD = {"secid": I32, "date": D, "expiration": D, "amsettlement": pa.int8(), "forwardprice": F64}
ZEROCD = {"date": D, "days": pa.int32(), "rate": F64}
IDXDVD = {"secid": I32, "date": D, "rate": F64}
DISTRD = {"secid": I32, "record_date": D, "seq_num": pa.int32(), "ex_date": D, "amount": F64, "adj_factor": F64,
          "declare_date": D, "payment_date": D, "link_secid": F64, "distr_type": S, "frequency": S, "currency": S,
          "approx_flag": S, "cancel_flag": S, "liquid_flag": S}
SECNMD = {"secid": I32, "effect_date": D, "cusip": S, "ticker": S, "class": S, "issuer": S, "issue": S, "sic": S}
MSP500 = {"permno": I32, "start": D, "ending": D}
DSP500V2 = {"permno": I32, "indno": I64, "mbrstartdt": D, "mbrenddt": D, "mbrflg": S, "indfam": S}
MSF = {"permno": I32, "date": D, "prc": F64, "shrout": F64, "ret": F64, "cfacpr": F64, "cfacshr": F64}
MSENAMES = {"permno": I32, "namedt": D, "nameendt": D, "ticker": S, "comnam": S, "shrcd": pa.int16(),
            "exchcd": pa.int16(), "siccd": S}
OPCRSP = {"secid": I32, "sdate": D, "edate": D, "permno": I32, "score": pa.int16()}
CBOE_VIX = {"date": D, **{f"{s}{x}": F64 for s in ("vix", "vxo", "vxn", "vxd") for x in ("o", "h", "l", "")}}
CBOE_VIX_ORDER = ["date", "vixo", "vixh", "vixl", "vix", "vxoo", "vxoh", "vxol", "vxo", "vxno", "vxnh", "vxnl", "vxn",
                  "vxdo", "vxdh", "vxdl", "vxd"]
OPTCONTRACT = {"optid": I64, "putcall": S, "strike": F64, "expdate": D, "seriesid": I64, "eqid": I64,
               "opraticker": S, "ositicker": S, "opraroot": S, "opttype": S, "exercise": S, "udltype": S,
               "multiplier": F64, "units": F64, "cash": F64, "startdate": D, "enddate": D, "corpaction": S,
               "accexpdate": D}
OPTPRICE = {"optid": I64, "date_": D, "exch": S, "close_": F64, "high": F64, "low": F64, "open_": F64, "volume": F64,
            "bid": F64, "ask": F64, "openint": F64}
IVLISTED = {"optid": I64, "date_": D, "ivbid": F64, "ivask": F64, "ivmid": F64, "delta": F64, "gamma": F64,
            "theta": F64, "vega": F64, "rho": F64, "confidence": F64, "filterbits": F64}
EQPRICE = {"eqid": I64, "date_": D, "close_": F64, "high": F64, "low": F64, "open_": F64, "volume": F64, "bid": F64,
           "ask": F64, "totrtn": F64}
EQMASTER = {"eqid": I64, "cusip": S, "issuer": S, "issue": S, "ticker": S, "siccode": S, "isstype": S,
            "suppisstype": S, "startdate": D, "enddate": D}
IVRATE = {"rateid": pa.int16(), "date_": D, "value_": F64}
IVRATEMAP = {"rateid": pa.int16(), "days": F64}
OVSCAL = {"date_": D, "caltype": pa.int16(), "details": S}
DS_INFO = {"futcode": I64, "contrcode": I64, "clscode": I64, "dsmnem": S, "contrname": S, "contrdate": S,
           "isocurrcode": S, "startdate": D, "lasttrddate": D, "sttlmntdate": D, "exchtickersymb": S}
DS_VAL = {"futcode": I64, "date_": D, "open_": F64, "high": F64, "low": F64, "volume": F64, "settlement": F64,
          "openinterest": F64}
FRB_COLS = ["date", "dgs1mo", "dgs3mo", "dgs6mo", "dgs1", "dgs2", "dgs5", "dgs10", "dtb3", "dff", "effr", "sofr"]
FRB = {c: (D if c == "date" else F64) for c in FRB_COLS}


def cols(schema: dict) -> str:
    return ", ".join(f'"{c}"' if c in ("return", "class") else c for c in schema)


def ids(d) -> str:
    vals = d.values() if isinstance(d, dict) else d
    return ",".join(str(int(v)) for v in sorted(set(vals)))


# --------------------------------------------------------------------------------------------- stages
def reference_specs() -> list[PullSpec]:
    """Small reference tables: names, curve, dividends, distributions, CRSP membership, links, VIX family, rates,
    VIX futures, Cboe contracts/calendar/rates."""
    pid = ids(PANEL_SECIDS)
    out = [
        PullSpec("optionm/zerocd.parquet", f"select {cols(ZEROCD)} from optionm.zerocd", ZEROCD, "reference"),
        PullSpec("optionm/idxdvd.parquet", f"select {cols(IDXDVD)} from optionm.idxdvd where secid in "
                 f"({ids(INDEX_SECIDS)})", IDXDVD, "reference"),
        PullSpec("optionm/secnmd_panel.parquet", f"select {cols(SECNMD)} from optionm.secnmd where secid in ({pid})",
                 SECNMD, "reference"),
        PullSpec("optionm/distrd_panel.parquet", f"select {cols(DISTRD)} from optionm.distrd where secid in ({pid})",
                 DISTRD, "reference"),
        PullSpec("crsp/msp500list.parquet", f"select {cols(MSP500)} from crsp.msp500list", MSP500, "reference"),
        PullSpec("crsp/dsp500list_v2.parquet", f"select {cols(DSP500V2)} from crsp.dsp500list_v2", DSP500V2,
                 "reference"),
        PullSpec("crsp/msf_members.parquet", f"select {cols(MSF)} from crsp.msf where date >= '1995-01-01' and "
                 "permno in (select distinct permno from crsp.msp500list where ending >= '1995-01-01')", MSF,
                 "reference"),
        PullSpec("crsp/msenames_members.parquet", f"select {cols(MSENAMES)} from crsp.msenames where permno in "
                 "(select distinct permno from crsp.msp500list where ending >= '1995-01-01')", MSENAMES,
                 "reference"),
        PullSpec("wrdsapps/opcrsphist.parquet", f"select {cols(OPCRSP)} from wrdsapps_link_crsp_optionm.opcrsphist",
                 OPCRSP, "reference"),
        PullSpec("cboe/vix_family.parquet", f"select {', '.join(CBOE_VIX_ORDER)} from cboe.cboe",
                 {c: CBOE_VIX[c] for c in CBOE_VIX_ORDER}, "reference"),
        PullSpec("cboe/ivrate.parquet", f"select {cols(IVRATE)} from cboe.ivrate", IVRATE, "reference"),
        PullSpec("cboe/ivratemap.parquet", f"select {cols(IVRATEMAP)} from cboe.ivratemap", IVRATEMAP, "reference"),
        PullSpec("cboe/ovscalendar.parquet", f"select {cols(OVSCAL)} from cboe.ovscalendar", OVSCAL, "reference"),
        PullSpec("cboe/eqmaster_panel.parquet", f"select {cols(EQMASTER)} from cboe.eqmaster where eqid in "
                 f"({ids(CBOE_EQIDS)})", EQMASTER, "reference"),
        PullSpec("cboe/eqprice_panel.parquet", f"select {cols(EQPRICE)} from cboe.eqprice where eqid in "
                 f"({ids(CBOE_EQIDS)})", EQPRICE, "reference"),
        PullSpec("cboe/optcontract_spx.parquet", f"select {cols(OPTCONTRACT)} from cboe.optcontract where eqid = "
                 f"{CBOE_EQIDS['SPX']}", OPTCONTRACT, "reference"),
        PullSpec("datastream/vix_contracts.parquet", f"select {cols(DS_INFO)} from tr_ds_fut.wrds_contract_info "
                 f"where contrcode = {DS_VIX_CONTRACT}", DS_INFO, "reference"),
        PullSpec("datastream/vix_settlements.parquet", f"select {', '.join('v.' + c for c in DS_VAL)} from "
                 "tr_ds_fut.wrds_fut_contract v where v.futcode in (select futcode from tr_ds_fut.wrds_contract_info "
                 f"where contrcode = {DS_VIX_CONTRACT})", DS_VAL, "reference"),
        PullSpec("frb/rates_daily.parquet", f"select {', '.join(FRB_COLS)} from frb.rates_daily where date >= "
                 "'1990-01-01'", FRB, "reference"),
    ]
    return out


def panel_specs(single_name_secids: list[int], years=range(FIRST_YEAR, LAST_YEAR + 1)) -> list[PullSpec]:
    """Daily underlying prices, forwards and standardized surfaces for the cross-asset panel and the point-in-time
    single-name universe. Panel: stdopd all tenors; vsurfd 30 and 91 days, out-of-the-money side (puts -10..-50,
    calls 10..50). Single names: stdopd 30/91 days; vsurfd 30 days at -25/-50/50/25 delta."""
    pid = ids(PANEL_SECIDS)
    sid = ids(single_name_secids) if single_name_secids else ""
    allid = ids(list(PANEL_SECIDS.values()) + list(single_name_secids))
    out = []
    for y in years:
        out.append(PullSpec(f"optionm/secprd/{y}.parquet", f"select {cols(SECPRD)} from optionm.secprd{y} where secid "
                            f"in ({allid})", SECPRD, "panel"))
        out.append(PullSpec(f"optionm/fwdprd/{y}.parquet", f"select {cols(FWDPRD)} from optionm.fwdprd{y} where secid "
                            f"in ({ids({**INDEX_SECIDS, **CHAIN_ETFS})})", FWDPRD, "panel"))
        out.append(PullSpec(f"optionm/stdopd_panel/{y}.parquet", f"select {cols(STDOPD)} from optionm.stdopd{y} where "
                            f"secid in ({pid})", STDOPD, "panel"))
        out.append(PullSpec(f"optionm/vsurfd_panel/{y}.parquet", f"select {cols(VSURFD)} from optionm.vsurfd{y} where "
                            f"secid in ({pid}) and days in (30, 91) and abs(delta) <= 50", VSURFD, "panel"))
        if sid:
            out.append(PullSpec(f"optionm/stdopd_names/{y}.parquet", f"select {cols(STDOPD)} from optionm.stdopd{y} "
                                f"where secid in ({sid}) and days in (30, 91)", STDOPD, "panel"))
            out.append(PullSpec(f"optionm/vsurfd_names/{y}.parquet", f"select {cols(VSURFD)} from optionm.vsurfd{y} "
                                f"where secid in ({sid}) and days = 30 and delta in (-25, -50, 50, 25)", VSURFD,
                                "panel"))
    return out


def spx_chain_specs(years=range(FIRST_YEAR, LAST_YEAR + 1)) -> list[PullSpec]:
    """Every SPX option row (all roots, expiries and strikes), one file per year."""
    return [PullSpec(f"optionm/opprcd/spx/{y}.parquet", f"select {cols(OPPRCD)} from optionm.opprcd{y} where secid = "
                     f"{SPX}", OPPRCD, "spx") for y in years]


def etf_chain_specs(years=range(FIRST_YEAR, LAST_YEAR + 1)) -> list[PullSpec]:
    """SPY/QQQ/IWM: (a) full chains on each month's last SPX trading day; (b) daily rows of standard monthly expiries
    with at most 75 days to expiry and strikes within 70%-140% of that day's close (backtest marks)."""
    eid = ids(CHAIN_ETFS)
    oc = ", ".join(f"o.{c}" for c in OPPRCD)
    out = []
    for y in years:
        out.append(PullSpec(f"optionm/opprcd/etf_monthend/{y}.parquet",
                            f"select {cols(OPPRCD)} from optionm.opprcd{y} where secid in ({eid}) and date in "
                            f"(select max(date) from optionm.secprd{y} where secid = {SPX} "
                            "group by date_trunc('month', date))", OPPRCD, "etf"))
        out.append(PullSpec(f"optionm/opprcd/etf_ntm/{y}.parquet",
                            f"select {oc} from optionm.opprcd{y} o join optionm.secprd{y} s on s.secid = o.secid and "
                            f"s.date = o.date where o.secid in ({eid}) and o.expiry_indicator is null and "
                            "o.exdate - o.date <= 75 and o.strike_price / 1000.0 between 0.7 * s.close and "
                            "1.4 * s.close", OPPRCD, "etf"))
    return out


def cboe_specs() -> list[PullSpec]:
    """Cboe SPX prices and IVs from 2025-07-01 (extends OptionMetrics to 2026-11-03), plus a 13-date sample over
    1998-2025 for the cross-vendor check."""
    spx = f"(select optid from cboe.optcontract where eqid = {CBOE_EQIDS['SPX']})"
    start = CBOE_EXTENSION_START.isoformat()
    out = []
    for y in (2025, 2026):
        cond = f"optid in {spx}" + (f" and date_ >= '{start}'" if y == 2025 else "")
        out.append(PullSpec(f"cboe/optprice_spx/{y}.parquet", f"select {cols(OPTPRICE)} from cboe.optprice_{y} where "
                            f"{cond}", OPTPRICE, "cboe"))
        out.append(PullSpec(f"cboe/ivlisted_spx/{y}.parquet", f"select {cols(IVLISTED)} from cboe.ivlisted_{y} where "
                            f"{cond}", IVLISTED, "cboe"))
    by_year: dict[int, list[str]] = {}
    for d in CBOE_QA_DATES:
        by_year.setdefault(int(d[:4]), []).append(d)
    for y, ds in sorted(by_year.items()):
        dl = ", ".join(f"'{d}'" for d in ds)
        out.append(PullSpec(f"cboe/qa_optprice_spx/{y}.parquet", f"select {cols(OPTPRICE)} from cboe.optprice_{y} "
                            f"where date_ in ({dl}) and optid in {spx}", OPTPRICE, "cboe"))
        out.append(PullSpec(f"cboe/qa_ivlisted_spx/{y}.parquet", f"select {cols(IVLISTED)} from cboe.ivlisted_{y} "
                            f"where date_ in ({dl}) and optid in {spx}", IVLISTED, "cboe"))
    return out
