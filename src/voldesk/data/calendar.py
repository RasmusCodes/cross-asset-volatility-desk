"""Expiration calendar: settlement instants, last trading days, expiry types, and the year fractions used everywhere.

Conventions
  valuation instant   16:00 ET on the trade date (the index close; option quotes are end-of-day closing quotes)
  AM-settled options  settle at the opening print on the settlement date (09:30 ET). OptionMetrics records standard
                      monthly expirations before February 2015 under the Saturday after the third Friday; the
                      settlement date is then the Friday. The last trading day is the business day before settlement.
  PM-settled options  settle at the 16:00 ET close of the expiration date, which is also the last trading day.
  Time                ACT/365F in years between instants: T = seconds / (365 * 86400).
Expiry types (OptionMetrics' expiry_indicator: blank = standard third-Friday cycle, 'w' weekly, 'd' daily, 'm' month-end)
  standard_am  third-Friday (or holiday-shifted) AM-settled SPX/standard contracts
  standard_pm  PM-settled contracts expiring on the standard date (SPXW on third Fridays, ETF options)
  weekly / daily / monthend  non-standard expirations
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_SECONDS = 365.0 * 86400.0
OPEN = pd.Timedelta(hours=9, minutes=30)
CLOSE = pd.Timedelta(hours=16)


def settlement_date(exdate, am_settlement=None) -> pd.Series:
    """The exercise/settlement date: OptionMetrics' Saturday expirations (standard cycle before Feb 2015) settle on
    the Friday before (AM print for SPX, Friday close for PM/American contracts); any other date is its own."""
    ex = pd.to_datetime(pd.Series(exdate)).reset_index(drop=True)
    return ex.where(ex.dt.dayofweek != 5, ex - pd.Timedelta(days=1))


def settlement_instant(exdate, am_settlement) -> pd.Series:
    """Naive ET timestamps: 09:30 on the settlement date for AM-settled contracts, 16:00 for PM-settled."""
    sd = settlement_date(exdate, am_settlement)
    am = pd.Series(am_settlement).reset_index(drop=True).fillna(0).astype(int).to_numpy() == 1
    return sd + pd.to_timedelta(np.where(am, 9.5, 16.0), unit="h")


def valuation_instant(dates) -> pd.Series:
    return pd.to_datetime(pd.Series(dates)).reset_index(drop=True) + CLOSE


def year_frac(start, end) -> np.ndarray:
    s, e = pd.to_datetime(pd.Series(start)), pd.to_datetime(pd.Series(end))
    return ((e.to_numpy() - s.to_numpy()) / np.timedelta64(1, "s")).astype(float) / YEAR_SECONDS


def time_to_expiry(dates, exdate, am_settlement) -> np.ndarray:
    return year_frac(valuation_instant(dates), settlement_instant(exdate, am_settlement))


def expiry_type(am_settlement, expiry_indicator) -> np.ndarray:
    am = pd.Series(am_settlement).fillna(0).astype(int).to_numpy() == 1
    ind = pd.Series(expiry_indicator).fillna("").astype(str).str.strip().str.lower().to_numpy()
    out = np.where(ind == "w", "weekly", np.where(ind == "d", "daily", np.where(ind == "m", "monthend",
                   np.where(am, "standard_am", "standard_pm"))))
    return out


def last_trading_day(exdate, am_settlement, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Last trading day: the trading day before settlement (AM) or the settlement date itself (PM)."""
    sd = settlement_date(exdate, am_settlement)
    am = pd.Series(am_settlement).reset_index(drop=True).fillna(0).astype(int) == 1
    td = pd.DatetimeIndex(sorted(trading_days))
    pos_pm = td.searchsorted(sd.to_numpy(), side="right") - 1          # last trading day <= settlement date
    pos_am = td.searchsorted(sd.to_numpy(), side="left") - 1           # last trading day < settlement date
    pos = np.where(am, pos_am, pos_pm)
    pos = np.clip(pos, 0, len(td) - 1)
    return pd.Series(td[pos])


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 14)
