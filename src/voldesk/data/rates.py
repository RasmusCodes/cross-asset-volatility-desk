"""Discount curves.

OptionMetrics zero curve (optionm.zerocd, 1996-01-02 .. 2025-08-29): continuously compounded zero rates in percent at
fixed day counts (10 .. 730 days and beyond in early years). Interpolation is linear in r(T) T (i.e. piecewise-flat
instantaneous forward rates), with the shortest rate held flat towards T = 0 and the longest rate flat beyond the last
node. Cboe's curve (cboe.ivrate, rate ids mapped to 1 .. 1800 days, decimal) extends to 2026-11-03 and is used only
for dates after the OptionMetrics curve ends, with the same interpolation.

For a date without a curve row (holidays in one source), the previous available date is used (at most 5 days back;
beyond that the rate is NaN and the caller must drop the row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ZeroCurve:
    def __init__(self, df: pd.DataFrame, date_col: str = "date", days_col: str = "days", rate_col: str = "rate",
                 scale: float = 0.01, label: str = ""):
        d = df[[date_col, days_col, rate_col]].dropna().copy()
        d.columns = ["date", "days", "rate"]
        d["date"] = pd.to_datetime(d["date"])
        d["rate"] = d["rate"].astype(float) * scale
        d = d.sort_values(["date", "days"])
        self.label = label
        self.dates = pd.DatetimeIndex(d["date"].unique())
        self._nodes = {pd.Timestamp(k): (g["days"].to_numpy(float) / 365.0, g["rate"].to_numpy(float))
                       for k, g in d.groupby("date")}

    @property
    def first(self) -> pd.Timestamp:
        return self.dates[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.dates[-1]

    def _node_date(self, date) -> pd.Timestamp | None:
        date = pd.Timestamp(date)
        i = self.dates.searchsorted(date, side="right") - 1
        if i < 0 or (date - self.dates[i]).days > 5:
            return None
        return self.dates[i]

    def rate(self, date, T) -> np.ndarray:
        """Continuously compounded zero rate for maturities T (years) on `date`."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        nd = self._node_date(date)
        if nd is None:
            return np.full(T.shape, np.nan)
        t, r = self._nodes[nd]
        rt = np.interp(T, t, r * t, left=np.nan, right=np.nan)
        out = np.where(T > 0, rt / np.where(T > 0, T, 1.0), r[0])
        out = np.where(T <= t[0], r[0], out)
        out = np.where(T >= t[-1], r[-1], out)
        return out

    def discount(self, date, T) -> np.ndarray:
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.exp(-self.rate(date, T) * T)


class CombinedCurve:
    """OptionMetrics curve up to its last date, Cboe's afterwards."""

    def __init__(self, primary: ZeroCurve, extension: ZeroCurve | None = None):
        self.primary, self.extension = primary, extension

    def _pick(self, date) -> ZeroCurve:
        if self.extension is not None and pd.Timestamp(date) > self.primary.last:
            return self.extension
        return self.primary

    def rate(self, date, T):
        return self._pick(date).rate(date, T)

    def discount(self, date, T):
        return self._pick(date).discount(date, T)

    def source(self, date) -> str:
        return self._pick(date).label


def load_curve(raw_root) -> CombinedCurve:
    from pathlib import Path
    raw_root = Path(raw_root)
    om = ZeroCurve(pd.read_parquet(raw_root / "optionm/zerocd.parquet"), label="optionm.zerocd")
    ext = None
    p, m = raw_root / "cboe/ivrate.parquet", raw_root / "cboe/ivratemap.parquet"
    if p.exists() and m.exists():
        iv = pd.read_parquet(p).merge(pd.read_parquet(m), on="rateid")
        iv = iv[pd.to_datetime(iv["date_"]) > om.last]
        if len(iv):
            ext = ZeroCurve(iv, date_col="date_", days_col="days", rate_col="value_", scale=1.0, label="cboe.ivrate")
    return CombinedCurve(om, ext)
