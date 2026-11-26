"""Readers for the raw pulls (data/raw) and the cleaned store (data/store)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from .. import config


def read_dir(rel_dir: str, years=None, columns=None, filter=None, root: Path | None = None) -> pd.DataFrame:
    """Concatenate the yearly parquet files under root/rel_dir (optionally only some years / columns / a pyarrow filter)."""
    base = Path(root or config.RAW) / rel_dir
    files = sorted(base.glob("*.parquet"))
    if years is not None:
        ys = {str(int(y)) for y in years}
        files = [f for f in files if f.stem in ys]
    if not files:
        return pd.DataFrame(columns=columns or [])
    tbl = ds.dataset([str(f) for f in files], format="parquet").to_table(columns=columns, filter=filter)
    return tbl.to_pandas()


def read_raw(rel: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(Path(config.RAW) / rel, columns=columns)


def secprd(secids=None, years=None, columns=None) -> pd.DataFrame:
    flt = None
    if secids is not None:
        flt = ds.field("secid").isin([int(s) for s in secids])
    df = read_dir("optionm/secprd", years=years, columns=columns, filter=flt)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


def close_series(secid: int) -> pd.Series:
    df = secprd([secid], columns=["secid", "date", "close"])
    return df.set_index("date")["close"].sort_index()


def div_yield(secid: int) -> pd.Series:
    df = read_raw("optionm/idxdvd.parquet")
    df = df[df["secid"] == secid]
    return df.set_index(pd.to_datetime(df["date"]))["rate"].sort_index()


def spx_raw(year: int, columns=None, filter=None) -> pd.DataFrame:
    return read_dir("optionm/opprcd/spx", years=[year], columns=columns, filter=filter)


def store_dir(name: str) -> Path:
    p = Path(config.STORE) / name
    p.mkdir(parents=True, exist_ok=True)
    return p
