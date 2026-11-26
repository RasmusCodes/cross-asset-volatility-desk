"""Figures for the reports (reports/figures/*.png), from results/ and the caches. Categorical colors follow a fixed
slot order (blue, orange, aqua, yellow, magenta, green, violet); one y-axis per chart.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.backtest.metrics import unit_returns, with_cost_model  # noqa: E402
from voldesk.strategies.costs import BASE  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_ORDER = ["svi", "sabr", "essvi", "ssvi", "heston"]
MODEL_NAME = {"svi": "SVI", "sabr": "SABR (β=1)", "essvi": "eSSVI", "ssvi": "SSVI", "heston": "Heston"}
FIG = config.FIGURES

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
                     "text.parse_math": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def model_figures():
    d = pd.read_csv(config.RESULTS / "surfaces" / "monthend_dates.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(MODEL_ORDER):
        s = d[d["model"] == m].set_index("date")["rmse_pooled"].sort_index()
        ax.plot(s.index, s.rolling(12, min_periods=6).median(), color=SLOTS[i], lw=1.8, label=MODEL_NAME[m])
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))      # math text is off: format log ticks by hand
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("fit RMSE, vol points (log scale)")
    ax.set_title("SPX surface fit error by model, month-ends 1996–2026 (12-month rolling median)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "model_rmse_time.png")
    s = pd.read_csv(config.RESULTS / "surfaces" / "monthend_slices.csv")
    s["bucket"] = pd.cut(s["T"] * 365, [0, 30, 91, 365, 1200], labels=["≤30 days", "31–91 days", "92–365 days", ">1 year"])
    p = s.pivot_table(index="bucket", columns="model", values="rmse", aggfunc="median", observed=True)[MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(p.index))
    w = 0.16
    for i, m in enumerate(MODEL_ORDER):
        ax.bar(x + (i - 2) * w, p[m], width=w - 0.02, color=SLOTS[i], label=MODEL_NAME[m])
    ax.set_xticks(x, p.index)
    ax.set_ylabel("median RMSE, vol points")
    ax.set_title("Fit error by maturity: every model is worst at the short end")
    ax.legend(ncol=5, loc="upper right")
    save(fig, "model_rmse_bucket.png")


KEY = ["short_straddle", "short_straddle_dh", "short_put_25_dh", "put_spread_25_10", "iron_condor_25_10",
       "short_var_strip_dh", "long_put_10"]


def backtest_figures():
    src = config.CACHE / "backtests" / "run"
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, n in enumerate(KEY):
        d = with_cost_model(pd.read_parquet(src / f"{n}_daily.parquet"), BASE)
        cum = unit_returns(d).cumsum()
        ax.plot(cum.index, cum, color=SLOTS[i], lw=1.4, label=n)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("cumulative P&L per $1 of index notional (base costs)")
    ax.set_title("SPX option strategies, monthly cycle, 1996 – Nov 2026")
    ax.legend(ncol=2, loc="upper left", fontsize=8.5)
    save(fig, "backtest_equity.png")
    sm = {c: pd.read_csv(config.RESULTS / "backtests" / f"summary_{c}.csv").set_index("strategy")["sharpe"]
          for c in ("frictionless", "base", "stress")}
    t = pd.DataFrame(sm).sort_values("frictionless")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(t))
    for i, c in enumerate(("frictionless", "base", "stress")):
        ax.scatter(t[c], y, color=SLOTS[i], s=36, zorder=3, label=c)
    for j in range(len(t)):
        ax.plot(t.iloc[j][["stress", "frictionless"]].to_numpy(), [j, j], color=GRID, lw=2, zorder=1)
    ax.set_yticks(y, t.index, fontsize=8.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("annualised Sharpe ratio of daily P&L per unit notional")
    ax.set_title("Costs decide: Sharpe without costs, with base costs and with stressed costs")
    ax.legend(loc="lower right")
    save(fig, "backtest_costs.png")
    b = pd.read_csv(config.RESULTS / "backtests" / "summary_base.csv").set_index("strategy")
    terms = ["delta", "gamma", "vega", "vanna", "volga", "theta", "rates", "residual", "hedge", "costs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, n in zip(axes, ["short_straddle_dh", "short_put_25_dh"]):
        v = np.array([b.loc[n, f"attr_{t}"] for t in terms]) / 1000
        ax.bar(terms, v, color=[SLOTS[0] if x >= 0 else SLOTS[1] for x in v], width=0.7)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(n)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("total P&L, $ thousands (one unit, 1996–2026)")
    fig.suptitle("Where the money came from: theta pays for gamma, costs take a large share", color=INK, fontsize=11)
    save(fig, "backtest_attribution.png")


def forecast_figures():
    t = pd.read_csv(config.RESULTS / "forecast" / "rv_losses.csv")
    order = ["GARCH", "GJR", "HAR", "logHAR_pk", "logHAR_pk_VIX", "VIX_affine", "VIX_raw"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(order))
    for i, h in enumerate((1, 5, 22)):
        v = t[t["h"] == h].set_index("model").reindex(order)["qlike_patton_vs_base_pct"]
        ax.bar(x + (i - 1) * 0.26, v.clip(lower=-45, upper=20), width=0.24, color=SLOTS[i], label=f"{h}-day horizon")
    ax.set_xticks(x, order, rotation=20)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("QLIKE vs EWMA, % (lower = better; clipped at +20%)")
    ax.set_title("SPX realized-variance forecasts, out of sample 2000–2026")
    ax.legend()
    save(fig, "forecast_rv.png")
    c = pd.read_parquet(config.CACHE / "forecast" / "corr_inputs.parquet")
    m = c[["implied", "target"]].resample("ME").mean().dropna()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(m.index, m["implied"], color=SLOTS[0], lw=1.6, label="implied (index vs top-50 ATM vols)")
    ax.plot(m.index, m["target"], color=SLOTS[1], lw=1.6, label="realized over the next 22 days")
    ax.set_ylabel("average pairwise correlation")
    ax.set_title("Implied correlation sits above subsequent realized correlation on average")
    ax.legend(loc="upper left")
    save(fig, "correlation.png")
    r = pd.read_csv(config.RESULTS / "rough" / "atm_skew_power_law.csv", parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(r.index, r["svi_slope"].rolling(6, min_periods=3).median(), color=SLOTS[0], lw=1.6,
            label="SVI-fitted market surfaces")
    ax.plot(r.index, r["heston_slope"].rolling(6, min_periods=3).median(), color=SLOTS[1], lw=1.6,
            label="fitted Heston surfaces")
    ax.axhline(-0.5, color=INK2, lw=0.7, ls=":")
    ax.set_ylabel("slope of log ATM skew on log T (7 days–6 months)")
    ax.set_title("ATM skew term structure: power-law slope, 6-month rolling median")
    ax.legend(loc="lower left")
    save(fig, "rough_skew_slope.png")


def main():
    model_figures()
    backtest_figures()
    forecast_figures()


if __name__ == "__main__":
    main()
