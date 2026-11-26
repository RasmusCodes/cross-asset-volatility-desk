"""Rough volatility on SPX: the ATM-skew power law across 371 month-end surfaces, Heston's skew term structure on the
same dates, and a rough Bergomi check on two dates.

    python scripts/rough_study.py
Outputs: results/rough/atm_skew_power_law.csv, rbergomi_check.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.surface import heston as hs  # noqa: E402
from voldesk.surface.rough import power_law_exponent, rbergomi_prices, svi_atm_skew  # noqa: E402

OUT = config.RESULTS / "rough"


def heston_atm_skew(p: dict, T: float, h: float = 1e-3) -> float:
    pp = hs.HestonParams(p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
    iv = hs.heston_iv(1.0, np.exp(np.array([-h, h])), T, 1.0, pp)
    return float(abs(iv[1] - iv[0]) / (2 * h))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted((config.CACHE / "surface_panel" / "monthend").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        m = r.get("models", {})
        if "svi" not in m or not m["svi"].get("params"):
            continue
        sl = m["svi"]["params"]["slices"]
        T = np.array([s["T"] for s in sl])
        psi = np.array([svi_atm_skew(s["a"], s["b"], s["rho"], s["m"], s["s"], s["T"]) for s in sl])
        fit = power_law_exponent(T, psi)
        rec = {"date": r["date"], "svi_slope": fit["slope"], "H_implied": fit["H"], "n": fit["n"], "r2": fit["r2"],
               "psi_1m": float(np.interp(30 / 365, T, psi)) if len(T) else np.nan}
        if "heston" in m and m["heston"].get("params", {}).get("v0") is not None:
            Th = T[(T >= 7 / 365) & (T <= 0.5)]
            if len(Th) >= 3:
                ph = np.array([heston_atm_skew(m["heston"]["params"], t) for t in Th])
                hf = power_law_exponent(Th, ph)
                rec.update({"heston_slope": hf["slope"]})
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "atm_skew_power_law.csv", index=False, float_format="%.5g")
    print(d[["svi_slope", "H_implied", "r2", "heston_slope"]].describe().round(3).to_string())
    # rough Bergomi on two dates: fit (H, eta, rho) roughly by grid on the short-dated SVI smiles
    checks = []
    for date in ("2017-06-30", "2020-03-31"):
        f = config.CACHE / "surface_panel" / "monthend" / f"{date}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        sl = [s for s in r["models"]["svi"]["params"]["slices"] if s["T"] <= 0.35]
        if len(sl) < 3:
            continue
        ks = np.linspace(-0.15, 0.05, 9)
        target = {s["T"]: np.sqrt(np.maximum(s["a"] + s["b"] * (s["rho"] * (ks - s["m"]) + np.sqrt((ks - s["m"]) ** 2 + s["s"] ** 2)), 1e-8) / s["T"]) for s in sl}
        xi0 = float(np.mean([v[np.argmin(np.abs(ks))] ** 2 for v in target.values()]))
        best = None
        for H in (0.05, 0.1, 0.2, 0.35, 0.5):
            for eta in (1.0, 1.5, 2.0, 2.5):
                for rho in (-0.95, -0.85, -0.7):
                    iv = rbergomi_prices(list(target), ks, H, eta, rho, xi0, n_paths=8000, seed=1)
                    err = []
                    for (T, mv), (T2, tv) in zip(iv.items(), target.items()):
                        err.append(np.nanmean((mv - tv) ** 2))
                    rmse = float(np.sqrt(np.mean(err))) * 100
                    if best is None or rmse < best[0]:
                        best = (rmse, H, eta, rho)
        heston_rmse = np.nan
        hp = r["models"].get("heston", {}).get("params")
        if hp:
            pp = hs.HestonParams(hp["v0"], hp["kappa"], hp["theta"], hp["sigma"], hp["rho"])
            e = [np.nanmean((hs.heston_iv(1.0, np.exp(ks), T, 1.0, pp) - tv) ** 2) for T, tv in target.items()]
            heston_rmse = float(np.sqrt(np.mean(e))) * 100
        checks.append({"date": date, "maturities": len(target), "rb_rmse_volpts": best[0], "H": best[1], "eta": best[2],
                       "rho": best[3], "heston_rmse_volpts_same_points": heston_rmse})
        print(checks[-1])
    pd.DataFrame(checks).to_csv(OUT / "rbergomi_check.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
