"""Reference values for the demo's JavaScript math (demo/data/test_vectors.json), computed with the Python library:
model implied vols from a bundled surface file, Black-76 values and sensitivities. The demo's self-test compares
its own numbers against these.

    python scripts/make_test_vectors.py [--surface demo/data/surfaces/SPX/2026-11-03.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.strategies.book import black_sensitivities  # noqa: E402
from voldesk.surface import heston as hs  # noqa: E402
from voldesk.surface import sabr as sb  # noqa: E402
from voldesk.surface import svi  # noqa: E402
from voldesk.surface.compare import SliceInterpSurface  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default=str(config.WEB_BUNDLE / "surfaces" / "SPX" / "2026-11-03.json"))
    args = ap.parse_args()
    b = json.loads(Path(args.surface).read_text(encoding="utf-8"))
    M = b["models"]
    Ts = [e["T"] for e in b["expiries"]]
    T_pts = sorted(set([Ts[0], Ts[min(2, len(Ts) - 1)], 0.5 * (Ts[0] + Ts[1]), 0.15, 1.0, Ts[-1] * 1.2]))
    k_pts = [-0.4, -0.2, -0.1, -0.03, 0.0, 0.03, 0.08]
    out = {"surface": Path(args.surface).name, "k": k_pts, "T": T_pts, "iv": {}}
    surfs = {}
    if "svi" in M:
        surfs["svi"] = svi.SVISurface([s["T"] for s in M["svi"]["slices"]],
                                      [svi.SVIParams(s["a"], s["b"], s["rho"], s["m"], s["s"]) for s in M["svi"]["slices"]])
    if "essvi" in M:
        surfs["essvi"] = svi.SVISurface([s["T"] for s in M["essvi"]["slices"]],
                                        [svi.ESSVISlice(s["theta"], s["rho"], s["psi"]) for s in M["essvi"]["slices"]])
    if "ssvi" in M:
        m = M["ssvi"]
        surfs["ssvi"] = svi.SSVISurface(m["Ts"], m["thetas"], svi.SSVIParams(m["rho"], m["eta"], m["gamma"]))
    if "sabr" in M:
        sl = M["sabr"]["slices"]
        surfs["sabr"] = SliceInterpSurface(
            [s["T"] for s in sl],
            [lambda k, s=s: sb.sabr_iv(np.exp(np.asarray(k, float)), 1.0, s["T"], s["alpha"], s["beta"], s["rho"],
                                       s["nu"]) ** 2 * s["T"] for s in sl])
    for name, srf in surfs.items():
        out["iv"][name] = [[float(srf.iv(np.array([k]), T)[0]) for k in k_pts] for T in T_pts]
    if "heston" in M:
        h = M["heston"]
        p = hs.HestonParams(h["v0"], h["kappa"], h["theta"], h["sigma"], h["rho"])
        out["iv"]["heston"] = [[float(hs.heston_iv(1.0, np.exp([k]), T, 1.0, p, method="quad")[0]) for k in k_pts]
                               for T in T_pts]
    cases = [(1, 100.0, 95.0, 0.25, 0.20, 0.99), (-1, 100.0, 95.0, 0.25, 0.20, 0.99), (1, 6800.0, 7000.0, 0.08, 0.15, 0.996),
             (-1, 6800.0, 6200.0, 0.5, 0.25, 0.98), (1, 50.0, 50.0, 1.0, 0.6, 0.95)]
    bl = []
    for cp, F, K, T, s, D in cases:
        r = black_sensitivities(np.array([cp]), np.array([F]), np.array([K]), np.array([T]), np.array([s]), np.array([D]))
        bl.append({"cp": cp, "F": F, "K": K, "T": T, "sigma": s, "D": D,
                   **{k: float(np.asarray(v).ravel()[0]) for k, v in r.items()}})
    out["black"] = bl
    out["black_keys"] = {"value": "Black-76 price", "dF": "dV/dF", "dFF": "d2V/dF2", "ds": "dV/dsigma (per 1.00)",
                         "dtau": "dV/dtau at fixed F and D (per year)", "dFs": "d2V/dF dsigma", "dss": "d2V/dsigma2",
                         "dD": "dV/dD"}
    dest = config.WEB_BUNDLE / "test_vectors.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
