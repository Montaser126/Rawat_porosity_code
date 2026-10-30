#!/usr/bin/env python3
"""Create a synthetic eight-well dataset that has the same layout as the paper's LAS files.

The real Rawat Basin logs are confidential, so this script builds a fully artificial
dataset for testing and learning the workflow. No value comes from a real well.

Each synthetic well is a sand-shale sequence. Clay volume (VCL) and total porosity
(PHIT) are generated first, the "interpreter" effective porosity is
PHIE = PHIT x (1 - VCL) plus small noise and a small per-well bias, and the raw tool
curves (GR, RHOB, NPHI, DT, PE, resistivities, SP, CALI, BS, DRHO) are forward-modelled
from those properties. Each well also gets its own tool calibration offsets, so a
random train-test split leaks well identity just as it does in real multi-well data.

Usage:
    python make_synthetic_data.py --out synthetic_data
This writes synthetic_data/las_raw/SYN-1.las ... SYN-8.las and
synthetic_data/las_calc/SYN-1.las ... SYN-8.las (LAS 2.0, depth in metres).
"""
import argparse
from pathlib import Path

import numpy as np

RAW_CURVES = [("GR", "GAPI"), ("RHOB", "G/C3"), ("NPHI", "V/V"), ("DT", "US/F"), ("PE", "B/E"),
              ("RT_DEEP", "OHMM"), ("RT_SHAL", "OHMM"), ("RXO", "OHMM"), ("SP", "MV"),
              ("CALI", "IN"), ("BS", "IN"), ("DRHO", "G/C3")]
CALC_CURVES = [("PHIT", "V/V"), ("VCL", "V/V"), ("PHIE", "V/V"), ("SW", "V/V")]
NULL = -999.25


def layered(rng, n, mean_bed=40):
    """Piecewise-constant sand (0) / shale (1) indicator with random bed thicknesses."""
    out, i, shale = np.empty(n), 0, rng.random() < 0.5
    while i < n:
        k = max(3, int(rng.exponential(mean_bed)))
        out[i:i + k] = 1.0 if shale else 0.0
        shale = not shale
        i += k
    return out


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def make_well(rng, n, top):
    depth = top + np.arange(n) * 0.1524
    trend = (depth - depth.min()) / (depth.max() - depth.min())          # 0 at top, 1 at base
    facies = smooth(layered(rng, n), 7)
    vcl = np.clip(0.08 + 0.75 * facies + rng.normal(0, 0.04, n), 0, 1)
    phit = np.clip(0.34 - 0.10 * trend - 0.08 * vcl + smooth(rng.normal(0, 0.02, n), 9), 0.03, 0.42)
    sw = np.clip(0.35 + 0.5 * vcl + rng.normal(0, 0.05, n), 0.1, 1.0)

    # per-well tool calibration offsets (the "well identity" a model can memorize)
    gr_gain, gr_shift = rng.uniform(0.85, 1.2), rng.uniform(-10, 15)
    rho_shift, nphi_shift, dt_shift = rng.normal(0, 0.03), rng.normal(0, 0.02), rng.normal(0, 4)
    rt_gain = rng.uniform(0.7, 1.5)

    rho_ma = 2.65 * (1 - vcl) + 2.55 * vcl
    gr = gr_gain * (25 + 110 * vcl) + gr_shift + rng.normal(0, 4, n)
    rhob = phit * 1.0 + (1 - phit) * rho_ma + rho_shift + rng.normal(0, 0.02, n)
    nphi = phit + 0.25 * vcl + nphi_shift + rng.normal(0, 0.015, n)
    dt = 55.5 + 133.5 * phit + 25 * vcl + dt_shift + rng.normal(0, 2, n)
    pe = 1.9 * (1 - vcl) + 3.3 * vcl + rng.normal(0, 0.1, n)
    rw = 0.08
    rt = rt_gain * rw / np.maximum(phit, 0.03) ** 2 / np.maximum(sw, 0.1) ** 2 * (1 - 0.6 * vcl)
    rt = np.clip(rt * np.exp(rng.normal(0, 0.1, n)), 0.2, 2000)
    rs = np.clip(rt * rng.uniform(0.6, 0.9), 0.2, 2000)
    rxo = np.clip(rt * rng.uniform(0.3, 0.7), 0.1, 2000)
    sp = -60 * (1 - vcl) + rng.normal(0, 3, n) + rng.uniform(-20, 20)
    bs = np.full(n, 8.5)
    cali = bs + 0.8 * vcl + np.abs(rng.normal(0, 0.15, n))
    drho = rng.normal(0, 0.015, n)

    phie = np.clip(phit * (1 - vcl) + rng.normal(0, 0.006, n) + rng.normal(0, 0.004), 0, 0.4)
    raw = dict(GR=gr, RHOB=rhob, NPHI=nphi, DT=dt, PE=pe, RT_DEEP=rt, RT_SHAL=rs, RXO=rxo,
               SP=sp, CALI=cali, BS=bs, DRHO=drho)
    calc = dict(PHIT=phit, VCL=vcl, PHIE=phie, SW=sw)
    return depth, raw, calc


def write_las(path, well, depth, curves, units):
    step = depth[1] - depth[0]
    names = ["DEPT"] + list(curves)
    lines = ["~Version Information",
             " VERS.                 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
             " WRAP.                  NO : ONE LINE PER DEPTH STEP",
             "~Well Information",
             f" STRT.M        {depth[0]:.4f} : START DEPTH",
             f" STOP.M        {depth[-1]:.4f} : STOP DEPTH",
             f" STEP.M        {step:.4f} : STEP",
             f" NULL.         {NULL} : NULL VALUE",
             f" WELL.         {well} : WELL (SYNTHETIC)",
             " COMP.         SYNTHETIC : NOT REAL DATA",
             "~Curve Information",
             " DEPT.M                  : DEPTH"]
    lines += [f" {c}.{units[c]} : {c}" for c in curves]
    lines.append("~ASCII")
    data = np.column_stack([depth] + [curves[c] for c in curves])
    lines += [" ".join(f"{v:.5f}" for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic_data", help="output folder")
    ap.add_argument("--wells", type=int, default=8, help="number of synthetic wells")
    ap.add_argument("--samples", type=int, default=3000, help="depth samples per well")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "las_raw").mkdir(parents=True, exist_ok=True)
    (out / "las_calc").mkdir(parents=True, exist_ok=True)
    units = dict(RAW_CURVES + CALC_CURVES)
    for i in range(1, a.wells + 1):
        well = f"SYN-{i}"
        n = int(a.samples * rng.uniform(0.8, 1.2))
        depth, raw, calc = make_well(rng, n, top=rng.uniform(800, 1200))
        write_las(out / "las_raw" / f"{well}.las", well, depth, raw, units)
        write_las(out / "las_calc" / f"{well}.las", well, depth, calc, units)
        print(f"{well}: {n} samples, {depth[0]:.0f}-{depth[-1]:.0f} m")
    print(f"Synthetic LAS files written to {out.resolve()}")


if __name__ == "__main__":
    main()
