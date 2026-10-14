#!/usr/bin/env python3
# =============================================================================
# rawat_full_pipeline.py  —  one self-contained script that regenerates the
# Regime-B (blind-well / LOWO) result set for the Rawat PHIE paper directly from
# the raw + calculated LAS files, and writes every table and figure under its
# paper-position name, plus a MANIFEST and a TABLES.md.
#
# Produces (Regime B = the paper's primary evidence):
#   Table 1  well inventory + target audit         Fig 2  target-audit crossplot
#   Table 7  deterministic baseline ladder         Fig 5  sign-consistency screen
#   Table 8  per-well LOWO metrics                  Fig 6  LOWO-SFS trace
#   Table 9  ablation S0..S5                        Fig 7  baseline ladder
#   Table 14 few-shot recalibration                Fig 8  pooled LOWO crossplot
#   Table 15 attribution (SHAP or permutation)     Fig 9  per-well residuals
#   Table 16 Taylor statistics                     Fig 10-12 depth overlays
#                                                  Fig 13 leakage gauge
#                                                  Fig 15 few-shot curves
#                                                  Fig 16 attribution bar
#                                                  Fig 20 Taylor diagram
#
# NOT produced here (need a GPU box / extra libs): the Regime-A optimizer
# benchmark (Tables 10-13, Fig 14; XGBoost + torch MLPs + metaheuristics) and
# the SHAP beeswarm/dependence + LIME panels (Figs 17-19; shap/lime). Hooks and
# guidance are provided; see run_regime_a() and the README.
#
# Model backend: LightGBM if installed (the paper's model), else a regularized
# scikit-learn GradientBoosting that mirrors its settings, so the script runs in
# any environment. The backend actually used is recorded in the MANIFEST.
#
# Dependencies (always): numpy, pandas, scikit-learn, matplotlib.
# Optional (used if importable): lasio, lightgbm, shap.
# Deterministic: one seed governs all randomness.
# =============================================================================

from __future__ import annotations
import argparse, json, re, time, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# optional backends -----------------------------------------------------------
try:
    import lightgbm as lgb; _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
try:
    import shap; _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    las_raw_dir: str = "./Las files"
    las_calc_dir: str = "./Calculated"
    out_dir: str = "./results_regimeB"
    depth_col: str = "DEPTH"
    target: str = "PHIE"
    exclude_wells: Tuple[str, ...] = ("Akaff-1",)
    # physics constants
    rho_ma: float = 2.65
    rho_fl: float = 1.00
    dt_ma: float = 55.5
    dt_fl: float = 189.0
    phi_clamp: Tuple[float, float] = (0.0, 0.6)
    merge_tol_frac: float = 0.6
    min_well_rows: int = 50
    valid_range: dict = field(default_factory=lambda: dict(
        GR=(0, 300), RHOB=(1.5, 3.1), NPHI=(-0.05, 1.0), DT=(30, 240),
        RT_DEEP=(0.05, 5000), CALI=(4, 30), PHIT=(0, 0.6), VCL=(0, 1)))
    # model / selection
    model: str = "auto"           # "auto"|"lightgbm"|"gbr"|"ridge"
    sfs_cap: int = 10
    fewshot_fracs: Tuple[float, ...] = (0.10, 0.20)
    seed: int = 42
    dpi: int = 600        # publication resolution (> 300 dpi requirement)
    fast: bool = False        # smaller boosters for quick validation
    row_stride: int = 1       # decimate rows per well (validation speed)
    depth_track_wells: Tuple[str, ...] = ()   # auto: best/most-biased/worst
    # ---- Regime A (within-well optimizer benchmark) --------------------
    stages: str = "all"          # "all" (both regimes) | "B" (blind only)
    opt_agents: int = 20
    opt_iters: int = 10
    opt_patience: int = 3
    rs_trials: int = 50
    regime_a_models: Tuple[str, ...] = ("RandomForest", "XGBoost", "SimpleMLP", "OptimizedMLP")
    regime_a_optimizers: Tuple[str, ...] = ("AQPSO", "GWO", "Hybrid")


# =============================================================================
# LAS INGEST + MERGE  (self-contained LAS 2.0 reader; lasio if present)
# =============================================================================
_CANON = {
    "GR": ["GR", "HCGR", "HSGR", "CGR", "SGR"],
    "RHOB": ["RHOB", "RHOZ", "ZDEN", "DEN"],
    "NPHI": ["NPHI", "TNPH", "CNC", "NPOR"],
    "DT": ["DT", "DTC", "AC"], "PE": ["PE", "PEFZ", "PEF"],
    "RT_DEEP": ["RT_DEEP", "RD", "RLA5", "HLLD", "AHF90", "RT", "ILD"],
    "RT_SHAL": ["RT_SHAL", "RS", "RLA1", "HLLS", "AHF10"],
    "RXO": ["RXO", "RXOZ"], "SP": ["SP", "SPBR", "SPD"],
    "CALI": ["CALI", "HCAL", "CAL"], "BS": ["BS", "BIT"],
    "DRHO": ["DRHO", "HDRA", "ZCOR"],
    "PHIT": ["PHIT"], "PHIE": ["PHIE"], "VCL": ["VCL"], "SW": ["SW"],
}


def _harmonise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); up = {c.upper(): c for c in out.columns}
    for canon, al in _CANON.items():
        if canon in out.columns:
            continue
        for a in al:
            if a.upper() in up and up[a.upper()] != canon:
                out = out.rename(columns={up[a.upper()]: canon}); break
    return out


def read_las(path: str) -> pd.DataFrame:
    try:
        import lasio
        return lasio.read(str(path)).df().reset_index()
    except ImportError:
        pass
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    null, mnems, section, dstart = -999.25, [], None, None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            section = s[1].upper()
            if section == "A":
                dstart = i + 1; break
            continue
        if section == "W" and s.upper().startswith("NULL"):
            try:
                null = float(s.split(".", 1)[1].split(":")[0].strip().split()[0])
            except Exception:
                pass
        elif section == "C":
            mnems.append(re.split(r"[.\s]", s, maxsplit=1)[0].strip())
    rows = []
    for ln in lines[dstart or len(lines):]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            break
        rows.append(s.split())
    k = len(mnems)
    rows = [r[:k] for r in rows if len(r) >= k]
    df = pd.DataFrame(rows, columns=mnems).apply(pd.to_numeric, errors="coerce")
    return df.replace(null, np.nan)


def load_and_merge(cfg: Config) -> Dict[str, pd.DataFrame]:
    raw_dir, calc_dir = Path(cfg.las_raw_dir), Path(cfg.las_calc_dir)
    raws = sorted(list(raw_dir.glob("*.las")) + list(raw_dir.glob("*.LAS")))
    if not raws:
        raise FileNotFoundError(f"No LAS in {raw_dir}")
    calc_map = {re.sub(r"[^a-z0-9]", "", q.stem.lower()): q
                for q in list(calc_dir.glob("*.las")) + list(calc_dir.glob("*.LAS"))}
    excl = {re.sub(r"[^a-z0-9]", "", w.lower()) for w in cfg.exclude_wells}
    frames, merged_dir = {}, Path(cfg.out_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for rp in raws:
        well = rp.stem
        key = re.sub(r"[^a-z0-9]", "", well.lower())
        if key in excl:
            continue
        raw = _harmonise(read_las(str(rp)))
        raw = raw.rename(columns={raw.columns[0]: cfg.depth_col}).sort_values(cfg.depth_col)
        if key in calc_map:
            calc = _harmonise(read_las(str(calc_map[key])))
            calc = calc.rename(columns={calc.columns[0]: cfg.depth_col}).sort_values(cfg.depth_col)
            have = [c for c in ("PHIE", "PHIT", "SW", "VCL") if c in calc.columns]
            step = float(np.nanmedian(np.diff(raw[cfg.depth_col].values)))
            raw = pd.merge_asof(raw, calc[[cfg.depth_col] + have], on=cfg.depth_col,
                                direction="nearest", tolerance=max(step * cfg.merge_tol_frac, 1e-4))
        raw = raw.loc[:, ~raw.columns.duplicated()]
        if "NPHI" in raw and raw["NPHI"].notna().sum() > cfg.min_well_rows and float(raw["NPHI"].median()) > 1.5:
            raw["NPHI"] = raw["NPHI"] / 100.0
        raw = raw.set_index(cfg.depth_col)
        if cfg.row_stride > 1:
            raw = raw.iloc[::cfg.row_stride]
        for c, (lo, hi) in cfg.valid_range.items():
            if c in raw.columns:
                raw.loc[(raw[c] < lo) | (raw[c] > hi), c] = np.nan
        raw.attrs["well"] = well
        frames[well] = raw
        raw.reset_index().to_csv(merged_dir / f"{well}_merged.csv", index=False)
    print(f"[data] merged {len(frames)} wells: {sorted(frames)}")
    return frames


# =============================================================================
# FEATURES  (physics + per-well robust norm + PHIE_DERIVED; PHIT/VCL admitted)
# =============================================================================
def engineer(frames: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    out = {}
    for w, df in frames.items():
        d = df.copy()
        if "GR" in d and d["GR"].notna().sum() >= cfg.min_well_rows:
            gr = d["GR"]; lo, hi = np.nanpercentile(gr, 5), np.nanpercentile(gr, 95)
            igr = ((gr - lo) / max(hi - lo, 1e-6)).clip(0, 1)
            d["VSH_GR"] = (0.083 * (2 ** (3.7 * igr) - 1)).clip(0, 0.6)
        if "RHOB" in d:
            d["PHID"] = ((cfg.rho_ma - d["RHOB"]) / max(cfg.rho_ma - cfg.rho_fl, 1e-6)).clip(-0.15, 0.75)
        if "DT" in d:
            d["PHIS"] = ((d["DT"] - cfg.dt_ma) / max(cfg.dt_fl - cfg.dt_ma, 1e-6)).clip(-0.15, 0.75)
        if {"PHID", "NPHI"} <= set(d.columns):
            d["PHIDN"] = (d["PHID"] + d["NPHI"]) / 2
            d["ND_SEP"] = d["NPHI"] - d["PHID"]
        if "RT_DEEP" in d:
            d["LOG_RD"] = np.log10(d["RT_DEEP"].clip(lower=0.05))
        if "RT_SHAL" in d:
            d["LOG_RS"] = np.log10(d["RT_SHAL"].clip(lower=0.05))
        idx = d.index.to_numpy(float)
        d["REL_DEPTH"] = (idx - idx.min()) / max(idx.max() - idx.min(), 1e-9)
        # physics-derived effective porosity (needs PHIT)
        if {"PHIT", "VSH_GR"} <= set(d.columns):
            d["PHIE_DERIVED"] = (d["PHIT"] * (1 - d["VSH_GR"])).clip(*cfg.phi_clamp)
        # deterministic references for ladder / residual target
        if "PHID" in d:
            d["PHI_D_REF"] = d["PHID"].clip(*cfg.phi_clamp)
        if {"PHIT", "VCL"} <= set(d.columns):
            d["PHIE_VCLREF"] = (d["PHIT"] * (1 - d["VCL"])).clip(*cfg.phi_clamp)
        # per-well robust normalization of tool-unit features (label-free)
        for c in ("GR", "SP", "LOG_RD", "LOG_RS", "DRHO"):
            if c in d and d[c].notna().sum() >= cfg.min_well_rows:
                s = d[c].dropna(); iqr = float(s.quantile(.75) - s.quantile(.25))
                d[f"{c}_N"] = (d[c] - float(s.median())) / (iqr / 1.349 if iqr > 1e-9 else 1.0)
        d.attrs["well"] = w
        out[w] = d
    return out


# paper candidate pool (physics-space + interpreter suite; raw tool units enter
# only via their _N companions, which the sign-consistency screen then rejects)
CANDIDATE_POOL = ["PHIT", "VCL", "PHIE_DERIVED", "PHID", "PHIS", "PHIDN",
                  "ND_SEP", "NPHI", "GR_N", "LOG_RD_N", "LOG_RS_N", "REL_DEPTH"]


# =============================================================================
# METRICS + MODEL FACTORY
# =============================================================================
def metrics(a, p) -> dict:
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p); a, p = a[m], p[m]
    if len(a) < 5:
        return dict(n=int(len(a)), R2=np.nan, RMSE=np.nan, MAE=np.nan, bias=np.nan, pearson_r=np.nan)
    return dict(n=int(len(a)), R2=float(r2_score(a, p)),
                RMSE=float(np.sqrt(mean_squared_error(a, p))),
                MAE=float(mean_absolute_error(a, p)), bias=float(np.mean(p - a)),
                pearson_r=float(np.corrcoef(a, p)[0, 1]) if np.std(p) > 0 else np.nan)


def resolve_backend(cfg: Config) -> str:
    if cfg.model != "auto":
        return cfg.model
    return "lightgbm" if _HAS_LGBM else "gbr"


def make_booster(cfg: Config, high_capacity: bool = False):
    """Regularized gradient booster (paper) or high-capacity variant (ablation S0/S1)."""
    backend = resolve_backend(cfg)
    if backend == "ridge":
        return Ridge(alpha=10.0)
    if backend == "lightgbm" and _HAS_LGBM:
        ne_hi, ne_lo = (150, 150) if cfg.fast else (500, 600)
        if high_capacity:
            return lgb.LGBMRegressor(n_estimators=ne_hi, learning_rate=0.1, num_leaves=31,
                                     min_child_samples=20, random_state=cfg.seed, verbose=-1)
        return lgb.LGBMRegressor(n_estimators=ne_lo, learning_rate=0.03, num_leaves=7,
                                 min_child_samples=200, reg_alpha=1.0, reg_lambda=10.0,
                                 subsample=0.8, colsample_bytree=0.7, random_state=cfg.seed, verbose=-1)
    # scikit-learn GradientBoosting fallback mirroring the LightGBM settings
    if high_capacity:
        return GradientBoostingRegressor(n_estimators=(150 if cfg.fast else 500), learning_rate=0.1, max_depth=5,
                                         min_samples_leaf=20, random_state=cfg.seed)
    return GradientBoostingRegressor(n_estimators=(150 if cfg.fast else 600), learning_rate=0.03, max_depth=3,
                                     min_samples_leaf=200, subsample=0.8, max_features=0.7,
                                     random_state=cfg.seed)


def make_ridge():
    return Ridge(alpha=10.0)


# =============================================================================
# LEAKAGE-SAFE LOWO CORE
# =============================================================================
def _clean(df, feats, target):
    if target not in df.columns or any(f not in df.columns for f in feats):
        return np.empty((0, len(feats))), np.empty(0), np.zeros(len(df), bool)
    m = df[target].notna() & df[feats].notna().all(axis=1)
    return df.loc[m, feats].to_numpy(float), df.loc[m, target].to_numpy(float), m.to_numpy()


def _fit_predict(train, held, feats, cfg, model_factory, target_mode="direct",
                 global_scaling=False):
    """Fit on train wells, predict held-out well. Standardization on train only.
    target_mode: 'direct' -> PHIE; 'residual' -> PHIE - PHI_D_REF (added back)."""
    Xs, ys = [], []
    for df in train.values():
        X, y, mask = _clean(df, feats, cfg.target)
        if len(y) < cfg.min_well_rows:
            continue
        if target_mode == "residual" and "PHI_D_REF" in df.columns:
            y = y - df.loc[mask, "PHI_D_REF"].to_numpy(float)
        Xs.append(X); ys.append(y)
    if not Xs:
        return None
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    m = model_factory()
    m.fit((Xtr - mu) / sd, ytr)
    Xte, yte, mask = _clean(held, feats, cfg.target)
    if mask.sum() < cfg.min_well_rows:
        return None
    pred = m.predict((Xte - mu) / sd)
    if target_mode == "residual" and "PHI_D_REF" in held.columns:
        pred = pred + held.loc[mask, "PHI_D_REF"].to_numpy(float)
    pred = np.clip(pred, *cfg.phi_clamp)
    return dict(idx=held.index[mask], y=yte, pred=pred)


def lowo_eval(frames, feats, cfg, model_factory, target_mode="direct",
              global_scaling=False, verbose=False):
    preds, acts, rows = {}, {}, []
    for held in sorted(frames):
        train = {w: d for w, d in frames.items() if w != held}
        r = _fit_predict(train, frames[held], feats, cfg, model_factory,
                         target_mode, global_scaling)
        if r is None:
            continue
        preds[held] = pd.Series(r["pred"], index=r["idx"])
        acts[held] = pd.Series(r["y"], index=r["idx"])
        rows.append(dict(well=held, **metrics(r["y"], r["pred"])))
        if verbose:
            print(f"    {held:14s} R2={rows[-1]['R2']:+.3f} RMSE={rows[-1]['RMSE']:.4f} n={rows[-1]['n']}")
    per_well = pd.DataFrame(rows)
    pooled = {}
    if preds:
        A = np.concatenate([acts[w].to_numpy() for w in preds])
        P = np.concatenate([preds[w].to_numpy() for w in preds])
        pooled = metrics(A, P)
        per_well = pd.concat([per_well, pd.DataFrame([dict(well="OVERALL", **pooled)])],
                             ignore_index=True)
    return dict(per_well=per_well, predictions=preds, actuals=acts, pooled=pooled)


# =============================================================================
# FEATURE SELECTION (availability -> sign-consistency -> LOWO-SFS, ridge scorer)
# =============================================================================
def availability_filter(frames, pool, cfg):
    return [f for f in pool
            if all(f in d.columns and d[f].notna().sum() >= cfg.min_well_rows
                   for d in frames.values())]


def sign_consistency(frames, feats, cfg):
    """Keep features whose sign of correlation with PHIE is stable across wells."""
    rows, keep = [], []
    for f in feats:
        signs, mags = [], []
        for d in frames.values():
            m = d[cfg.target].notna() & d[f].notna()
            if m.sum() < cfg.min_well_rows:
                continue
            r = np.corrcoef(d.loc[m, f], d.loc[m, cfg.target])[0, 1]
            if np.isfinite(r):
                signs.append(np.sign(r)); mags.append(abs(r))
        if not signs:
            continue
        consistent = len(set(signs)) == 1
        rows.append(dict(feature=f, median_abs_r=float(np.median(mags)),
                         n_same_sign=int(max(signs.count(1), signs.count(-1))),
                         n_wells=len(signs), consistent=consistent))
        if consistent:
            keep.append(f)
    return keep, pd.DataFrame(rows).sort_values("median_abs_r", ascending=False)


def lowo_sfs(frames, feats, cfg):
    """Sequential forward selection scored by mean per-fold LOWO R² (ridge)."""
    selected, trace, remaining = [], [], list(feats)
    best_overall = -np.inf
    while remaining and len(selected) < cfg.sfs_cap:
        scored = []
        for f in remaining:
            trial = selected + [f]
            res = lowo_eval(frames, trial, cfg, make_ridge)
            scored.append((res["pooled"].get("R2", -np.inf), f))
        scored.sort(reverse=True)
        best_r2, best_f = scored[0]
        trace.append(dict(step=len(selected) + 1, added=best_f, pooled_lowo_r2=round(best_r2, 4)))
        if best_r2 <= best_overall + 1e-4 and selected:
            break
        selected.append(best_f); remaining.remove(best_f); best_overall = best_r2
    return selected, pd.DataFrame(trace)


# =============================================================================
# DETERMINISTIC LADDER + TARGET AUDIT
# =============================================================================
def target_audit(frames, cfg):
    rows = []
    for w, d in frames.items():
        if not {"PHIE", "PHIE_VCLREF"} <= set(d.columns):
            continue
        m = d["PHIE"].notna() & d["PHIE_VCLREF"].notna()
        rows.append(dict(well=w, **{k: metrics(d.loc[m, "PHIE"], d.loc[m, "PHIE_VCLREF"])[k]
                                    for k in ("n", "R2", "RMSE")}))
    return pd.DataFrame(rows)


def deterministic_ladder(frames, cfg):
    def pooled(ref):
        a, p = [], []
        for d in frames.values():
            if ref not in d.columns:
                continue
            m = d["PHIE"].notna() & d[ref].notna()
            a.append(d.loc[m, "PHIE"].to_numpy(float)); p.append(d.loc[m, ref].to_numpy(float))
        return metrics(np.concatenate(a), np.concatenate(p)) if a else {}
    rows = []
    rows.append(dict(estimator="PHIT x (1 - VCL)  (interpreter policy)", **pooled("PHIE_VCLREF")))
    rows.append(dict(estimator="PHIT x (1 - VSH_GR)  (GR-only)", **pooled("PHIE_DERIVED")))
    # PHID vs PHIT (like-for-like physics)
    a, p = [], []
    for d in frames.values():
        if not {"PHIT", "PHI_D_REF"} <= set(d.columns):
            continue
        m = d["PHIT"].notna() & d["PHI_D_REF"].notna()
        a.append(d.loc[m, "PHIT"].to_numpy(float)); p.append(d.loc[m, "PHI_D_REF"].to_numpy(float))
    if a:
        rows.append(dict(estimator="PHID vs PHIT  (like-for-like)", **metrics(np.concatenate(a), np.concatenate(p))))
    return pd.DataFrame(rows)


# =============================================================================
# FEW-SHOT AFFINE RECALIBRATION
# =============================================================================
def fewshot(lowo_res, frames, cfg):
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for w in sorted(lowo_res["predictions"]):
        yhat = lowo_res["predictions"][w].to_numpy()
        ytrue = lowo_res["actuals"][w].to_numpy()
        base = metrics(ytrue, yhat)
        rec = dict(well=w, R2_blind=round(base["R2"], 4), bias_blind=round(base["bias"], 4))
        for frac in cfg.fewshot_fracs:
            n = len(yhat); k = max(10, int(frac * n))
            idx = rng.choice(n, k, replace=False)
            mask = np.ones(n, bool); mask[idx] = False
            # affine fit on the seeded calibration subset
            A = np.vstack([yhat[idx], np.ones(k)]).T
            slope, inter = np.linalg.lstsq(A, ytrue[idx], rcond=None)[0]
            cal = np.clip(slope * yhat[mask] + inter, *cfg.phi_clamp)
            mm = metrics(ytrue[mask], cal)
            pct = int(frac * 100)
            rec[f"R2_{pct}pct"] = round(mm["R2"], 4)
            rec[f"bias_{pct}pct"] = round(mm["bias"], 4)
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# ATTRIBUTION (SHAP if available, else permutation importance)  — blind
# =============================================================================
def attribution(frames, feats, cfg, model_factory):
    """Held-out attribution: for each well, explain the fold model on that well
    (SHAP if installed, else permutation importance). Aggregated across wells."""
    per_feat = {f: [] for f in feats}
    method = "SHAP (held-out)" if _HAS_SHAP else "permutation importance (held-out)"
    for held in sorted(frames):
        train = {w: d for w, d in frames.items() if w != held}
        Xs, ys = [], []
        for df in train.values():
            X, y, _ = _clean(df, feats, cfg.target)
            if len(y) >= cfg.min_well_rows:
                Xs.append(X); ys.append(y)
        if not Xs:
            continue
        Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        m = model_factory(); m.fit((Xtr - mu) / sd, ytr)
        Xte, yte, _ = _clean(frames[held], feats, cfg.target)
        if len(yte) < cfg.min_well_rows:
            continue
        Xz = (Xte - mu) / sd
        take = min(1000, len(Xz))
        sel = np.random.default_rng(cfg.seed).choice(len(Xz), take, replace=False)
        if _HAS_SHAP:
            try:
                expl = shap.TreeExplainer(m)
                sv = np.abs(expl.shap_values(Xz[sel])).mean(0)
            except Exception:
                r = permutation_importance(m, Xz[sel], yte[sel], n_repeats=5,
                                           random_state=cfg.seed, scoring="r2")
                sv = np.clip(r.importances_mean, 0, None)
        else:
            r = permutation_importance(m, Xz[sel], yte[sel], n_repeats=5,
                                       random_state=cfg.seed, scoring="r2")
            sv = np.clip(r.importances_mean, 0, None)
        for f, v in zip(feats, sv):
            per_feat[f].append(v)
    imp = pd.DataFrame([dict(feature=f, importance=float(np.mean(v)) if v else 0.0)
                        for f, v in per_feat.items()]).sort_values("importance", ascending=False)
    imp.insert(0, "rank", range(1, len(imp) + 1))
    return imp, method


# =============================================================================
# TAYLOR STATISTICS
# =============================================================================
def taylor_stats(frames, lowo_res, rand_res, cfg):
    def series(ref_col=None, res=None):
        a, p = [], []
        if res is not None:
            for w in res["predictions"]:
                a.append(res["actuals"][w].to_numpy()); p.append(res["predictions"][w].to_numpy())
        else:
            for d in frames.values():
                if ref_col not in d.columns:
                    continue
                m = d["PHIE"].notna() & d[ref_col].notna()
                a.append(d.loc[m, "PHIE"].to_numpy(float)); p.append(d.loc[m, ref_col].to_numpy(float))
        a, p = np.concatenate(a), np.concatenate(p)
        r = np.corrcoef(a, p)[0, 1]; sd_ratio = np.std(p) / np.std(a)
        crmse = np.sqrt(np.mean(((p - p.mean()) - (a - a.mean())) ** 2)) / np.std(a)
        return dict(corr=round(float(r), 3), sd_ratio=round(float(sd_ratio), 3),
                    crmse_norm=round(float(crmse), 3), n=int(len(a)))
    rows = [dict(estimator="ML blind (LOWO)", **series(res=lowo_res)),
            dict(estimator="ML random-split", **series(res=rand_res)),
            dict(estimator="PHIT x (1-VCL)", **series(ref_col="PHIE_VCLREF")),
            dict(estimator="PHIT x (1-VSH_GR)", **series(ref_col="PHIE_DERIVED"))]
    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================
PAL = {"lowo": "#2F5B6C", "random": "#C9812E", "policy": "#8C6D4F",
       "measured": "#333", "phit": "#9AA7AD", "accent": "#B0472F"}
CMAP = LinearSegmentedColormap.from_list("d", ["#fff", "#9DB8C0", PAL["lowo"]])


def _save(fig, out, stem, cfg, manifest, label, caption):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(file=f"figures/{stem}.png", paper_label=label, caption=caption))


def fig_crossplot(res, out, cfg, manifest, label, stem, title):
    a = np.concatenate([res["actuals"][w].to_numpy() for w in res["predictions"]])
    p = np.concatenate([res["predictions"][w].to_numpy() for w in res["predictions"]])
    ov = res["pooled"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.hexbin(a, p, gridsize=60, cmap=CMAP, norm=LogNorm(), mincnt=1)
    hi = max(0.45, np.percentile(a, 99.5)) * 1.05
    ax.plot([0, hi], [0, hi], "k-", lw=1.1, label="1:1")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    ax.set_xlabel("Measured PHIE (v/v)"); ax.set_ylabel("Predicted PHIE (v/v)")
    ax.set_title(f"{title}\nR$^2$={ov['R2']:.3f}  RMSE={ov['RMSE']:.4f}  n={ov['n']:,}")
    ax.legend(loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, title)


def fig_perwell_resid(res, out, cfg, manifest, label, stem):
    wells = [w for w in sorted(res["predictions"])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [(res["predictions"][w].to_numpy() - res["actuals"][w].to_numpy()) for w in wells]
    ax.boxplot(data, tick_labels=wells, showfliers=False)
    ax.axhline(0, color=PAL["accent"], lw=1)
    ax.set_ylabel("Residual (pred - obs, v/v)")
    ax.set_xticklabels(wells, rotation=40, ha="right")
    ax.set_title("Per-well LOWO residual distributions")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, "Per-well LOWO residuals")


def fig_depth_overlay(frames, res, w, out, cfg, manifest, label, stem, subtitle,
                      window_m=150.0):
    """Two panels: full column (context) + a representative zoomed window
    (the ~window_m interval of highest measured-PHIE variability), so measured
    vs predicted are actually distinguishable."""
    if w not in res["predictions"]:
        return
    df = frames[w]; pred = res["predictions"][w]
    idx = pred.index.to_numpy(float)
    phie = df["PHIE"].reindex(pred.index).to_numpy(float)
    lo0, hi0 = float(idx.min()), float(idx.max())
    # slide a window and pick the most variable (reservoir) interval
    best = (lo0, min(lo0 + window_m, hi0), -1.0)
    x = lo0
    while x + window_m <= hi0:
        m = (idx >= x) & (idx < x + window_m)
        if m.sum() > 30:
            s = float(np.nanstd(phie[m]))
            if s > best[2]:
                best = (x, x + window_m, s)
        x += window_m / 2
    zlo, zhi, _ = best

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 8),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    for ax, (a, b, ttl, zoom) in zip(
            axes, [(lo0, hi0, "full column", False),
                   (zlo, zhi, f"detail {int(zlo)}\u2013{int(zhi)} m", True)]):
        sub = df[(df.index >= a) & (df.index <= b)]
        if not zoom:                                   # decimate full column only
            st = max(1, len(sub) // 4000); sub = sub.iloc[::st]
        lw = 0.9 if zoom else 0.5
        if "PHIT" in sub:
            ax.plot(sub["PHIT"], sub.index, color=PAL["phit"], lw=lw * 0.8,
                    alpha=0.65, label="PHIT")
        ax.plot(sub["PHIE"], sub.index, color=PAL["measured"], lw=lw, alpha=0.9,
                label="Measured PHIE")
        ax.plot(pred.reindex(sub.index), sub.index, color=PAL["lowo"], lw=lw,
                alpha=0.85, label="Predicted (LOWO)")
        ax.set_ylim(a, b); ax.invert_yaxis()
        ax.set_xlim(-0.02, 0.45); ax.set_xlabel("Porosity (v/v)")
        ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel("Depth (m)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{w} \u2014 {subtitle}", fontsize=11, y=1.0)
    fig.tight_layout()
    _save(fig, out, stem, cfg, manifest, label, f"Depth overlay {w}")


def fig_leakage_gauge(lowo_res, rand_res, out, cfg, manifest, label, stem):
    lw = lowo_res["per_well"].set_index("well")
    wells = [w for w in lw.index if w != "OVERALL"]
    rn = rand_res["per_well"].set_index("well").reindex(wells)["R2"]
    x = np.arange(len(wells))
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(wells) + 2), 4.2))
    ax.bar(x - .2, lw.reindex(wells)["R2"], .4, color=PAL["lowo"], label="LOWO (blind)")
    ax.bar(x + .2, rn.values, .4, color=PAL["random"], label="Random split")
    ax.set_xticks(x); ax.set_xticklabels(wells, rotation=40, ha="right")
    ax.set_ylabel("R$^2$"); ax.set_ylim(min(0, lw["R2"].min()) - .02, 1.02)
    gap = float((rn - lw.reindex(wells)["R2"]).mean())
    ax.set_title(f"Leakage gauge: random - LOWO = {gap:+.3f} R$^2$ (mean)")
    ax.legend(loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, "Leakage gauge")


def fig_ladder(ladder, lowo_pooled, rand_pooled, out, cfg, manifest, label, stem):
    items = [(r["estimator"], r["R2"]) for _, r in ladder.iterrows()]
    items.append(("ML blind (LOWO)", lowo_pooled["R2"]))
    items.append(("ML random-split", rand_pooled["R2"]))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [PAL["random"] if "random" in n else PAL["lowo"] if "LOWO" in n else PAL["policy"] for n, _ in items]
    ax.barh(range(len(items)), [v for _, v in items], color=colors)
    for i, (_, v) in enumerate(items):
        ax.text(v + .01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _ in items], fontsize=8)
    ax.set_xlim(min(0, min(v for _, v in items)) - .05, 1.08)
    ax.set_xlabel("Pooled R$^2$ vs observed PHIE"); ax.set_title("Honest baseline ladder")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, "Baseline ladder")


def fig_audit(frames, out, cfg, manifest, label, stem):
    wells = sorted(frames)
    n = len(wells); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    for ax, w in zip(axes.ravel(), wells):
        d = frames[w]
        if not {"PHIE", "PHIE_VCLREF"} <= set(d.columns):
            ax.axis("off"); continue
        m = d["PHIE"].notna() & d["PHIE_VCLREF"].notna()
        ax.scatter(d.loc[m, "PHIE_VCLREF"], d.loc[m, "PHIE"], s=3, alpha=.2, color=PAL["lowo"], edgecolors="none")
        ax.plot([0, .45], [0, .45], "k-", lw=.8)
        r2 = metrics(d.loc[m, "PHIE"], d.loc[m, "PHIE_VCLREF"])["R2"]
        ax.set_title(f"{w}  R²={r2:.3f}", fontsize=9); ax.set_xlim(0, .45); ax.set_ylim(0, .45)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("PHIT x (1 - VCL)  (v/v)"); fig.supylabel("Observed PHIE (v/v)")
    fig.suptitle("Target audit: observed PHIE vs PHIT x (1 - VCL)", y=1.01)
    fig.tight_layout()
    _save(fig, out, stem, cfg, manifest, label, "Target audit")


def fig_sfs(trace, out, cfg, manifest, label, stem):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    r2 = trace["pooled_lowo_r2"].to_numpy(float)
    steps = np.arange(1, len(trace) + 1)
    # a step is "retained" if it improved on the best-so-far before it
    retained, best = [], -np.inf
    for v in r2:
        retained.append(v > best + 1e-4); best = max(best, v)
    ax.plot(steps, r2, "-", color=PAL["lowo"], lw=1.5, zorder=1)
    for s, v, r, name in zip(steps, r2, retained, trace["added"]):
        if r:
            ax.plot(s, v, "o", ms=9, color=PAL["lowo"], zorder=2)
            ax.annotate(name, (s, v), fontsize=8, xytext=(5, -10),
                        textcoords="offset points")
        else:                                          # evaluated but not kept
            ax.plot(s, v, "o", ms=9, mfc="white", mec=PAL["accent"],
                    mew=1.6, zorder=2)
            ax.annotate(f"{name}\n(not retained)", (s, v), fontsize=8,
                        color=PAL["accent"], xytext=(5, -22),
                        textcoords="offset points")
    n_keep = int(sum(retained))
    ax.set_xlabel("SFS step"); ax.set_ylabel("pooled LOWO R$^2$ (ridge)")
    ax.set_title(f"LOWO-guided sequential forward selection "
                 f"({n_keep} features retained)")
    ax.set_xticks(steps)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, stem, cfg, manifest, label, "LOWO-SFS trace")


def fig_signscreen(audit_df, out, cfg, manifest, label, stem):
    df = audit_df.sort_values("median_abs_r")
    colors = [PAL["lowo"] if c else PAL["accent"] for c in df["consistent"]]
    fig, ax = plt.subplots(figsize=(6, .4 * len(df) + 1))
    ax.barh(range(len(df)), df["median_abs_r"], color=colors)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["feature"])
    ax.set_xlabel("median |r| with PHIE across wells")
    ax.set_title("Sign-consistency screen (teal=consistent, red=flips)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, "Sign-consistency screen")


def fig_fewshot(few, out, cfg, manifest, label, stem):
    wells = few["well"].tolist(); x = np.arange(len(wells))
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(wells) + 2), 4.2))
    cols = ["R2_blind"] + [f"R2_{int(f*100)}pct" for f in cfg.fewshot_fracs]
    width = .8 / len(cols)
    for i, c in enumerate(cols):
        if c in few:
            ax.bar(x + (i - (len(cols) - 1) / 2) * width, few[c], width, label=c.replace("R2_", ""))
    ax.set_xticks(x); ax.set_xticklabels(wells, rotation=40, ha="right")
    ax.set_ylabel("R$^2$"); ax.set_ylim(min(0, few[cols].min().min()) - .02, 1.02)
    ax.set_title("Few-shot affine recalibration"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, "Few-shot recalibration")


def fig_importance(imp, method, out, cfg, manifest, label, stem):
    df = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, .5 * len(df) + 1))
    ax.barh(range(len(df)), df["importance"], color=PAL["lowo"])
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["feature"])
    ax.set_xlabel(f"mean importance — {method}")
    ax.set_title("Blind attribution (held-out)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, stem, cfg, manifest, label, "Blind attribution")


def fig_taylor(tay, out, cfg, manifest, label, stem):
    fig = plt.figure(figsize=(7.2, 6.6)); ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0); ax.set_thetamax(90)
    rmax = max(1.05, float(tay["sd_ratio"].max()) * 1.15)
    ax.set_ylim(0, rmax)
    # centred-RMS-difference arcs (dotted), centred on the reference (r=1, theta=0)
    for E in (0.25, 0.5, 0.75, 1.0, 1.25):
        t = np.linspace(0, np.pi, 300)
        x = 1 + E * np.cos(t); y = E * np.sin(t)
        r = np.hypot(x, y); th = np.arctan2(y, x)
        keep = (th >= 0) & (th <= np.pi / 2) & (r <= rmax)
        if keep.any():
            ax.plot(th[keep], r[keep], ls=":", color="#9AA7AD", lw=0.8, zorder=1)
    # reference (observed) standard-deviation arc r=1
    tt = np.linspace(0, np.pi / 2, 100)
    ax.plot(tt, np.ones_like(tt), ls="--", color="#666", lw=0.8, zorder=1)
    colors = {"ML blind (LOWO)": PAL["lowo"], "ML random-split": PAL["random"],
              "PHIT x (1-VCL)": PAL["policy"], "PHIT x (1-VSH_GR)": PAL["accent"]}
    for _, r in tay.iterrows():
        theta = np.arccos(np.clip(r["corr"], -1, 1))
        ax.plot(theta, r["sd_ratio"], "o", ms=12,
                color=colors.get(r["estimator"], "#555"), zorder=3,
                label=f"{r['estimator']} (r={r['corr']:.3f}, E'={r['crmse_norm']:.3f})")
    ax.plot(0, 1, "k*", ms=16, zorder=4, label="Observed (reference)")
    # short title (avoids right-edge clipping); axis meaning as a subtitle line
    ax.set_title("Taylor diagram vs observed PHIE", pad=28, fontsize=12)
    ax.text(np.radians(45), rmax * 1.14,
            "radius = \u03c3/\u03c3$_{ref}$ ; dotted arcs = centred RMS difference",
            ha="center", va="bottom", fontsize=8.5, color="#444")
    ax.legend(loc="upper right", bbox_to_anchor=(1.46, 1.05), fontsize=8, frameon=True)
    _save(fig, out, stem, cfg, manifest, label, "Taylor diagram")


# =============================================================================
# ORCHESTRATION
# =============================================================================

def shap_lime_figures(frames, feats, cfg, model_factory, out_dir, manifest):
    """Fig 17 (SHAP beeswarm), Fig 18 (SHAP dependence), Fig 19 (LIME).
    Guarded: produced only if shap / lime are installed (paper environment)."""
    import matplotlib.pyplot as plt
    # deployment model: trained on all wells
    Xs, ys = [], []
    for d in frames.values():
        X, y, _ = _clean(d, feats, cfg.target)
        if len(y) >= cfg.min_well_rows:
            Xs.append(X); ys.append(y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    model = model_factory().fit((X - mu) / sd, y)
    rng = np.random.default_rng(cfg.seed)
    take = min(2000, len(X)); sel = rng.choice(len(X), take, replace=False)
    Xz = (X[sel] - mu) / sd

    # ---- SHAP beeswarm + dependence ----
    if _HAS_SHAP:
        try:
            expl = shap.TreeExplainer(model); sv = expl.shap_values(Xz)
            # Fig 17 beeswarm
            plt.figure(figsize=(7, 4.5))
            shap.summary_plot(sv, features=X[sel], feature_names=feats, show=False)
            plt.title("SHAP beeswarm (deployment model)")
            plt.tight_layout()
            plt.savefig(out_dir / "figures" / "Fig_17_shap_beeswarm.png", dpi=cfg.dpi, bbox_inches="tight")
            plt.savefig(out_dir / "figures" / "Fig_17_shap_beeswarm.pdf", bbox_inches="tight"); plt.close()
            manifest.append(dict(file="figures/Fig_17_shap_beeswarm.png",
                                 paper_label="Fig. 17", caption="SHAP beeswarm."))
            # Fig 18 dependence: VCL colored by PHIT (if both present)
            if "VCL" in feats:
                vi = feats.index("VCL"); ci = feats.index("PHIT") if "PHIT" in feats else None
                plt.figure(figsize=(6, 4.5))
                shap.dependence_plot(vi, sv, X[sel], feature_names=feats,
                                     interaction_index=ci, show=False)
                plt.title("SHAP dependence: VCL (interaction PHIT)")
                plt.tight_layout()
                plt.savefig(out_dir / "figures" / "Fig_18_shap_dependence.png", dpi=cfg.dpi, bbox_inches="tight")
                plt.savefig(out_dir / "figures" / "Fig_18_shap_dependence.pdf", bbox_inches="tight"); plt.close()
                manifest.append(dict(file="figures/Fig_18_shap_dependence.png",
                                     paper_label="Fig. 18", caption="SHAP dependence (VCL x PHIT)."))
            print("[figures] SHAP beeswarm + dependence written (Fig 17, 18)")
        except Exception as e:
            print(f"[figures] SHAP figures skipped ({e})")
    else:
        print("[figures] Fig 17/18 skipped: `pip install shap` to generate SHAP beeswarm/dependence")

    # ---- LIME at high / median / low porosity ----
    try:
        from lime.lime_tabular import LimeTabularExplainer
        explr = LimeTabularExplainer(training_data=(X - mu) / sd, feature_names=feats,
                                     mode="regression", discretize_continuous=False,
                                     random_state=cfg.seed)
        yhat = model.predict((X - mu) / sd)
        picks = {"high porosity": int(np.argmax(yhat)),
                 "median porosity": int(np.argsort(yhat)[len(yhat) // 2]),
                 "tight/shaly": int(np.argmin(yhat))}
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, (label, idx) in zip(axes, picks.items()):
            exp = explr.explain_instance((X[idx] - mu) / sd, model.predict, num_features=len(feats))
            items = exp.as_list()
            names = [k for k, _ in items][::-1]; vals = [v for _, v in items][::-1]
            ax.barh(range(len(vals)), vals,
                    color=["#2F5B6C" if v >= 0 else "#B0472F" for v in vals])
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            ax.axvline(0, color="k", lw=0.8); ax.set_title(f"{label}\npred={yhat[idx]:.3f}")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("LIME local explanations (deployment model)")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "Fig_19_lime_local.png", dpi=cfg.dpi, bbox_inches="tight")
        fig.savefig(out_dir / "figures" / "Fig_19_lime_local.pdf", bbox_inches="tight"); plt.close(fig)
        manifest.append(dict(file="figures/Fig_19_lime_local.png",
                             paper_label="Fig. 19", caption="LIME local explanations."))
        print("[figures] LIME local explanations written (Fig 19)")
    except ImportError:
        print("[figures] Fig 19 skipped: `pip install lime` to generate LIME explanations")
    except Exception as e:
        print(f"[figures] LIME skipped ({e})")


def run(cfg: Config) -> dict:
    t0 = time.time()
    out = Path(cfg.out_dir); (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    manifest, tables_md = [], []
    backend = resolve_backend(cfg)
    print(f"[cfg] model backend = {backend}"
          f"{'  (LightGBM not installed -> scikit-learn GradientBoosting fallback)' if backend=='gbr' and cfg.model=='auto' else ''}")

    def emit_table(df, label, stem, caption):
        df.round(5).to_csv(out / "tables" / f"{stem}.csv", index=False)
        manifest.append(dict(file=f"tables/{stem}.csv", paper_label=label, caption=caption))
        tables_md.append((label, caption, df))

    # ---- load, merge, engineer ------------------------------------------
    frames = engineer(load_and_merge(cfg), cfg)

    # ---- Table 1 + Fig 2: inventory & target audit ----------------------
    audit = target_audit(frames, cfg).set_index("well")
    inv = []
    for w, d in sorted(frames.items()):
        idx = d.index.to_numpy(float)
        step = float(np.nanmedian(np.diff(idx))) if len(idx) > 2 else np.nan
        inv.append(dict(well=w, depth_min_m=round(float(idx.min()), 1),
                        depth_max_m=round(float(idx.max()), 1), median_step_m=round(step, 3),
                        labelled_depths_n=int(d[cfg.target].notna().sum()),
                        target_audit_R2=round(float(audit.loc[w, "R2"]), 4) if w in audit.index else np.nan))
    inv_df = pd.DataFrame(inv)
    tot = dict(well="Total", depth_min_m="", depth_max_m="", median_step_m="",
               labelled_depths_n=int(inv_df["labelled_depths_n"].sum()),
               target_audit_R2=f"{inv_df.target_audit_R2.min():.2f}-{inv_df.target_audit_R2.max():.2f}")
    emit_table(pd.concat([inv_df, pd.DataFrame([tot])], ignore_index=True),
               "Table 1", "Table_01_well_inventory_audit", "Well inventory and target-audit summary.")
    fig_audit(frames, out / "figures", cfg, manifest, "Fig. 2", "Fig_02_target_audit_crossplot")

    # ---- Table 7 + Fig 7: deterministic ladder --------------------------
    ladder = deterministic_ladder(frames, cfg)
    emit_table(ladder, "Table 7", "Table_07_deterministic_ladder",
               "Deterministic estimators against observed PHIE, pooled.")

    # ---- feature selection: availability -> sign -> LOWO-SFS ------------
    avail = availability_filter(frames, CANDIDATE_POOL, cfg)
    consistent, sign_df = sign_consistency(frames, avail, cfg)
    emit_table(sign_df, "Table S1", "Table_S1_sign_consistency", "Per-well sign-consistency screen.")
    fig_signscreen(sign_df, out / "figures", cfg, manifest, "Fig. 5", "Fig_05_sign_consistency_screen")
    selected, sfs_trace = lowo_sfs(frames, consistent, cfg)
    emit_table(sfs_trace, "Table S2", "Table_S2_sfs_trace", "LOWO-SFS trace.")
    fig_sfs(sfs_trace, out / "figures", cfg, manifest, "Fig. 6", "Fig_06_lowo_sfs_trace")
    print(f"[select] availability={avail}\n[select] sign-consistent={consistent}\n[select] SELECTED={selected}")

    # ---- Table 8 + Figs 8-12: primary LOWO ------------------------------
    print("[LOWO] evaluating selected pipeline...")
    lowo = lowo_eval(frames, selected, cfg, lambda: make_booster(cfg), verbose=True)
    t8 = lowo["per_well"][["well", "n", "R2", "RMSE", "MAE", "bias", "pearson_r"]].copy()
    t8 = pd.concat([t8[t8.well != "OVERALL"].sort_values("R2", ascending=False),
                    t8[t8.well == "OVERALL"]], ignore_index=True)
    emit_table(t8, "Table 8", "Table_08_perwell_lowo_metrics", "Per-well blind (LOWO) performance.")
    fig_crossplot(lowo, out / "figures", cfg, manifest, "Fig. 8", "Fig_08_lowo_pooled_crossplot",
                  "Pooled blind-well (LOWO)")
    fig_perwell_resid(lowo, out / "figures", cfg, manifest, "Fig. 9", "Fig_09_lowo_perwell_residuals")
    pw = lowo["per_well"][lowo["per_well"].well != "OVERALL"].set_index("well")
    order = pw.sort_values("R2")
    if cfg.depth_track_wells:
        picks = list(cfg.depth_track_wells)
    else:
        picks = []
        for cand in [order.index[-1], order["bias"].abs().idxmax(), order.index[0]]:
            if cand not in picks:
                picks.append(cand)
        for cand in order.index:            # top up to 3 distinct wells
            if len(picks) >= 3: break
            if cand not in picks: picks.append(cand)
    subt = {order.index[-1]: "highest LOWO R\u00b2", order["bias"].abs().idxmax(): "largest bias", order.index[0]: "lowest LOWO R\u00b2"}
    for lab, stem, w in zip(["Fig. 10", "Fig. 11", "Fig. 12"],
                            ["Fig_10_depth_overlay", "Fig_11_depth_overlay", "Fig_12_depth_overlay"], picks):
        fig_depth_overlay(frames, lowo, w, out / "figures", cfg, manifest, lab,
                          f"{stem}_{w.replace('-','_')}", subt.get(w, "blind-well LOWO overlay"))

    # ---- random-split leakage gauge (same pipeline) + Fig 13 ------------
    print("[random] leakage-gauge random split...")
    rand = random_split_eval(frames, selected, cfg, lambda: make_booster(cfg))
    fig_leakage_gauge(lowo, rand, out / "figures", cfg, manifest, "Fig. 13", "Fig_13_leakage_gauge")
    fig_ladder(ladder, lowo["pooled"], rand["pooled"], out / "figures", cfg, manifest,
               "Fig. 7", "Fig_07_baseline_ladder")

    # ---- per-well leakage table (LOWO vs random) ------------------------
    _lw = lowo["per_well"].set_index("well"); _rn = rand["per_well"].set_index("well")
    _pw = []
    for w in [x for x in _lw.index if x != "OVERALL"]:
        lo = float(_lw.loc[w, "R2"]); rr = float(_rn.loc[w, "R2"]) if w in _rn.index else float("nan")
        _pw.append(dict(well=w, R2_lowo=round(lo, 4), R2_random=round(rr, 4),
                        inflation_random_minus_lowo=round(rr - lo, 4)))
    _pwdf = pd.DataFrame(_pw).sort_values("inflation_random_minus_lowo", ascending=False)
    _ov = dict(well="OVERALL", R2_lowo=round(lowo["pooled"]["R2"], 4),
               R2_random=round(rand["pooled"]["R2"], 4),
               inflation_random_minus_lowo=round(rand["pooled"]["R2"] - lowo["pooled"]["R2"], 4))
    emit_table(pd.concat([_pwdf, pd.DataFrame([_ov])], ignore_index=True), "Table 8b",
               "Table_08b_perwell_leakage", "Per-well random-split vs LOWO inflation.")

    # ---- per-well deterministic ladder table ----------------------------
    _det = []
    for w, dd in sorted(frames.items()):
        row = {"well": w}
        for nm, col in [("PHIT_x_1-VCL", "PHIE_VCLREF"), ("PHIT_x_1-VSH_GR", "PHIE_DERIVED")]:
            if col in dd.columns:
                m = dd["PHIE"].notna() & dd[col].notna()
                row[nm] = round(metrics(dd.loc[m, "PHIE"], dd.loc[m, col])["R2"], 4) if m.sum() > 5 else None
        if {"PHIT", "PHI_D_REF"} <= set(dd.columns):
            m = dd["PHIT"].notna() & dd["PHI_D_REF"].notna()
            row["PHID_vs_PHIT"] = round(metrics(dd.loc[m, "PHIT"], dd.loc[m, "PHI_D_REF"])["R2"], 4) if m.sum() > 5 else None
        _det.append(row)
    emit_table(pd.DataFrame(_det), "Table 7b", "Table_07b_perwell_deterministic",
               "Per-well R2 of deterministic estimators (incl. density-vs-PHIT).")

    # ---- Table 9: ablation ----------------------------------------------
    print("[ablation] S0..S5...")
    t9 = run_ablation(frames, avail, consistent, selected, cfg)
    emit_table(t9, "Table 9", "Table_09_ablation", "Ablation of design choices under pooled LOWO.")

    # ---- Table 14 + Fig 15: few-shot ------------------------------------
    few = fewshot(lowo, frames, cfg)
    emit_table(few, "Table 14", "Table_14_fewshot", "Few-shot affine recalibration (held-out remainder).")
    fig_fewshot(few, out / "figures", cfg, manifest, "Fig. 15", "Fig_15_fewshot_recalibration")

    # ---- Table 15 + Fig 16: attribution ---------------------------------
    print("[attribution] blind importance...")
    imp, method = attribution(frames, selected, cfg, lambda: make_booster(cfg))
    emit_table(imp, "Table 15", "Table_15_attribution", f"Blind attribution ({method}).")
    fig_importance(imp, method, out / "figures", cfg, manifest, "Fig. 16", "Fig_16_attribution_bar")
    shap_lime_figures(frames, selected, cfg, lambda: make_booster(cfg), out, manifest)

    # ---- Table 16 + Fig 20: Taylor --------------------------------------
    tay = taylor_stats(frames, lowo, rand, cfg)
    emit_table(tay, "Table 16", "Table_16_taylor_stats", "Taylor statistics, pooled.")
    fig_taylor(tay, out / "figures", cfg, manifest, "Fig. 20", "Fig_20_taylor_diagram")

    # Regime A (within-well benchmark, Tables 10-13, Fig 14) is generated by
    # regime_a_benchmark.py via run_all_regimes.py — this module is Regime B only.

    # ---- MANIFEST + TABLES.md + summary ---------------------------------
    pd.DataFrame(manifest).to_csv(out / "MANIFEST.csv", index=False)
    with open(out / "TABLES.md", "w") as fh:
        fh.write(f"# Rawat PHIE results (backend: {backend})\n\n")
        for label, caption, df in tables_md:
            fh.write(f"## {label}. {caption}\n\n{df.round(4).to_markdown(index=False)}\n\n")
    summary = dict(backend=backend, stages=cfg.stages, selected_features=selected,
                   lowo_pooled=lowo["pooled"], random_pooled=rand["pooled"],
                   ladder={r['estimator']: round(r['R2'], 4) for _, r in ladder.iterrows()},
                   runtime_s=round(time.time() - t0, 1))
    json.dump(summary, open(out / "SUMMARY.json", "w"), indent=2, default=float)
    print(f"\n[done] {len(manifest)} artifacts in {out}  ({summary['runtime_s']}s)")
    print(f"[summary] LOWO pooled R2={lowo['pooled'].get('R2'):.4f}  "
          f"random R2={rand['pooled'].get('R2'):.4f}  "
          f"policy PHIT(1-VCL) R2={ladder.iloc[0]['R2']:.4f}")
    return dict(frames=frames, lowo=lowo, rand=rand, ladder=ladder, selected=selected,
                ablation=t9, fewshot=few, importance=imp, taylor=tay, summary=summary)


def random_split_eval(frames, feats, cfg, model_factory):
    """Pooled 80/20 random split with train-only scaling (leakage gauge)."""
    rng = np.random.default_rng(cfg.seed)
    Xtr, ytr, te = [], [], []
    for w, df in sorted(frames.items()):
        X, y, mask = _clean(df, feats, cfg.target)
        if len(y) < cfg.min_well_rows:
            continue
        o = np.arange(len(y)); rng.shuffle(o); cut = int(.8 * len(y))
        Xtr.append(X[o[:cut]]); ytr.append(y[o[:cut]])
        te.append((w, X[o[cut:]], y[o[cut:]], df.index[mask][o[cut:]]))
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    m = model_factory(); m.fit((Xtr - mu) / sd, ytr)
    preds, acts, rows = {}, {}, []
    for w, X, y, idx in te:
        p = np.clip(m.predict((X - mu) / sd), *cfg.phi_clamp)
        preds[w] = pd.Series(p, index=idx); acts[w] = pd.Series(y, index=idx)
        rows.append(dict(well=w, **metrics(y, p)))
    A = np.concatenate([acts[w].to_numpy() for w in preds])
    P = np.concatenate([preds[w].to_numpy() for w in preds])
    pooled = metrics(A, P)
    per_well = pd.concat([pd.DataFrame(rows), pd.DataFrame([dict(well="OVERALL", **pooled)])], ignore_index=True)
    return dict(per_well=per_well, predictions=preds, actuals=acts, pooled=pooled)


def run_ablation(frames, avail, consistent, selected, cfg):
    DEV_POOL = [f for f in ["PHIT", "PHIE_DERIVED", "PHID", "PHIDN", "PHIS", "ND_SEP",
                            "NPHI", "GR_N", "LOG_RD_N", "REL_DEPTH"]
                if all(f in d.columns for d in frames.values())]
    stages = [
        ("S0", DEV_POOL, "residual", lambda: make_booster(cfg, high_capacity=True), "conventional: residual target, high-capacity GBM"),
        ("S1", DEV_POOL, "direct", lambda: make_booster(cfg, high_capacity=True), "direct PHIE target"),
        ("S2", avail, "direct", lambda: make_booster(cfg, high_capacity=True), "per-well norm + physics pool"),
        ("S3", consistent, "direct", lambda: make_booster(cfg, high_capacity=True), "sign-consistency screen"),
        ("S4", selected, "direct", lambda: make_booster(cfg, high_capacity=True), "LOWO-SFS selection"),
        ("S5", selected, "direct", lambda: make_booster(cfg), "final regularized booster (deployed)"),
    ]
    rows = []
    for sid, feats, mode, mf, desc in stages:
        res = lowo_eval(frames, feats, cfg, mf, target_mode=mode)
        p = res["pooled"]
        worst = res["per_well"][res["per_well"].well != "OVERALL"]["R2"].min()
        rows.append(dict(stage=sid, design_choice=desc, n_features=len(feats),
                         pooled_lowo_r2=round(p.get("R2", np.nan), 4),
                         pooled_rmse=round(p.get("RMSE", np.nan), 4),
                         worst_well_r2=round(float(worst), 4)))
        print(f"    {sid}: R2={p.get('R2', float('nan')):.4f} ({desc})")
    return pd.DataFrame(rows)


def run_regime_a(cfg: Config):
    """Hook for the Regime-A optimizer benchmark. Needs xgboost+torch and the
    benchmark reference module; not run in this environment. See README."""
    raise NotImplementedError(
        "Regime A (Tables 10-13, Fig 14) requires the AQPSO_GWO_RandomSplit benchmark "
        "module plus xgboost/torch. Run it separately on the GPU box, or import it here.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rawat PHIE Regime-B full pipeline")
    ap.add_argument("--las-raw"); ap.add_argument("--las-calc")
    ap.add_argument("--out"); ap.add_argument("--model", choices=["auto", "lightgbm", "gbr", "ridge"])
    a = ap.parse_args(argv)
    cfg = Config()
    if a.las_raw: cfg.las_raw_dir = a.las_raw
    if a.las_calc: cfg.las_calc_dir = a.las_calc
    if a.out: cfg.out_dir = a.out
    if a.model: cfg.model = a.model
    return run(cfg)


if __name__ == "__main__":
    main()
