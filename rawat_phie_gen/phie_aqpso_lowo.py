#!/usr/bin/env python3
# =============================================================================
# phie_aqpso_lowo.py
# -----------------------------------------------------------------------------
# One module that predicts petrophysical porosity curves from RAW wireline logs
# under honest leave-one-well-out (LOWO) evaluation, with the four study models
# each tuned by a Hybrid AQPSO-GWO optimizer using GROUPED (well-aware) inner CV.
#
# Input  : a folder containing only per-well "*_processed.csv" files (9 wells),
#          each with raw logs + the IP curves PHIE, PHIT, VCL, SW.
#
# Target switch (cfg.target):
#   "PHIT"          - predict total porosity from raw logs        (log-native)
#   "VCL"           - predict shale volume from raw logs          (log-native)
#   "PHIE_direct"   - predict effective porosity directly         (hard, honest)
#   "PHIE_composite"- predict PHIT and VCL separately from logs,  (recommended)
#                     then PHIE = PHIT x (1 - VCL)                 physics-consistent
#
# Why these choices (see the review): the porosity tools respond to TOTAL pore
# space, so PHIT/VCL are the log-native quantities; PHIE is a derived policy
# (PHIT x (1-VCL)). Feeding PHIT/VCL/SW as inputs to predict PHIE is circular,
# so they are ALWAYS excluded from the input features here.
#
# Guarantees
#   * Raw-log inputs only; PHIT/VCL/SW/PHIE never used as features.
#   * Mnemonics harmonized across logging generations; the common curve set
#     present in all wells is used, plus physics features derived from it.
#   * LOWO outer loop: the held-out well is seen by NOTHING that is fitted.
#   * Inner tuning is GROUPED by well (GroupKFold / inner-LOWO), so the tuning
#     objective cannot leak well identity.
#   * All scaling fitted on training wells only; metrics on original v/v scale.
#
# Optional deps: xgboost (else GradientBoosting fallback), torch (else sklearn
# MLPRegressor fallback). RandomForest and the whole harness need only sklearn.
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# optional backends ----------------------------------------------------------
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class Config:
    """All settings in one place; override in code or via --config YAML."""
    input_dir: str = "./processed"          # folder of *_processed.csv
    out_dir: str = "./results_aqpso_lowo"

    # target: "PHIT" | "VCL" | "PHIE_direct" | "PHIE_composite"
    target: str = "PHIE_composite"
    # models to run/compare (subset of the four)
    models: Tuple[str, ...] = ("random_forest", "xgboost",
                               "simple_mlp", "optimized_mlp")

    # ---- data source -------------------------------------------------------
    # "csv" : read per-well *_processed.csv from input_dir
    # "las" : merge raw LAS (las_raw_dir) with calculated LAS (las_calc_dir)
    input_mode: str = "csv"
    las_raw_dir: str = "./Las files"
    las_calc_dir: str = "./Calculated"
    merge_tol_frac: float = 0.6              # depth-merge tolerance = step * frac
    save_merged_csv: bool = True            # write merged per-well CSVs to out_dir

    # schema
    depth_col: str = "DEPTH"
    exclude_wells: Tuple[str, ...] = ("Akaff-1",)   # anomalous policy/QC; see review
    # curves that must never be inputs (targets / interpretation products)
    leakage_cols: Tuple[str, ...] = ("PHIE", "PHIT", "VCL", "SW", "SWT", "SWE",
                                     "BVOL", "CVOL")

    # physics constants
    rho_matrix: float = 2.65
    rho_fluid: float = 1.00
    dt_matrix: float = 55.5
    dt_fluid: float = 189.0
    phi_clamp: Tuple[float, float] = (0.0, 0.6)

    # data hygiene
    min_well_rows: int = 50
    valid_range: dict = field(default_factory=lambda: dict(
        GR=(0, 300), RHOB=(1.5, 3.1), NPHI=(-0.05, 1.0), DT=(30, 240),
        RT_DEEP=(0.05, 5000), CALI=(4, 30)))

    # AQPSO-GWO optimizer
    opt_agents: int = 16
    opt_iters: int = 10
    opt_patience: int = 3
    inner_folds: int = 3                     # grouped-by-well inner CV folds
    max_train_per_well: int = 0              # >0 subsamples train rows for speed
    seed: int = 42

    # figures
    make_figures: bool = True
    dpi: int = 300

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        raw = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        for k in ("models", "exclude_wells", "leakage_cols", "phi_clamp"):
            if k in raw and raw[k] is not None:
                raw[k] = tuple(raw[k])
        return cls(**{k: v for k, v in raw.items()
                      if k in cls.__dataclass_fields__})

    def to_json(self, path: str) -> None:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(self).items()},
                  open(path, "w", encoding="utf-8"), indent=2)


# =============================================================================
# 2. HARMONISATION + IO
# =============================================================================
# canonical mnemonic map (first alias found wins), covering the two logging
# generations present in the Rawat wells.
_CANON = {
    "GR":      ["GR", "HCGR", "HSGR", "CGR", "SGR"],
    "RHOB":    ["RHOB", "RHOZ", "ZDEN", "DEN"],
    "NPHI":    ["NPHI", "TNPH", "CNC", "NPOR"],
    "DT":      ["DT", "DTC", "AC"],
    "PE":      ["PE", "PEFZ", "PEF"],
    "RT_DEEP": ["RT_DEEP", "RD", "RLA5", "HLLD", "AHF90", "RT", "ILD"],
    "RT_SHAL": ["RT_SHAL", "RS", "RLA1", "HLLS", "AHF10"],
    "RXO":     ["RXO", "RXOZ"],
    "SP":      ["SP", "SPBR", "SPD"],
    "CALI":    ["CALI", "HCAL", "CAL"],
    "BS":      ["BS", "BIT"],
    "DRHO":    ["DRHO", "HDRA", "ZCOR"],
    # IP products kept for target/QC, never inputs
    "PHIT": ["PHIT"], "PHIE": ["PHIE"], "VCL": ["VCL"], "SW": ["SW"],
}


def _harmonise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    upper = {c.upper(): c for c in out.columns}
    for canon, aliases in _CANON.items():
        if canon in out.columns:
            continue
        for a in aliases:
            if a.upper() in upper and upper[a.upper()] != canon:
                out = out.rename(columns={upper[a.upper()]: canon})
                break
    return out


def _finalise(df: pd.DataFrame, well: str, cfg: Config) -> Optional[pd.DataFrame]:
    """Shared post-load cleanup used by BOTH the CSV and LAS paths:
    harmonise mnemonics, drop duplicate columns, fix neutron units, set the
    depth index, and apply physical-range QC (out-of-range -> NaN)."""
    df = _harmonise(df)
    df = df.loc[:, ~df.columns.duplicated()]              # drop DEPTH:2 etc.
    # neutron porosity -> FRACTION (Rawat wells mix p.u. 0-100 and fraction 0-1)
    if "NPHI" in df.columns and df["NPHI"].notna().sum() > cfg.min_well_rows:
        if float(df["NPHI"].median()) > 1.5:
            df["NPHI"] = df["NPHI"] / 100.0
    if cfg.depth_col in df.columns:
        df = df.sort_values(cfg.depth_col).set_index(cfg.depth_col)
    for c, (lo, hi) in cfg.valid_range.items():           # physical-range QC
        if c in df.columns:
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    df.attrs["well"] = well
    return df


def read_las(path: str) -> pd.DataFrame:
    """Read a LAS 2.0 (no-wrap) file. Uses lasio if installed, else a compact
    built-in parser that handles the ~C/~A sections, the NULL value, and any
    ~P parameter block between them."""
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
                dstart = i + 1
                break
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


def _load_from_las(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Merge each well's raw LAS with its calculated LAS (PHIE/PHIT/VCL/SW) on
    depth (nearest within step*merge_tol_frac), then finalise. Wells are matched
    by filename stem between las_raw_dir and las_calc_dir."""
    raw_dir, calc_dir = Path(cfg.las_raw_dir), Path(cfg.las_calc_dir)
    raws = sorted(list(raw_dir.glob("*.las")) + list(raw_dir.glob("*.LAS")))
    if not raws:
        raise FileNotFoundError(f"No LAS files in {raw_dir}")
    calc_map = {re.sub(r'[^a-z0-9]', '', q.stem.lower()): q
                for q in list(calc_dir.glob("*.las")) + list(calc_dir.glob("*.LAS"))}
    excl = {re.sub(r'[^a-z0-9]', '', w.lower()) for w in cfg.exclude_wells}
    targets = ["PHIE", "PHIT", "SW", "VCL"]
    frames, merged_dir = {}, Path(cfg.out_dir) / "merged"
    if cfg.save_merged_csv:
        merged_dir.mkdir(parents=True, exist_ok=True)

    for rp in raws:
        well = rp.stem
        key = re.sub(r'[^a-z0-9]', '', well.lower())
        if key in excl:
            continue
        raw = read_las(str(rp))
        raw = raw.rename(columns={raw.columns[0]: cfg.depth_col}).sort_values(cfg.depth_col)
        raw = _harmonise(raw)
        # attach targets from the calculated LAS, if present
        if key in calc_map:
            calc = read_las(str(calc_map[key]))
            calc = calc.rename(columns={calc.columns[0]: cfg.depth_col})
            calc = _harmonise(calc).sort_values(cfg.depth_col)
            have = [c for c in targets if c in calc.columns]
            step = float(np.nanmedian(np.diff(raw[cfg.depth_col].values)))
            tol = max(step * cfg.merge_tol_frac, 1e-4)
            raw = pd.merge_asof(raw, calc[[cfg.depth_col] + have],
                                on=cfg.depth_col, direction="nearest", tolerance=tol)
        else:
            print(f"[data][warn] no calculated LAS for {well}; targets missing")
        df = _finalise(raw, well, cfg)
        if cfg.save_merged_csv:
            df.reset_index().to_csv(merged_dir / f"{well}_merged.csv", index=False)
        frames[well] = df
    if cfg.save_merged_csv:
        print(f"[data] merged per-well CSVs written to {merged_dir}")
    return frames


def load_wells(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Dispatch to the CSV or LAS loader, then report the loaded wells."""
    if cfg.input_mode.lower() == "las":
        frames = _load_from_las(cfg)
        print(f"[data] loaded {len(frames)} wells (LAS-merged): {sorted(frames)}")
        return frames

    p = Path(cfg.input_dir)
    files = sorted(p.glob("*_processed.csv")) or sorted(p.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No *_processed.csv in {p}")
    excl = {re.sub(r'[^a-z0-9]', '', w.lower()) for w in cfg.exclude_wells}
    frames: Dict[str, pd.DataFrame] = {}
    for f in files:
        well = re.sub(r"_processed$", "", f.stem)
        if re.sub(r'[^a-z0-9]', '', well.lower()) in excl:
            continue
        if well.lower().startswith(("combined", "global", "all_wells")):
            continue                                   # skip combined files
        df = pd.read_csv(f)
        wc = next((c for c in df.columns if c.upper() in ("WELL", "WELLNAME")), None)
        if wc is not None and df[wc].nunique() > 1:
            continue                                   # skip multi-well files
        frames[well] = _finalise(df, well, cfg)
    print(f"[data] loaded {len(frames)} wells: {sorted(frames)}")
    return frames


# =============================================================================
# 3. FEATURE ENGINEERING + COMMON FEATURE SPACE
# =============================================================================
def engineer_features(frames: Dict[str, pd.DataFrame],
                      cfg: Config) -> Dict[str, pd.DataFrame]:
    """Physics features + label-free per-well normalisation (no leakage)."""
    out = {}
    for well, df in frames.items():
        d = df.copy()
        # shale volume from GR (Larionov tertiary), per-well 5/95 percentiles
        if "GR" in d and d["GR"].notna().sum() >= cfg.min_well_rows:
            gr = d["GR"]
            lo, hi = np.nanpercentile(gr, 5), np.nanpercentile(gr, 95)
            igr = ((gr - lo) / max(hi - lo, 1e-6)).clip(0, 1)
            d["VSH_GR"] = (0.083 * (2 ** (3.7 * igr) - 1)).clip(0, 1)
        # porosity-space transforms
        if "RHOB" in d:
            d["PHID"] = ((cfg.rho_matrix - d["RHOB"]) /
                         max(cfg.rho_matrix - cfg.rho_fluid, 1e-6)).clip(-0.15, 0.75)
        if "DT" in d:
            d["PHIS"] = ((d["DT"] - cfg.dt_matrix) /
                         max(cfg.dt_fluid - cfg.dt_matrix, 1e-6)).clip(-0.15, 0.75)
        if {"PHID", "NPHI"} <= set(d.columns):
            d["PHIDN"] = (d["PHID"] + d["NPHI"]) / 2
            d["ND_SEP"] = d["NPHI"] - d["PHID"]
        if "RT_DEEP" in d:
            d["LOG_RT"] = np.log10(d["RT_DEEP"].clip(lower=0.05))
        if "RT_SHAL" in d:
            d["LOG_RS"] = np.log10(d["RT_SHAL"].clip(lower=0.05))
        if "RXO" in d:
            d["LOG_RXO"] = np.log10(d["RXO"].clip(lower=0.05))
        # relative depth within the well
        idx = d.index.to_numpy(float)
        d["REL_DEPTH"] = (idx - idx.min()) / max(idx.max() - idx.min(), 1e-9)
        # label-free per-well robust normalisation of tool-unit features
        for c in ("GR", "SP", "LOG_RT", "LOG_RS", "LOG_RXO", "DRHO"):
            if c in d and d[c].notna().sum() >= cfg.min_well_rows:
                s = d[c].dropna()
                iqr = float(s.quantile(0.75) - s.quantile(0.25))
                scale = iqr / 1.349 if iqr > 1e-9 else 1.0
                d[f"{c}_N"] = (d[c] - float(s.median())) / scale
        d.attrs["well"] = well
        out[well] = d
    return out


# candidate raw-log physics features, in preference order
_CANDIDATE_FEATS = ["PHID", "PHIS", "PHIDN", "ND_SEP", "NPHI",
                    "GR_N", "LOG_RT_N", "LOG_RS_N", "LOG_RXO_N", "SP_N",
                    "DRHO_N", "VSH_GR", "REL_DEPTH", "PE"]


def resolve_common_features(frames: Dict[str, pd.DataFrame],
                            cfg: Config) -> List[str]:
    """Features present (>=min_well_rows valid) in EVERY well and not leaky."""
    leak = {c.upper() for c in cfg.leakage_cols}
    feats = []
    for f in _CANDIDATE_FEATS:
        if f.upper() in leak:
            continue
        if all(f in d.columns and d[f].notna().sum() >= cfg.min_well_rows
               for d in frames.values()):
            feats.append(f)
    if not feats:
        raise RuntimeError("No common raw-log features across all wells.")
    print(f"[features] common raw-log feature set ({len(feats)}): {feats}")
    return feats


def _matrix(df: pd.DataFrame, feats: List[str], target: str
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows with all features and the target present. Returns X, y, row-mask."""
    if target not in df.columns or any(f not in df.columns for f in feats):
        return np.empty((0, len(feats))), np.empty(0), np.zeros(len(df), bool)
    m = df[target].notna() & df[feats].notna().all(axis=1)
    return (df.loc[m, feats].to_numpy(float),
            df.loc[m, target].to_numpy(float), m.to_numpy())


# =============================================================================
# 4. METRICS
# =============================================================================
def metrics(a: np.ndarray, p: np.ndarray) -> dict:
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) < 5:
        return dict(n=len(a), R2=np.nan, RMSE=np.nan, MAE=np.nan, bias=np.nan)
    return dict(n=int(len(a)), R2=float(r2_score(a, p)),
                RMSE=float(np.sqrt(mean_squared_error(a, p))),
                MAE=float(mean_absolute_error(a, p)),
                bias=float(np.mean(p - a)))


# =============================================================================
# 5. HYBRID AQPSO-GWO OPTIMIZER
# =============================================================================
class HybridAQPSOGWO:
    """Hybrid Adaptive Quantum-PSO / Grey Wolf Optimizer (minimizer).

    A cosine-weighted, time-varying fraction of the swarm follows AQPSO
    dynamics (exploration, early) and the remainder follows GWO dynamics
    (exploitation, late), giving an adaptive exploration->exploitation
    transition. Minimizes a scalar objective over a box-bounded space.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]],
                 n_agents: int = 16, max_iter: int = 10, patience: int = 3,
                 beta_range: Tuple[float, float] = (0.5, 1.0), seed: int = 42,
                 mode: str = "hybrid"):   # 'aqpso' | 'gwo' | 'hybrid'
        self.lb = np.array([b[0] for b in bounds], float)
        self.ub = np.array([b[1] for b in bounds], float)
        self.dim = len(bounds)
        self.N, self.T = n_agents, max_iter
        self.patience, self.beta_range, self.seed = patience, beta_range, seed
        self.mode = mode

    def optimize(self, objective: Callable[[np.ndarray], float]):
        rng = np.random.default_rng(self.seed)
        X = rng.uniform(self.lb, self.ub, (self.N, self.dim))
        fit = np.array([objective(x) for x in X])
        pbest, pbest_fit = X.copy(), fit.copy()
        g = int(np.argmin(fit))
        gbest, gbest_fit = X[g].copy(), float(fit[g])
        history, no_improve = [gbest_fit], 0

        for t in range(self.T):
            beta = self.beta_range[1] - (self.beta_range[1] - self.beta_range[0]) * t / max(self.T, 1)
            a = 2.0 - 2.0 * t / max(self.T, 1)                     # GWO coefficient
            if self.mode == "aqpso":
                n_aqpso = self.N                     # all agents AQPSO
            elif self.mode == "gwo":
                n_aqpso = 0                          # all agents GWO
            else:
                frac = 0.5 * (1 + np.cos(np.pi * t / max(self.T, 1)))  # AQPSO share 1->0
                n_aqpso = int(round(frac * self.N))
            mbest = pbest.mean(axis=0)

            order = np.argsort(fit)          # best first
            leaders = X[order[:3]]           # alpha, beta, delta for GWO
            aqpso_ids = set(order[:n_aqpso].tolist())   # top fraction -> AQPSO

            Xnew = X.copy()
            for i in range(self.N):
                if i in aqpso_ids:            # ---- AQPSO update ----
                    phi = rng.random(self.dim)
                    u = np.clip(rng.random(self.dim), 1e-12, 1.0)
                    p = phi * pbest[i] + (1 - phi) * gbest
                    sign = np.where(rng.random(self.dim) < 0.5, 1.0, -1.0)
                    Xnew[i] = p + sign * beta * np.abs(mbest - X[i]) * np.log(1.0 / u)
                else:                         # ---- GWO update ----
                    parts = []
                    for L in leaders:
                        r1, r2 = rng.random(self.dim), rng.random(self.dim)
                        A, C = 2 * a * r1 - a, 2 * r2
                        parts.append(L - A * np.abs(C * L - X[i]))
                    Xnew[i] = np.mean(parts, axis=0)
            X = np.clip(Xnew, self.lb, self.ub)
            fit = np.array([objective(x) for x in X])

            better = fit < pbest_fit
            pbest[better], pbest_fit[better] = X[better], fit[better]
            gi = int(np.argmin(pbest_fit))
            if pbest_fit[gi] < gbest_fit - 1e-6:
                gbest, gbest_fit, no_improve = pbest[gi].copy(), float(pbest_fit[gi]), 0
            else:
                no_improve += 1
            history.append(gbest_fit)
            if no_improve >= self.patience:
                break
        return gbest, gbest_fit, history


# =============================================================================
# 6. MODELS: bounds, decoders, factory (RF, XGBoost, SimpleMLP, OptimizedMLP)
# =============================================================================
# Hyperparameter search spaces as bounded continuous vectors (decoded below).
BOUNDS = {
    "random_forest": [(100, 600), (4, 24), (1, 20), (0.3, 1.0)],
    "xgboost":       [(100, 700), (2, 8), (-2.0, -0.5), (0.5, 1.0)],   # lr as log10
    "simple_mlp":    [(-4.0, -2.0), (0.0, 0.5), (-6.0, -2.0), (0.5, 2.0)],  # log-lr, drop, log-l2, width
    "optimized_mlp": [(-4.0, -2.0), (0.0, 0.5), (-6.0, -2.0),
                      (0.5, 2.0), (0.5, 2.0), (0.5, 2.0)],
}


def _decode(model: str, hp: np.ndarray) -> dict:
    if model == "random_forest":
        return dict(n_estimators=int(round(hp[0])), max_depth=int(round(hp[1])),
                    min_samples_leaf=int(round(hp[2])),
                    max_features=float(np.clip(hp[3], 0.1, 1.0)))
    if model == "xgboost":
        return dict(n_estimators=int(round(hp[0])), max_depth=int(round(hp[1])),
                    learning_rate=float(10 ** hp[2]), subsample=float(hp[3]))
    if model == "simple_mlp":
        return dict(lr=float(10 ** hp[0]), dropout=float(hp[1]),
                    weight_decay=float(10 ** hp[2]),
                    widths=[int(64 * hp[3]), int(32 * hp[3])])
    if model == "optimized_mlp":
        return dict(lr=float(10 ** hp[0]), dropout=float(hp[1]),
                    weight_decay=float(10 ** hp[2]),
                    widths=[int(64 * hp[3]), int(32 * hp[4]), int(16 * hp[5])])
    raise ValueError(model)


class _MLP:
    """SimpleMLP / OptimizedMLP wrapper. Uses PyTorch when available, else
    falls back to sklearn's MLPRegressor with a matching architecture."""

    def __init__(self, widths, lr=1e-3, dropout=0.3, weight_decay=1e-3,
                 batchnorm=False, epochs=80, seed=42):
        self.widths, self.lr, self.dropout = widths, lr, dropout
        self.weight_decay, self.batchnorm = weight_decay, batchnorm
        self.epochs, self.seed = epochs, seed

    def fit(self, X, y):
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32).ravel()
        if _HAS_TORCH:
            torch.manual_seed(self.seed)
            layers, d = [], X.shape[1]
            for w in self.widths:
                layers += [nn.Linear(d, w)]
                if self.batchnorm:
                    layers += [nn.BatchNorm1d(w)]
                layers += [nn.ReLU(), nn.Dropout(self.dropout)]
                d = w
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)
            opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
            lossf = nn.MSELoss()
            n = len(X); idx = np.arange(n)
            rng = np.random.default_rng(self.seed)
            cut = int(0.85 * n)
            Xt = torch.tensor(X); yt = torch.tensor(y).view(-1, 1)
            best, best_state, bad, bs = np.inf, None, 0, 256
            for ep in range(self.epochs):
                self.net.train(); rng.shuffle(idx)
                for s in range(0, cut, bs):
                    b = idx[s:s + bs]
                    if len(b) < 2:
                        continue
                    opt.zero_grad()
                    loss = lossf(self.net(Xt[b]), yt[b]); loss.backward(); opt.step()
                self.net.eval()
                with torch.no_grad():
                    v = lossf(self.net(Xt[idx[cut:]]), yt[idx[cut:]]).item() if cut < n else loss.item()
                if v < best - 1e-6:
                    best, best_state, bad = v, {k: p.clone() for k, p in self.net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 12:
                        break
            if best_state:
                self.net.load_state_dict(best_state)
        else:
            from sklearn.neural_network import MLPRegressor
            self.net = MLPRegressor(hidden_layer_sizes=tuple(self.widths),
                                    alpha=self.weight_decay,
                                    learning_rate_init=self.lr, max_iter=300,
                                    early_stopping=True, random_state=self.seed)
            self.net.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, np.float32)
        if _HAS_TORCH:
            self.net.eval()
            with torch.no_grad():
                return self.net(torch.tensor(X)).numpy().ravel()
        return self.net.predict(X)


def build_model(model: str, hp: np.ndarray, seed: int):
    """Return an unfitted estimator with .fit/.predict for the decoded hp."""
    k = _decode(model, hp)
    if model == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **k)
    if model == "xgboost":
        if _HAS_XGB:
            return XGBRegressor(random_state=seed, tree_method="hist",
                                colsample_bytree=0.8, reg_lambda=5.0,
                                n_jobs=-1, **k)
        # faithful-enough fallback when xgboost is absent
        return GradientBoostingRegressor(
            random_state=seed, n_estimators=k["n_estimators"],
            max_depth=k["max_depth"], learning_rate=k["learning_rate"],
            subsample=k["subsample"])
    if model in ("simple_mlp", "optimized_mlp"):
        return _MLP(widths=k["widths"], lr=k["lr"], dropout=k["dropout"],
                    weight_decay=k["weight_decay"],
                    batchnorm=(model == "optimized_mlp"), seed=seed)
    raise ValueError(model)


# =============================================================================
# 7. INNER GROUPED-CV OBJECTIVE (well-aware; no leakage)
# =============================================================================
def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def make_objective(model: str, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, cfg: Config) -> Callable:
    """Negative mean grouped-CV R^2 on the ORIGINAL scale. Groups = wells, so
    every inner validation fold is a set of whole wells the inner-train never
    saw -> the tuning objective is itself leakage-free."""
    n_groups = len(np.unique(groups))
    n_splits = int(min(cfg.inner_folds, n_groups))
    gkf = GroupKFold(n_splits=max(2, n_splits))

    def objective(hp: np.ndarray) -> float:
        scores = []
        for tr, va in gkf.split(X, y, groups):
            Xtr, Xva = _standardise(X[tr], X[va])
            try:
                m = build_model(model, hp, cfg.seed).fit(Xtr, y[tr])
                p = np.clip(m.predict(Xva), *cfg.phi_clamp)
                scores.append(r2_score(y[va], p))
            except Exception:
                scores.append(-1.0)
        return -float(np.mean(scores))     # minimise
    return objective


# =============================================================================
# 8. OUTER LOWO RUNNER (per target, per model)
# =============================================================================
def _stack(train_frames, feats, target, cfg, rng):
    """Assemble training matrix from whole wells, with optional subsampling."""
    Xs, ys, gs = [], [], []
    for gi, (w, df) in enumerate(sorted(train_frames.items())):
        X, y, _ = _matrix(df, feats, target)
        if len(y) < cfg.min_well_rows:
            continue
        if cfg.max_train_per_well and len(y) > cfg.max_train_per_well:
            sel = rng.choice(len(y), cfg.max_train_per_well, replace=False)
            X, y = X[sel], y[sel]
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), gi))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def run_lowo_for_target(frames, feats, target: str, model: str,
                        cfg: Config) -> dict:
    """LOWO over all wells for one (target, model): tune with AQPSO-GWO on the
    training wells (grouped inner CV), refit on all training wells, predict the
    held-out well. Returns per-well metrics, pooled metrics, and predictions."""
    print(f"\n[LOWO] target={target}  model={model}")
    rng = np.random.default_rng(cfg.seed)
    wells = sorted(frames)
    preds, acts, rows, best_hps = {}, {}, [], {}

    for held in wells:
        train = {w: d for w, d in frames.items() if w != held}
        Xtr, ytr, gtr = _stack(train, feats, target, cfg, rng)
        if len(np.unique(gtr)) < 2:
            continue
        # ---- inner tuning (grouped, leakage-free) ----
        obj = make_objective(model, Xtr, ytr, gtr, cfg)
        opt = HybridAQPSOGWO(BOUNDS[model], n_agents=cfg.opt_agents,
                             max_iter=cfg.opt_iters, patience=cfg.opt_patience,
                             seed=cfg.seed)
        best_hp, best_neg_r2, _ = opt.optimize(obj)
        best_hps[held] = _decode(model, best_hp)
        # ---- refit on all training wells, predict held-out well ----
        Xte, yte, mask = _matrix(frames[held], feats, target)
        if mask.sum() < cfg.min_well_rows:
            continue
        Xtr_s, Xte_s = _standardise(Xtr, Xte)
        m = build_model(model, best_hp, cfg.seed).fit(Xtr_s, ytr)
        yhat = np.clip(m.predict(Xte_s), *cfg.phi_clamp)
        idx = frames[held].index[mask]
        preds[held] = pd.Series(yhat, index=idx)
        acts[held] = pd.Series(yte, index=idx)
        r = dict(well=held, **metrics(yte, yhat))
        rows.append(r)
        print(f"[LOWO]   {held:16s} R2={r['R2']:+.3f} RMSE={r['RMSE']:.4f} "
              f"bias={r['bias']:+.4f} n={r['n']} (innerCV R2={-best_neg_r2:.3f})")

    per_well = pd.DataFrame(rows)
    if preds:
        A = np.concatenate([acts[w].to_numpy() for w in preds])
        P = np.concatenate([preds[w].to_numpy() for w in preds])
        per_well = pd.concat([per_well, pd.DataFrame([dict(well="OVERALL",
                                                           **metrics(A, P))])],
                             ignore_index=True)
        ov = per_well[per_well.well == "OVERALL"].iloc[0]
        print(f"[LOWO]   POOLED  R2={ov.R2:+.3f}  RMSE={ov.RMSE:.4f}")
    return dict(target=target, model=model, per_well=per_well,
                predictions=preds, actuals=acts, best_hps=best_hps)


# =============================================================================
# 9. COMPOSITE PHIE = PHIT x (1 - VCL)
# =============================================================================
def assemble_composite(res_phit: dict, res_vcl: dict, frames, cfg) -> dict:
    """Combine independent blind PHIT and VCL predictions into PHIE and score
    against the ACTUAL PHIE on each well. Both component models were LOWO-blind,
    so the composite is a genuine blind PHIE estimate that never used PHIT/VCL
    as inputs."""
    print(f"\n[COMPOSITE] PHIE = PHIT x (1 - VCL)  model={res_phit['model']}")
    preds, acts, rows = {}, {}, []
    for w in sorted(set(res_phit["predictions"]) & set(res_vcl["predictions"])):
        phit_hat = res_phit["predictions"][w]
        vcl_hat = res_vcl["predictions"][w].reindex(phit_hat.index)
        phie_hat = (phit_hat * (1 - vcl_hat)).clip(*cfg.phi_clamp)
        if "PHIE" not in frames[w].columns:
            continue
        phie_true = frames[w]["PHIE"].reindex(phit_hat.index)
        m = phie_hat.notna() & phie_true.notna()
        if m.sum() < cfg.min_well_rows:
            continue
        preds[w] = phie_hat[m]; acts[w] = phie_true[m]
        r = dict(well=w, **metrics(phie_true[m].to_numpy(), phie_hat[m].to_numpy()))
        rows.append(r)
        print(f"[COMPOSITE] {w:16s} R2={r['R2']:+.3f} RMSE={r['RMSE']:.4f} n={r['n']}")
    per_well = pd.DataFrame(rows)
    if preds:
        A = np.concatenate([acts[w].to_numpy() for w in preds])
        P = np.concatenate([preds[w].to_numpy() for w in preds])
        per_well = pd.concat([per_well, pd.DataFrame([dict(well="OVERALL",
                                                           **metrics(A, P))])],
                             ignore_index=True)
        ov = per_well[per_well.well == "OVERALL"].iloc[0]
        print(f"[COMPOSITE] POOLED PHIE  R2={ov.R2:+.3f}  RMSE={ov.RMSE:.4f}")
    return dict(target="PHIE_composite", model=res_phit["model"],
                per_well=per_well, predictions=preds, actuals=acts)


# =============================================================================
# 10. ORCHESTRATION
# =============================================================================
def run(cfg: Config) -> dict:
    out = Path(cfg.out_dir); (out / "tables").mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(out / "config_used.json"))
    if not _HAS_XGB and "xgboost" in cfg.models:
        print("[warn] xgboost not installed -> GradientBoosting fallback for 'xgboost'")
    if not _HAS_TORCH and any("mlp" in m for m in cfg.models):
        print("[warn] torch not installed -> sklearn MLPRegressor fallback for MLPs")

    frames = engineer_features(load_wells(cfg), cfg)
    feats = resolve_common_features(frames, cfg)

    results: Dict[str, dict] = {}
    # which base targets do we need to actually regress?
    if cfg.target == "PHIE_composite":
        base_targets = ["PHIT", "VCL"]
    elif cfg.target in ("PHIT", "VCL", "PHIE_direct"):
        base_targets = ["PHIE" if cfg.target == "PHIE_direct" else cfg.target]
    else:
        raise ValueError(f"Unknown target '{cfg.target}'")

    for model in cfg.models:
        for bt in base_targets:
            key = f"{cfg.target}::{model}::{bt}"
            results[key] = run_lowo_for_target(frames, feats, bt, model, cfg)

        if cfg.target == "PHIE_composite":
            comp = assemble_composite(results[f"{cfg.target}::{model}::PHIT"],
                                      results[f"{cfg.target}::{model}::VCL"],
                                      frames, cfg)
            results[f"{cfg.target}::{model}::COMPOSITE"] = comp

    # ---- write tables ----
    summary_rows = []
    for key, res in results.items():
        res["per_well"].round(5).to_csv(
            out / "tables" / f"metrics_{key.replace('::', '__')}.csv", index=False)
        ov = res["per_well"][res["per_well"].well == "OVERALL"]
        if len(ov):
            summary_rows.append(dict(target=cfg.target, model=res["model"],
                                     regressed=key.split("::")[-1],
                                     **{k: ov[k].iloc[0] for k in
                                        ("n", "R2", "RMSE", "MAE", "bias")}))
    summary = pd.DataFrame(summary_rows)
    summary.round(5).to_csv(out / "tables" / "SUMMARY.csv", index=False)
    print("\n[summary]\n" + summary.to_string(index=False))

    pack = dict(frames=frames, features=feats, results=results,
                summary=summary, out_dir=out, cfg=cfg)
    if cfg.make_figures:
        try:
            _figures(pack)
        except Exception as e:
            print(f"[figures] skipped ({e})")
    print(f"\n[done] outputs in {out}")
    return pack


# =============================================================================
# 11. FIGURES (compact; the earlier phie_figures.py can also consume the CSVs)
# =============================================================================
def _figures(pack: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = pack["cfg"]; out = Path(pack["out_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})

    # (a) model comparison: pooled R2 per regressed target
    s = pack["summary"]
    if len(s):
        fig, ax = plt.subplots(figsize=(7, 4))
        piv = s.pivot_table(index="regressed", columns="model", values="R2")
        piv.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("pooled blind (LOWO) R$^2$")
        ax.set_title(f"Target = {cfg.target}: pooled LOWO R$^2$ by model")
        ax.axhline(0, color="k", lw=0.8); ax.legend(title="", fontsize=8)
        fig.savefig(out / "F_model_comparison.png", dpi=cfg.dpi)
        fig.savefig(out / "F_model_comparison.pdf"); plt.close(fig)

    # (b) predicted vs actual for the best result (highest pooled R2)
    best_key = max(pack["results"],
                   key=lambda k: (pack["results"][k]["per_well"]
                                  .query("well=='OVERALL'")["R2"].max()
                                  if (pack["results"][k]["per_well"].well == "OVERALL").any()
                                  else -9))
    res = pack["results"][best_key]
    A = np.concatenate([res["actuals"][w].to_numpy() for w in res["predictions"]])
    P = np.concatenate([res["predictions"][w].to_numpy() for w in res["predictions"]])
    ov = res["per_well"].query("well=='OVERALL'").iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.hexbin(A, P, gridsize=55, cmap="Blues", mincnt=1)
    hi = max(0.45, np.percentile(A, 99.5)) * 1.05
    ax.plot([0, hi], [0, hi], "k-", lw=1.1)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    ax.set_xlabel("Measured (v/v)"); ax.set_ylabel("Predicted (v/v)")
    ax.set_title(f"{best_key}\nR2={ov.R2:.3f} RMSE={ov.RMSE:.4f} n={int(ov.n):,}")
    fig.savefig(out / "F_best_pred_vs_actual.png", dpi=cfg.dpi)
    fig.savefig(out / "F_best_pred_vs_actual.pdf"); plt.close(fig)
    print(f"[figures] written to {out}")


# =============================================================================
# 12. CLI
# =============================================================================
def main(config_path: Optional[str] = None) -> dict:
    cfg = Config.from_yaml(config_path) if config_path else Config()
    return run(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AQPSO-GWO + LOWO porosity pipeline")
    ap.add_argument("--config", help="optional YAML config")
    ap.add_argument("--input_dir", help="folder of *_processed.csv")
    ap.add_argument("--target", choices=["PHIT", "VCL", "PHIE_direct", "PHIE_composite"])
    ap.add_argument("--out_dir")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.target:
        cfg.target = args.target
    if args.out_dir:
        cfg.out_dir = args.out_dir
    run(cfg)
