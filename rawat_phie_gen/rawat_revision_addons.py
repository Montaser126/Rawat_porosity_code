#!/usr/bin/env python3
# =============================================================================
# rawat_revision_addons.py — the analyses added in the v8 revision, run on YOUR
# raw + calculated LAS files through rawat_full_pipeline (same merge, same
# features, same eight wells), so every number in the paper comes from one
# data source.
#
# Produces, in <out>/revision/:
#   Table_10_rawlog_only.csv          raw-log-only random split vs LOWO  (manuscript Table 10)
#   Table_S7_fewshot_contiguous.csv   random vs contiguous 10% few-shot   (manuscript Table S7)
#   Table_09_ablation_isolated.csv    ablation that changes ONE design choice per stage
#   Table_S4_outlier_screen.csv       MAD Modified-Z screen               (manuscript Table S4)
#   Table_S5_correlation_pairs.csv    pairwise correlations               (manuscript Table S5)
#   Fig_02_mad_outlier_screen.png/pdf (manuscript Fig. 2)
#   Fig_03_correlation_matrix.png/pdf (manuscript Fig. 3)
#   revision_numbers.json             ridge pooled LOWO R² (placeholder in §4.3(d)),
#                                     per-well mean R² ± SD and well-bootstrap 95% CI (Table 8 text),
#                                     outlier counts at τ = 3.0 and 3.5
#
# Keep this file in the same folder as rawat_full_pipeline.py,
# regime_a_benchmark.py and phie_aqpso_lowo.py.
#
# Anaconda Prompt:
#   python rawat_revision_addons.py --las-raw "D:\...\Las files" ^
#       --las-calc "D:\...\the_9_wells" --out "D:\rawat_local\results_all"
# Add --selected VCL,PHIT,PHIE_DERIVED to skip feature selection (faster).
# =============================================================================
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd

import rawat_full_pipeline as B
import regime_a_benchmark as RA

try:
    import xgboost as xgb; _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# raw tool curves (conventional configuration) and the raw-log physics pool
RAW_TOOL = ["GR", "RHOB", "NPHI", "DT", "PE", "RT_DEEP", "RT_SHAL", "RXO", "SP", "REL_DEPTH"]
RAWLOG_POOL = ["PHID", "PHIS", "PHIDN", "ND_SEP", "NPHI", "VSH_GR",
               "GR_N", "LOG_RD_N", "LOG_RS_N", "SP_N", "REL_DEPTH"]


def conventional_model(cfg):
    """High-capacity booster on raw tool units, as in the conventional workflow."""
    if _HAS_XGB:
        return xgb.XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                random_state=cfg.seed, n_jobs=-1)
    return B.make_booster(cfg, high_capacity=True)


def per_well_summary(res, seed):
    pw = res["per_well"]; r2 = pw.loc[pw.well != "OVERALL", "R2"].to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(r2, len(r2)).mean() for _ in range(5000)]
    return dict(pooled_R2=round(res["pooled"]["R2"], 4), mean_well_R2=round(float(r2.mean()), 4),
                sd_well_R2=round(float(r2.std(ddof=1)), 4),
                well_bootstrap_ci95=[round(float(np.percentile(bs, 2.5)), 4),
                                     round(float(np.percentile(bs, 97.5)), 4)],
                min_well_R2=round(float(r2.min()), 4), max_well_R2=round(float(r2.max()), 4))


def fewshot_compare(res, frac=0.10, n_random=5, n_contig=5):
    """Affine recalibration on 10% of each blind well: scattered random samples
    (5 seeds) vs one contiguous depth interval at 5 positions; scored on the rest."""
    rows = []
    for w in sorted(res["predictions"]):
        p = res["predictions"][w].sort_index(); a = res["actuals"][w].reindex(p.index)
        p, a = p.to_numpy(float), a.to_numpy(float); n = len(p); k = max(10, int(frac * n))

        def score(mask):
            A = np.polyfit(p[mask], a[mask], 1)
            return B.metrics(a[~mask], np.polyval(A, p[~mask]))["R2"]
        rnd = []
        for s in range(n_random):
            m = np.zeros(n, bool); m[np.random.default_rng(s).choice(n, k, replace=False)] = True
            rnd.append(score(m))
        con = []
        for st in np.linspace(0, n - k, n_contig).astype(int):
            m = np.zeros(n, bool); m[st:st + k] = True
            con.append(score(m))
        rows.append(dict(well=w, R2_blind=round(B.metrics(a, p)["R2"], 4),
                         R2_random10_mean=round(float(np.mean(rnd)), 4),
                         R2_contig10_mean=round(float(np.mean(con)), 4),
                         R2_contig10_min=round(float(np.min(con)), 4),
                         R2_contig10_max=round(float(np.max(con)), 4)))
    return pd.DataFrame(rows)


def isolated_ablation(frames, selected, cfg):
    """Each stage changes exactly one thing, so per-well normalization and the
    interpreter inputs (PHIT, VCL) are credited separately."""
    have = lambda fs: B.availability_filter(frames, fs, cfg)
    phys_global = have(["PHID", "PHIS", "PHIDN", "ND_SEP", "NPHI", "GR", "LOG_RD", "LOG_RS", "REL_DEPTH"])
    phys_perwell = [f + "_N" if f in ("GR", "LOG_RD", "LOG_RS") else f for f in phys_global]
    phys_perwell = have(phys_perwell)
    hi = lambda: B.make_booster(cfg, high_capacity=True)
    stages = [
        ("A0", have(RAW_TOOL), hi, "raw tool curves, global scaling, high-capacity booster"),
        ("A1", phys_global, hi, "+ physics transforms (PHID, PHIS, PHIDN, ND_SEP)"),
        ("A2", phys_perwell, hi, "+ per-well robust normalization of GR and resistivity"),
        ("A3", have(phys_perwell + ["PHIT"]), hi, "+ interpreter PHIT"),
        ("A4", have(phys_perwell + ["PHIT", "VCL"]), hi, "+ interpreter VCL"),
        ("A5", have(phys_perwell + ["PHIT", "VCL", "PHIE_DERIVED"]), hi, "+ PHIE_DERIVED"),
        ("A6", selected, hi, "LOWO-selected subset"),
        ("A7", selected, lambda: B.make_booster(cfg), "regularized booster (deployed)"),
    ]
    rows = []
    for sid, feats, mf, desc in stages:
        r = B.lowo_eval(frames, feats, cfg, mf)
        worst = r["per_well"].loc[r["per_well"].well != "OVERALL", "R2"].min()
        rows.append(dict(stage=sid, change=desc, n_features=len(feats), features=" ".join(feats),
                         pooled_lowo_R2=round(r["pooled"]["R2"], 4),
                         pooled_RMSE=round(r["pooled"]["RMSE"], 4), worst_well_R2=round(float(worst), 4)))
        print(f"  {sid}: R2={rows[-1]['pooled_lowo_R2']:.4f}  {desc}", flush=True)
    return pd.DataFrame(rows)


def preprocessing_figures(frames, cfg, thr, rev):
    """Manuscript Fig. 2 / Table S4 and Fig. 3 / Table S5, without running the
    optimizer benchmark. Reuses regime_a_benchmark's own functions."""
    feats = [f for f in RA.RAWLOG_FEATS
             if all(f in d.columns and d[f].notna().sum() >= cfg.min_well_rows for d in frames.values())]
    blocks = [B._clean(d, feats, cfg.target) for d in frames.values()]
    Xpool = np.vstack([X for X, y, _ in blocks if len(y) >= cfg.min_well_rows])
    counts = {}
    for t in (3.0, 3.5):
        nf = int((RA._modified_z(Xpool).max(axis=1) > t).sum())
        counts[str(t)] = dict(rows=int(len(Xpool)), flagged=nf, pct=round(100 * nf / len(Xpool), 3))
    tmp = rev / "_ra"; (tmp / "tables").mkdir(parents=True, exist_ok=True); (tmp / "figures").mkdir(exist_ok=True)
    ra = RA.RegimeAConfig(outlier_threshold=thr)
    RA.outlier_diagnostic(Xpool, feats, ra, cfg, tmp, [])
    RA.correlation_diagnostic(Xpool, feats, cfg, tmp, [])
    ren = {"tables/Table_S3_outlier_screen.csv": "Table_S4_outlier_screen.csv",
           "tables/Table_S4_correlation_pairs.csv": "Table_S5_correlation_pairs.csv"}
    for ext in ("png", "pdf"):
        ren[f"figures/Fig_03_mad_outlier_screen.{ext}"] = f"Fig_02_mad_outlier_screen.{ext}"
        ren[f"figures/Fig_04_correlation_matrix.{ext}"] = f"Fig_03_correlation_matrix.{ext}"
    for src, dst in ren.items():
        shutil.move(str(tmp / src), str(rev / dst))
    shutil.rmtree(tmp)
    return feats, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="v8 revision analyses on the LAS data")
    ap.add_argument("--las-raw", required=True); ap.add_argument("--las-calc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["auto", "lightgbm", "gbr"], default="auto")
    ap.add_argument("--selected", help="comma list, e.g. VCL,PHIT,PHIE_DERIVED (skips selection)")
    ap.add_argument("--mad-threshold", type=float, default=3.0, help="τ for Fig. 2 (manuscript: 3.0)")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out); rev = out / "revision"; rev.mkdir(parents=True, exist_ok=True)
    cfg = B.Config(las_raw_dir=a.las_raw, las_calc_dir=a.las_calc, out_dir=str(out), model=a.model)
    print(f"[cfg] backend={B.resolve_backend(cfg)}  conventional model={'XGBoost' if _HAS_XGB else 'booster fallback'}")
    frames = B.engineer(B.load_and_merge(cfg), cfg)
    for d in frames.values():                       # raw-only deterministic reference
        if {"PHIDN", "VSH_GR"} <= set(d.columns):
            d["PHIE_RAW"] = (d["PHIDN"] * (1 - d["VSH_GR"])).clip(*cfg.phi_clamp)
    nums = {}

    # ---- Fig. 2 / 3, Tables S4 / S5 --------------------------------------
    feats_ra, counts = preprocessing_figures(frames, cfg, a.mad_threshold, rev)
    nums["outlier_screen"] = dict(features=feats_ra, by_threshold=counts)
    print(f"[outliers] {counts}")

    # ---- deployed pipeline + ridge ---------------------------------------
    if a.selected:
        selected = [s.strip() for s in a.selected.split(",")]
    else:
        avail = B.availability_filter(frames, B.CANDIDATE_POOL, cfg)
        consistent, _ = B.sign_consistency(frames, avail, cfg)
        selected, _ = B.lowo_sfs(frames, consistent, cfg)
    print(f"[select] {selected}")
    dep = B.lowo_eval(frames, selected, cfg, lambda: B.make_booster(cfg))
    rid = B.lowo_eval(frames, selected, cfg, B.make_ridge)
    nums["selected_features"] = selected
    nums["deployed_lowo"] = per_well_summary(dep, cfg.seed)
    nums["ridge_lowo_selected"] = per_well_summary(rid, cfg.seed)
    print(f"[LOWO] deployed {nums['deployed_lowo']}\n[LOWO] ridge    {nums['ridge_lowo_selected']}")

    # ---- Table 10: raw-log-only ------------------------------------------
    tool = B.availability_filter(frames, RAW_TOOL, cfg)
    pool = B.availability_filter(frames, RAWLOG_POOL, cfg)
    print(f"[raw-log] tool curves {tool}\n[raw-log] physics pool {pool}")
    conv = lambda: conventional_model(cfg); reg = lambda: B.make_booster(cfg)
    rows, gen_lowo = [], None
    for name, feats, mf in [("Conventional: raw curves, high-capacity booster", tool, conv),
                            ("Raw curves, regularized booster", tool, reg),
                            ("Generalization-first raw-log (per-well norm + physics), regularized booster", pool, reg),
                            ("Ridge on the raw-log physics pool", pool, B.make_ridge)]:
        lo = B.lowo_eval(frames, feats, cfg, mf)
        rs = B.random_split_eval(frames, feats, cfg, mf) if mf is not B.make_ridge else None
        s = per_well_summary(lo, cfg.seed)
        rows.append(dict(configuration=name, n_features=len(feats),
                         random_split_R2=round(rs["pooled"]["R2"], 4) if rs else np.nan,
                         lowo_pooled_R2=s["pooled_R2"], lowo_mean_well_R2=s["mean_well_R2"],
                         lowo_worst_well_R2=s["min_well_R2"],
                         gap=round(rs["pooled"]["R2"] - s["pooled_R2"], 4) if rs else np.nan))
        print(f"  {name}: {rows[-1]}", flush=True)
        if name.startswith("Generalization-first"):
            gen_lowo = lo
    A_, P_ = [], []
    for d in frames.values():
        m = d[cfg.target].notna() & d["PHIE_RAW"].notna()
        A_.append(d.loc[m, cfg.target].to_numpy(float)); P_.append(d.loc[m, "PHIE_RAW"].to_numpy(float))
    rows.append(dict(configuration="Deterministic PHIDN x (1 - VSH_GR)", n_features=0,
                     random_split_R2=np.nan,
                     lowo_pooled_R2=round(B.metrics(np.concatenate(A_), np.concatenate(P_))["R2"], 4),
                     lowo_mean_well_R2=np.nan, lowo_worst_well_R2=np.nan, gap=np.nan))
    pd.DataFrame(rows).to_csv(rev / "Table_10_rawlog_only.csv", index=False)

    # ---- Table S7: random vs contiguous few-shot -------------------------
    fs = []
    for label, res in [("deployed (PHIT, VCL inputs)", dep), ("raw-log generalization-first", gen_lowo)]:
        t = fewshot_compare(res); t.insert(0, "model", label); fs.append(t)
    fsdf = pd.concat(fs, ignore_index=True); fsdf.to_csv(rev / "Table_S7_fewshot_contiguous.csv", index=False)
    nums["fewshot_wells_contig_below_random"] = {
        lab: int((g.R2_contig10_mean < g.R2_random10_mean).sum()) for lab, g in fsdf.groupby("model")}
    nums["fewshot_wells_contig_below_blind"] = {
        lab: int((g.R2_contig10_mean < g.R2_blind).sum()) for lab, g in fsdf.groupby("model")}

    # ---- Table 9: isolated ablation --------------------------------------
    if not a.skip_ablation:
        print("[ablation] one change per stage...")
        isolated_ablation(frames, selected, cfg).to_csv(rev / "Table_09_ablation_isolated.csv", index=False)

    json.dump(nums, open(rev / "revision_numbers.json", "w", encoding="utf-8"), indent=2, default=float)
    print(f"[done] revision outputs in {rev}")
    return nums


if __name__ == "__main__":
    main()
