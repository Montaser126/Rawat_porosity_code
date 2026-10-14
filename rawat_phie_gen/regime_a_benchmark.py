#!/usr/bin/env python3
# =============================================================================
# regime_a_benchmark.py — Regime A (within-well optimizer benchmark)
# -----------------------------------------------------------------------------
# The paper's diagnostic protocol: pooled 80/20 RANDOM split of RAW-LOG features
# (no interpreter products), four models (RandomForest, XGBoost, SimpleMLP,
# OptimizedMLP) each tuned by standalone AQPSO, standalone GWO, the hybrid
# AQPSO-GWO, or random search, plus untuned baselines and a density-porosity
# baseline — 21 configurations in all. Produces Tables 10-13 and Fig 14.
#
# Reuses the SAME LAS ingestion/merge as Regime B (rawat_full_pipeline) so both
# regimes run off one data source, and the tested optimizer + model factory from
# phie_aqpso_lowo. Model backends fall back gracefully (XGBoost->GradientBoosting,
# torch MLP->sklearn MLP) when the paper's libraries are absent; the backend used
# is recorded. This regime is WITHIN-WELL and diagnostic only — never deployment
# evidence (see the manuscript).
# =============================================================================
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import rawat_full_pipeline as B      # data ingest/merge/engineer + metrics + Config
import phie_aqpso_lowo as E          # HybridAQPSOGWO, build_model, BOUNDS, _decode, _standardise


@dataclass
class RegimeAConfig:
    test_size: float = 0.20
    n_rs_trials: int = 20            # random-search budget
    opt_agents: int = 8              # metaheuristic swarm
    opt_iters: int = 5
    opt_patience: int = 2
    inner_folds: int = 3             # CV folds for the tuning objective
    max_train_tune: int = 4000       # subsample train rows during TUNING (final fit uses all)
    models: tuple = ("random_forest", "xgboost", "simple_mlp", "optimized_mlp")
    optimizers: tuple = ("AQPSO", "GWO", "Hybrid")
    fast: bool = False               # shrink budgets for quick validation
    outlier_threshold: float = 3.5   # Modified Z-score (MAD) flag threshold
    remove_outliers: bool = False    # if True, drop flagged rows before the benchmark
                                     # (default False: report as diagnostic, keep numbers)


DISPLAY = {"random_forest": "RF", "xgboost": "XGBoost",
           "simple_mlp": "SimpleMLP", "optimized_mlp": "OptimizedMLP"}
# raw-log physics features only (interpreter products excluded, by design)
RAWLOG_FEATS = ["PHID", "PHIS", "PHIDN", "ND_SEP", "NPHI",
                "GR_N", "LOG_RD_N", "LOG_RS_N", "REL_DEPTH"]


def _pooled_split(frames, feats, cfg: B.Config, ra: RegimeAConfig):
    """Pooled random 80/20 split; returns train/test arrays + per-test-row well."""
    rng = np.random.default_rng(cfg.seed)
    Xtr, ytr, Xte, yte, wte = [], [], [], [], []
    for w, df in sorted(frames.items()):
        X, y, _ = B._clean(df, feats, cfg.target)
        if len(y) < cfg.min_well_rows:
            continue
        o = np.arange(len(y)); rng.shuffle(o); cut = int((1 - ra.test_size) * len(y))
        Xtr.append(X[o[:cut]]); ytr.append(y[o[:cut]])
        Xte.append(X[o[cut:]]); yte.append(y[o[cut:]]); wte += [w] * (len(y) - cut)
    return (np.vstack(Xtr), np.concatenate(ytr),
            np.vstack(Xte), np.concatenate(yte), np.array(wte))


def _cv_objective(model, Xtr, ytr, cfg: B.Config, ra: RegimeAConfig):
    kf = KFold(n_splits=max(2, ra.inner_folds), shuffle=True, random_state=cfg.seed)
    def obj(hp):
        sc = []
        for a, b in kf.split(Xtr):
            Xa, Xb = E._standardise(Xtr[a], Xtr[b])
            try:
                m = E.build_model(model, hp, cfg.seed).fit(Xa, ytr[a])
                sc.append(r2_score(ytr[b], np.clip(m.predict(Xb), *cfg.phi_clamp)))
            except Exception:
                sc.append(-1.0)
        return -float(np.mean(sc))
    return obj


def _fit_score(model, hp, Xtr, ytr, Xte, yte, cfg: B.Config):
    Xtr_s, mu = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-12), Xtr.mean(0)
    sd = Xtr.std(0) + 1e-12
    m = E.build_model(model, hp, cfg.seed).fit((Xtr - mu) / sd, ytr)
    ptr = np.clip(m.predict((Xtr - mu) / sd), *cfg.phi_clamp)
    pte = np.clip(m.predict((Xte - mu) / sd), *cfg.phi_clamp)
    return ptr, pte, r2_score(ytr, ptr), r2_score(yte, pte)



def _modified_z(X):
    """Per-feature Modified Z-score (Iglewicz & Hoaglin 1993): 0.6745*(x-med)/MAD."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0); mad[mad == 0] = 1e-9
    return 0.6745 * np.abs(X - med) / mad


def outlier_diagnostic(Xpool, feats, ra, cfg, out_dir, manifest):
    """MAD/Modified-Z outlier screening (manuscript Fig 3 + counts). Reports how
    many rows exceed the threshold; does not delete unless ra.remove_outliers."""
    import matplotlib.pyplot as plt
    Z = _modified_z(Xpool)
    thr = ra.outlier_threshold
    per_feat = (Z > thr).sum(axis=0)
    row_max = Z.max(axis=1)
    flagged = row_max > thr
    n, nf = len(Xpool), int(flagged.sum())
    print(f"[A][outliers] Modified-Z>{thr}: {nf}/{n} rows flagged ({100*nf/n:.2f}%); "
          f"{'REMOVED' if ra.remove_outliers else 'retained (diagnostic only)'}")

    # summary table
    tbl = pd.DataFrame({"feature": feats,
                        "n_flagged": per_feat.astype(int),
                        "pct_flagged": (100 * per_feat / n).round(3)}
                       ).sort_values("n_flagged", ascending=False)
    tbl = pd.concat([tbl, pd.DataFrame([{"feature": "ANY feature (rows)",
                                         "n_flagged": nf,
                                         "pct_flagged": round(100 * nf / n, 3)}])],
                    ignore_index=True)
    tbl.to_csv(out_dir / "tables" / "Table_S3_outlier_screen.csv", index=False)
    manifest.append(dict(file="tables/Table_S3_outlier_screen.csv",
                         paper_label="Table S3",
                         caption=f"MAD Modified-Z outlier screening (threshold {thr})."))

    # Fig 3: distribution of per-row max Modified-Z (before/after) + per-feature bar
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(np.clip(row_max, 0, thr * 4), bins=80, color="#2F5B6C", alpha=0.85)
    ax[0].axvline(thr, color="#B0472F", lw=1.5, ls="--", label=f"threshold = {thr}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("per-row max Modified Z-score")
    ax[0].set_ylabel("row count (log)"); ax[0].legend()
    _status = "removed from Regime A" if ra.remove_outliers else "retained (diagnostic only)"
    ax[0].set_title(f"Outlier screening: {nf}/{n} rows exceed threshold "
                    f"({100*nf/n:.2f}%) \u2014 {_status}")
    order = np.argsort(per_feat)
    ax[1].barh(np.arange(len(feats)), per_feat[order], color="#C9812E")
    ax[1].set_yticks(np.arange(len(feats))); ax[1].set_yticklabels([feats[i] for i in order])
    ax[1].set_xlabel("rows flagged"); ax[1].set_title("Flags by feature")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Fig_03_mad_outlier_screen.png", dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "Fig_03_mad_outlier_screen.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(file="figures/Fig_03_mad_outlier_screen.png",
                         paper_label="Fig. 3", caption="MAD-based outlier screening."))
    return flagged, dict(n_rows=n, n_flagged=nf, pct=round(100 * nf / n, 3))


def correlation_diagnostic(Xpool, feats, cfg, out_dir, manifest, thresh=0.95):
    """Pairwise-correlation screening. Table S4 reports the COMPLETE ranked list
    of feature-pair correlations with the screening outcome (so it is never
    empty even when no pair is removed); Fig 4 is the correlation heatmap."""
    import matplotlib.pyplot as plt
    C = np.corrcoef(Xpool, rowvar=False)
    # recursive elimination of the higher-mean-correlation member of each pair > thresh
    keep = list(range(len(feats))); removed = set()
    while True:
        sub = C[np.ix_(keep, keep)].copy(); np.fill_diagonal(sub, 0)
        i, j = np.unravel_index(np.argmax(np.abs(sub)), sub.shape)
        if abs(sub[i, j]) <= thresh:
            break
        gi, gj = keep[i], keep[j]
        mi = np.abs(C[gi, keep]).mean(); mj = np.abs(C[gj, keep]).mean()
        drop = gi if mi >= mj else gj
        removed.add(feats[drop]); keep.remove(drop)

    # Table S4: every unique pair, ranked by |r|, with the screening decision
    rows = []
    for a in range(len(feats)):
        for b in range(a + 1, len(feats)):
            r = float(C[a, b])
            flagged = abs(r) > thresh
            outcome = (f"exceeds |r|>{thresh}: one member removed" if flagged
                       else "retained (below threshold)")
            rows.append(dict(feature_1=feats[a], feature_2=feats[b],
                             pearson_r=round(r, 3), abs_r=round(abs(r), 3),
                             above_threshold=("yes" if flagged else "no"),
                             screening_outcome=outcome))
    tblS4 = pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    tblS4.to_csv(out_dir / "tables" / "Table_S4_correlation_pairs.csv", index=False)
    n_rm = len(removed)
    manifest.append(dict(file="tables/Table_S4_correlation_pairs.csv", paper_label="Table S4",
        caption=(f"Pairwise correlations among the {len(feats)} candidate features, ranked by "
                 f"|r|. Screening removes one member of any pair with |r| > {thresh}; "
                 f"here {n_rm} feature(s) removed"
                 + (f" ({', '.join(sorted(removed))})." if removed else ", so the set is unchanged."))))
    print(f"[A][corr] Table S4: {len(tblS4)} pairs listed; |r|>{thresh} removed {n_rm}: {sorted(removed)}")

    # Fig 4: correlation heatmap
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(feats), 0.8 + 0.6 * len(feats)))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
    for a in range(len(feats)):
        for b in range(len(feats)):
            ax.text(b, a, f"{C[a,b]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(C[a, b]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r")
    ax.set_title(f"Feature correlation (|r| > {thresh} screened; {n_rm} removed)")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Fig_04_correlation_matrix.png", dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "Fig_04_correlation_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(file="figures/Fig_04_correlation_matrix.png",
                         paper_label="Fig. 4", caption="Feature correlation matrix / elimination."))
    return sorted(removed)


def run_regime_a(cfg: B.Config, ra: RegimeAConfig, out_dir: Path,
                 manifest: list) -> dict:
    print("\n=== REGIME A — within-well optimizer benchmark (diagnostic) ===")
    backend = B.resolve_backend(cfg)
    if ra.fast:
        ra.opt_agents, ra.opt_iters, ra.n_rs_trials, ra.max_train_tune = 5, 2, 6, 1500
    frames = B.engineer(B.load_and_merge(cfg), cfg)
    feats = [f for f in RAWLOG_FEATS
             if all(f in d.columns and d[f].notna().sum() >= cfg.min_well_rows
                    for d in frames.values())]
    print(f"[A] raw-log features: {feats}")
    # ---- preprocessing diagnostics (manuscript Fig 3 / Fig 4) ----
    Xpool = np.vstack([B._clean(d, feats, cfg.target)[0] for d in frames.values()
                       if len(B._clean(d, feats, cfg.target)[1]) >= cfg.min_well_rows])
    flagged_pool, outlier_summary = outlier_diagnostic(Xpool, feats, ra, cfg, out_dir, manifest)
    correlation_diagnostic(Xpool, feats, cfg, out_dir, manifest)
    if ra.remove_outliers:
        # rebuild frames with flagged rows dropped (Regime A only)
        cleaned = {}
        for w, d in frames.items():
            X, y, mask = B._clean(d, feats, cfg.target)
            if len(y) < cfg.min_well_rows:
                cleaned[w] = d; continue
            z = _modified_z(X); keeprows = z.max(axis=1) <= ra.outlier_threshold
            idx = d.index[mask][keeprows]
            cleaned[w] = d.loc[idx] if len(idx) else d
        frames = cleaned
    Xtr, ytr, Xte, yte, wte = _pooled_split(frames, feats, cfg, ra)
    if ra.max_train_tune and len(Xtr) > ra.max_train_tune:
        idx = np.random.default_rng(cfg.seed).choice(len(Xtr), ra.max_train_tune, replace=False)
        Xtr_t, ytr_t = Xtr[idx], ytr[idx]
    else:
        Xtr_t, ytr_t = Xtr, ytr
    print(f"[A] pooled split: train={len(Xtr)} test={len(Xte)} (tuning on {len(Xtr_t)})")

    rows, conv, best_pred_by_well = [], {}, {}

    def record(model, config_label, hp, history=None):
        ptr, pte, tr_r2, te_r2 = _fit_score(model, hp, Xtr, ytr, Xte, yte, cfg)
        mm = B.metrics(yte, pte)
        rows.append(dict(Model=DISPLAY[model], Config=config_label,
                         Train_R2=round(tr_r2, 4), Test_R2=round(te_r2, 4),
                         RMSE=round(mm["RMSE"], 4), MAE=round(mm["MAE"], 4),
                         Overfit_Gap=round(tr_r2 - te_r2, 4)))
        if history is not None:
            conv[f"{DISPLAY[model]}+{config_label}"] = history
        return te_r2, pte

    import time as _t
    for model in ra.models:
        bounds = E.BOUNDS[model]; t_m = _t.time()
        record(model, "Baseline", np.array([np.mean(b) for b in bounds]))
        print(f"[A] {DISPLAY[model]}: baseline done ({_t.time()-t_m:.0f}s)", flush=True)
        rng = np.random.default_rng(cfg.seed)
        obj = _cv_objective(model, Xtr_t, ytr_t, cfg, ra)
        rs = [(obj(hp := rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])), hp)
              for _ in range(ra.n_rs_trials)]
        record(model, f"Random Search ({ra.n_rs_trials})", min(rs, key=lambda t: t[0])[1])
        print(f"[A] {DISPLAY[model]}: random-search done ({_t.time()-t_m:.0f}s)", flush=True)
        for opt_name in ra.optimizers:
            t_o = _t.time()
            mode = {"AQPSO": "aqpso", "GWO": "gwo", "Hybrid": "hybrid"}[opt_name]
            opt = E.HybridAQPSOGWO(bounds, n_agents=ra.opt_agents, max_iter=ra.opt_iters,
                                   patience=ra.opt_patience, seed=cfg.seed, mode=mode)
            best_hp, _, hist = opt.optimize(_cv_objective(model, Xtr_t, ytr_t, cfg, ra))
            te_r2, pte = record(model, opt_name, best_hp, history=hist)
            best_pred_by_well[f"{DISPLAY[model]}+{opt_name}"] = (te_r2, pte)
            print(f"[A]   {DISPLAY[model]}+{opt_name}: test R2={te_r2:.4f} ({_t.time()-t_o:.0f}s)", flush=True)
        print(f"[A] {DISPLAY[model]}: DONE ({_t.time()-t_m:.0f}s total)", flush=True)

    # (4) density-porosity baseline on the pooled test split
    phid_test = []
    rng = np.random.default_rng(cfg.seed)
    for w, df in sorted(frames.items()):
        X, y, mask = B._clean(df, feats, cfg.target)
        if len(y) < cfg.min_well_rows or "PHID" not in df.columns:
            continue
        o = np.arange(len(y)); rng.shuffle(o); cut = int((1 - ra.test_size) * len(y))
        phid = df.loc[mask, "PHID"].to_numpy(float)[o[cut:]]
        phid_test.append(phid)
    if phid_test:
        pd_pred = np.clip(np.concatenate(phid_test), *cfg.phi_clamp)
        mm = B.metrics(yte, pd_pred)
        rows.append(dict(Model="Density Porosity", Config="Physics Formula",
                         Train_R2=np.nan, Test_R2=round(mm["R2"], 4),
                         RMSE=round(mm["RMSE"], 4), MAE=round(mm["MAE"], 4),
                         Overfit_Gap=np.nan))

    comp = pd.DataFrame(rows)
    is_meta = comp["Config"].isin(list(ra.optimizers))

    def emit(df, label, stem, caption):
        df.round(4).to_csv(out_dir / "tables" / f"{stem}.csv", index=False)
        manifest.append(dict(file=f"tables/{stem}.csv", paper_label=label, caption=caption))

    emit(comp[~is_meta].reset_index(drop=True), "Table 10",
         "Table_10_benchmark_baseline_randomsearch",
         "Regime A: baseline and random-search configurations (80/20 split).")
    emit(comp[is_meta][["Model", "Config", "Test_R2", "RMSE", "MAE", "Overfit_Gap"]]
         .rename(columns={"Config": "Optimizer"}).reset_index(drop=True), "Table 11",
         "Table_11_benchmark_metaheuristic", "Regime A: metaheuristic-tuned configurations.")
    emit(comp.reset_index(drop=True), "Table 12", "Table_12_benchmark_all",
         "Regime A: all configurations (complete).")

    # Table 13: per-well test R2 of the winning configuration
    winner_key = max(best_pred_by_well, key=lambda k: best_pred_by_well[k][0])
    _, wpred = best_pred_by_well[winner_key]
    pw = []
    for w in sorted(set(wte)):
        m = wte == w
        pw.append(dict(Well=w, Test_R2=round(r2_score(yte[m], wpred[m]), 4), n=int(m.sum())))
    pw_df = pd.DataFrame(pw)
    pooled = pd.DataFrame([dict(Well="Pooled", Test_R2=round(r2_score(yte, wpred), 4), n=int(len(yte)))])
    t13 = pd.concat([pw_df, pooled], ignore_index=True)
    t13.to_csv(out_dir / "tables" / "Table_13_perwell_benchmark_winner.csv", index=False)
    manifest.append(dict(file="tables/Table_13_perwell_benchmark_winner.csv",
                         paper_label="Table 13",
                         caption=f"Per-well test R2 of the winning configuration ({winner_key})."))

    # Fig 14: optimizer convergence — faceted by model, CV R2 (higher is better)
    models_present = []
    for k in conv:
        m = k.split("+")[0]
        if m not in models_present:
            models_present.append(m)
    colors = {"AQPSO": "#2F5B6C", "GWO": "#C9812E", "Hybrid": "#B0472F"}
    nm = max(1, len(models_present))
    ncol = 2 if nm > 1 else 1
    nrow = int(np.ceil(nm / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    axf = axes.ravel()
    for ax, model in zip(axf, models_present):
        for opt in ("AQPSO", "GWO", "Hybrid"):
            key = f"{model}+{opt}"
            if key in conv:
                h = np.asarray(conv[key], float)
                ax.plot(range(1, len(h) + 1), -h, "o-", ms=3, lw=1.4,
                        color=colors.get(opt, "#888"), label=opt)   # -obj = CV R2
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("iteration"); ax.set_ylabel("best CV R$^2$")
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, title="optimizer")
    for ax in axf[nm:]:
        ax.axis("off")
    fig.suptitle("Regime A optimizer convergence (per base learner; higher is better)",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Fig_14_optimizer_convergence.png", dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "Fig_14_optimizer_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(file="figures/Fig_14_optimizer_convergence.png",
                         paper_label="Fig. 14", caption="Optimizer convergence traces."))

    winner_r2 = float(comp["Test_R2"].max())
    print(f"[A] winner = {winner_key}  test R²={winner_r2:.4f}  (backend={backend})")
    return dict(comparison=comp, winner=winner_key, winner_test_r2=winner_r2,
                per_well=t13, backend=backend)
