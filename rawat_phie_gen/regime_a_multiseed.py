#!/usr/bin/env python3
# =============================================================================
# regime_a_multiseed.py — five-seed, equal-budget optimizer comparison for
# Regime A (manuscript Table 12 and Fig. 14), run on YOUR LAS data with the
# SAME split, features, objective, model and optimizer code as
# regime_a_benchmark.py (Table 11), so Tables 11 and 12 come from one setup.
#
# Methods: random search, TPE (Optuna, if installed), AQPSO, GWO and the hybrid
# AQPSO-GWO from phie_aqpso_lowo.HybridAQPSOGWO. Every method gets at most
# --budget objective evaluations; the metaheuristics may stop earlier (patience).
#
# Outputs in <out>/regime_a_multiseed/:
#   runs.json                    one record per (seed, method); resumable
#   Table_12_multiseed.csv       mean ± SD of CV and test R², range, evaluations
#   Fig_14_optimizer_convergence.png/pdf   best CV R² vs evaluations
#
# Anaconda Prompt (same folder as the other pipeline files):
#   python regime_a_multiseed.py --las-raw "D:\...\Las files" ^
#       --las-calc "D:\...\the_9_wells" --out "D:\rawat_local\results_all"
# Use the same --agents/--iters/--patience/--tune-rows you used for Table 11.
# =============================================================================
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rawat_full_pipeline as B
import regime_a_benchmark as RA
import phie_aqpso_lowo as E

METHODS = ["RandomSearch", "TPE", "AQPSO", "GWO", "Hybrid"]
COLORS = {"RandomSearch": "#7A7A7A", "TPE": "#5B4B8A", "AQPSO": "#2F5B6C",
          "GWO": "#C9812E", "Hybrid": "#B0472F"}


def run_method(name, seed, bounds, obj, a):
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    if name == "RandomSearch":
        rng = np.random.default_rng(seed)
        cand = [rng.uniform(lb, ub) for _ in range(a.budget)]
        vals = [obj(h) for h in cand]
        return cand[int(np.argmin(vals))]
    if name == "TPE":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        st.optimize(lambda t: obj(np.array([t.suggest_float(f"x{i}", l, u)
                                            for i, (l, u) in enumerate(bounds)])), n_trials=a.budget)
        return np.array([st.best_params[f"x{i}"] for i in range(len(bounds))])
    mode = {"AQPSO": "aqpso", "GWO": "gwo", "Hybrid": "hybrid"}[name]
    opt = E.HybridAQPSOGWO(bounds, n_agents=a.agents, max_iter=a.iters,
                           patience=a.patience, seed=seed, mode=mode)
    return opt.optimize(obj)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regime A five-seed optimizer comparison")
    ap.add_argument("--las-raw", required=True); ap.add_argument("--las-calc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xgboost", choices=list(E.BOUNDS))
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--budget", type=int, default=50, help="max objective evaluations per run")
    ap.add_argument("--agents", type=int, default=5); ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tune-rows", type=int, default=RA.RegimeAConfig.max_train_tune,
                    help="training rows used inside the CV objective (Table 11 used RegimeAConfig.max_train_tune)")
    a = ap.parse_args(argv)

    out = Path(a.out) / "regime_a_multiseed"; out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.json"
    runs = json.load(open(runs_path, encoding="utf-8")) if runs_path.exists() else []
    done = {(r["seed"], r["method"]) for r in runs}

    cfg = B.Config(las_raw_dir=a.las_raw, las_calc_dir=a.las_calc, out_dir=str(a.out))
    ra = RA.RegimeAConfig(max_train_tune=a.tune_rows)
    frames = B.engineer(B.load_and_merge(cfg), cfg)
    feats = [f for f in RA.RAWLOG_FEATS
             if all(f in d.columns and d[f].notna().sum() >= cfg.min_well_rows for d in frames.values())]
    Xtr, ytr, Xte, yte, _ = RA._pooled_split(frames, feats, cfg, ra)
    if a.tune_rows and len(Xtr) > a.tune_rows:      # fixed tuning subset, as in regime_a_benchmark
        idx = np.random.default_rng(cfg.seed).choice(len(Xtr), a.tune_rows, replace=False)
        Xt, yt = Xtr[idx], ytr[idx]
    else:
        Xt, yt = Xtr, ytr
    print(f"[A-multi] features {feats}; train={len(Xtr)} test={len(Xte)} tuning rows={len(Xt)}")
    bounds = E.BOUNDS[a.model]
    base_obj = RA._cv_objective(a.model, Xt, yt, cfg, ra)

    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in a.methods.split(","):
            if (seed, name) in done:
                continue
            if name == "TPE":
                try:
                    import optuna  # noqa: F401
                except ImportError:
                    print("[A-multi] optuna not installed: TPE skipped"); continue
            trace = []

            def obj(h):
                if len(trace) >= a.budget:           # hard cap on evaluations
                    return 1.0
                v = base_obj(np.clip(h, [b[0] for b in bounds], [b[1] for b in bounds]))
                trace.append(float(v)); return v
            t0 = time.time()
            best = run_method(name, seed, bounds, obj, a)
            _, _, tr_r2, te_r2 = RA._fit_score(a.model, best, Xtr, ytr, Xte, yte, cfg)
            runs.append(dict(seed=seed, method=name, cv_r2=-min(trace), test_r2=float(te_r2),
                             train_r2=float(tr_r2), evals=len(trace),
                             best_so_far=list(-np.minimum.accumulate(trace)),
                             hp=E._decode(a.model, best), seconds=round(time.time() - t0, 1)))
            json.dump(runs, open(runs_path, "w", encoding="utf-8"), indent=1, default=float)
            print(f"[A-multi] seed {seed} {name:12s} CV={-min(trace):.4f} test={te_r2:.4f} evals={len(trace)}", flush=True)

    df = pd.DataFrame(runs)
    order = [m for m in METHODS if m in set(df.method)]
    rows = []
    for m in order:
        g = df[df.method == m]
        rows.append(dict(method=m, seeds=len(g),
                         cv_r2=f"{g.cv_r2.mean():.4f} ± {g.cv_r2.std(ddof=1):.4f}",
                         test_r2=f"{g.test_r2.mean():.4f} ± {g.test_r2.std(ddof=1):.4f}",
                         test_range=f"{g.test_r2.min():.4f}-{g.test_r2.max():.4f}",
                         evals_mean=round(g.evals.mean(), 1)))
    pd.DataFrame(rows).to_csv(out / "Table_12_multiseed.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in order:
        curves = [np.array(c) for c in df[df.method == m].best_so_far]
        L = max(len(c) for c in curves)
        pad = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in curves])
        for c in curves:
            ax.plot(np.arange(1, len(c) + 1), c, color=COLORS[m], lw=0.6, alpha=0.35)
        ax.plot(np.arange(1, L + 1), pad.mean(0), color=COLORS[m], lw=2, label=m)
    ax.set_xlabel("objective evaluations"); ax.set_ylabel("best 3-fold CV R$^2$")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_14_optimizer_convergence.{ext}", dpi=cfg.dpi, bbox_inches="tight")
    print(f"[A-multi] outputs in {out}")


if __name__ == "__main__":
    main()
