#!/usr/bin/env python3
# =============================================================================
# run_all_regimes.py — ONE entry point that generates ALL paper results for
# BOTH regimes, from the raw + calculated LAS files.
#
#   Regime B (blind-well / LOWO)  -> rawat_full_pipeline.py   (primary evidence)
#   Regime A (within-well benchmark) -> regime_a_benchmark.py (diagnostic)
#
# Runs from the Anaconda Prompt (CLI) or from Jupyter (import run_all).
# Keep these four files together in one folder:
#   run_all_regimes.py, rawat_full_pipeline.py, regime_a_benchmark.py,
#   phie_aqpso_lowo.py
#
# Anaconda Prompt:
#   python run_all_regimes.py --stages all ^
#       --las-raw  "D:\...\Las files" ^
#       --las-calc "D:\...\Rawat Calculated Logs\the_9_wells" ^
#       --out "D:\rawat_local\results_all" --model lightgbm
#
# Jupyter:
#   from run_all_regimes import run_all
#   run_all(las_raw=r"D:\...\Las files",
#           las_calc=r"D:\...\the_9_wells",
#           out=r"D:\rawat_local\results_all", stages="all", model="lightgbm")
# =============================================================================
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

import rawat_full_pipeline as B
from regime_a_benchmark import RegimeAConfig, run_regime_a


def run_all(las_raw: str, las_calc: str, out: str, stages: str = "all",
            model: str = "auto", fast: bool = False, row_stride: int = 1,
            regime_a_cfg: RegimeAConfig | None = None) -> dict:
    """Generate results for Regime A and/or B into one output folder.

    stages: "all" | "a" | "b".  model: "auto"|"lightgbm"|"gbr"|"ridge".
    Returns {"regime_a": ..., "regime_b": ...} with whatever ran.
    """
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cfg = B.Config(las_raw_dir=las_raw, las_calc_dir=las_calc, out_dir=str(out),
                   model=model, fast=fast, row_stride=row_stride)
    ra = regime_a_cfg or RegimeAConfig(fast=fast)

    results, manifest_rows = {}, []

    # ---- Regime B (primary) ---------------------------------------------
    if stages in ("all", "b"):
        rb = B.run(cfg)                       # writes tables/figures + its MANIFEST.csv
        results["regime_b"] = rb.get("summary", {})
        mb = out / "MANIFEST.csv"
        if mb.exists():
            manifest_rows.append(pd.read_csv(mb))

    # ---- Regime A (diagnostic) ------------------------------------------
    if stages in ("all", "a"):
        man_a: list = []
        ra_res = run_regime_a(cfg, ra, out, man_a)
        results["regime_a"] = dict(winner=ra_res["winner"],
                                   winner_test_r2=ra_res["winner_test_r2"],
                                   backend=ra_res["backend"])
        manifest_rows.append(pd.DataFrame(man_a))

    # ---- combined MANIFEST + SUMMARY ------------------------------------
    if manifest_rows:
        combined = pd.concat(manifest_rows, ignore_index=True).drop_duplicates("file")
        combined.sort_values("paper_label").to_csv(out / "MANIFEST.csv", index=False)
    json.dump(results, open(out / "SUMMARY_all_regimes.json", "w"), indent=2, default=float)

    print("\n================= SUMMARY =================")
    if "regime_b" in results:
        s = results["regime_b"]
        print(f"Regime B (LOWO, blind): pooled R2 = {s.get('lowo_pooled', {}).get('R2')}")
        print(f"   random-split gauge   : pooled R2 = {s.get('random_pooled', {}).get('R2')}")
        print(f"   PHIT x (1-VCL) policy: R2 = {list(s.get('ladder', {}).values())[0] if s.get('ladder') else 'n/a'}")
        print(f"   selected features    : {s.get('selected_features')}")
    if "regime_a" in results:
        a = results["regime_a"]
        print(f"Regime A (within-well): winner = {a['winner']}  test R2 = {a['winner_test_r2']:.4f}")
    print(f"backend used: {B.resolve_backend(cfg)}")
    print(f"all artifacts -> {out}")
    print("==========================================")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate ALL paper results (Regimes A & B)")
    ap.add_argument("--las-raw", required=True)
    ap.add_argument("--las-calc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", choices=["all", "a", "b"], default="all")
    ap.add_argument("--model", choices=["auto", "lightgbm", "gbr", "ridge"], default="auto")
    ap.add_argument("--fast", action="store_true", help="shrink budgets (quick validation only)")
    ap.add_argument("--ra-subsample", type=int, default=None, help="Regime A: rows used for tuning (default 5000)")
    ap.add_argument("--ra-agents", type=int, default=None, help="Regime A: optimizer swarm size (default 10)")
    ap.add_argument("--ra-iters", type=int, default=None, help="Regime A: optimizer iterations (default 6)")
    ap.add_argument("--ra-rs-trials", type=int, default=None, help="Regime A: random-search trials (default 25)")
    ap.add_argument("--ra-models", default=None,
                    help="Regime A: comma list of models (default all 4: random_forest,xgboost,simple_mlp,optimized_mlp)")
    a = ap.parse_args(argv)
    ra = RegimeAConfig(fast=a.fast)
    if a.ra_subsample is not None: ra.max_train_tune = a.ra_subsample
    if a.ra_agents is not None: ra.opt_agents = a.ra_agents
    if a.ra_iters is not None: ra.opt_iters = a.ra_iters
    if a.ra_rs_trials is not None: ra.n_rs_trials = a.ra_rs_trials
    if a.ra_models: ra.models = tuple(m.strip() for m in a.ra_models.split(","))
    return run_all(a.las_raw, a.las_calc, a.out, a.stages, a.model, a.fast, regime_a_cfg=ra)


if __name__ == "__main__":
    main()
