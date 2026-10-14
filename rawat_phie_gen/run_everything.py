#!/usr/bin/env python3
"""Run every analysis of the paper from the raw + calculated LAS files, in order.

Usage (Anaconda Prompt, inside this folder):
    python run_everything.py
    python run_everything.py --las-raw "D:\\...\\Las files" --las-calc "D:\\...\\the_9_wells" --out "D:\\rawat_local\\results_final"

Steps (each is skipped if its outputs already exist; add --force to redo all):
  1. Regime B + Regime A   run_all_regimes.py --stages all   Tables 1, 7, 8, 11, 13, 14, 15, 16, S1, S2; Figs 4-13, 15-20
  2. Revision analyses     rawat_revision_addons.py          Tables 9, 10, S4, S5, S7; Figs 2, 3; ridge value; CI
  3. Five-seed benchmark   regime_a_multiseed.py             Table 12, Fig. 14 (resumable)
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = r"D:\data\las_raw"
DEFAULT_CALC = r"D:\data\las_calc"
DEFAULT_OUT = r"D:\rawat_local\results_final"


def run(step, args):
    print(f"\n==== {step} ====\n   $ python {' '.join(str(a) for a in args)}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=HERE,
                   env={**os.environ, "PYTHONUTF8": "1"})      # same text handling on Windows
    print(f"   finished in {(time.time() - t) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las-raw", default=DEFAULT_RAW)
    ap.add_argument("--las-calc", default=DEFAULT_CALC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    # Regime A settings: use the ones you report in Section 4.6 / Table 5
    ap.add_argument("--agents", type=int, default=5, help="optimizer population N")
    ap.add_argument("--iters", type=int, default=10, help="optimizer iterations T_max")
    ap.add_argument("--rs-trials", type=int, default=50, help="random-search trials = evaluation budget")
    ap.add_argument("--tune-rows", type=int, default=4000, help="training rows used inside the tuning objective")
    ap.add_argument("--force", action="store_true", help="rerun steps whose outputs already exist")
    a = ap.parse_args()

    for p in (a.las_raw, a.las_calc):
        if not Path(p).is_dir():
            sys.exit(f"Folder not found: {p}\nPass the right path with --las-raw / --las-calc.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common = ["--las-raw", a.las_raw, "--las-calc", a.las_calc, "--out", out]

    if a.force or not (out / "SUMMARY_all_regimes.json").exists():
        run("1/3 Regime B (blind wells) + Regime A (benchmark)",
            ["run_all_regimes.py", "--stages", "all", "--model", "lightgbm", *common,
             "--ra-agents", a.agents, "--ra-iters", a.iters, "--ra-rs-trials", a.rs_trials,
             "--ra-subsample", a.tune_rows])
    else:
        print("1/3 already done (SUMMARY_all_regimes.json exists)")

    if a.force or not (out / "revision" / "revision_numbers.json").exists():
        run("2/3 Revision analyses", ["rawat_revision_addons.py", *common])
    else:
        print("2/3 already done (revision/revision_numbers.json exists)")

    run("3/3 Five-seed optimizer comparison",
        ["regime_a_multiseed.py", *common, "--agents", a.agents, "--iters", a.iters,
         "--budget", a.rs_trials, "--tune-rows", a.tune_rows])

    print(f"\nAll done. Results are in {out}")
    print("  tables\\  figures\\  MANIFEST.csv (file -> paper label)  TABLES.md")
    print("  revision\\  (Tables 9, 10, S4, S5, S7, Figs 2-3, revision_numbers.json)")
    print("  regime_a_multiseed\\  (Table 12, Fig. 14)")


if __name__ == "__main__":
    main()
