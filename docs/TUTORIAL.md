# Tutorial

These how-to examples cover the typical uses of the code. The paper's well data are confidential, so the examples use a synthetic dataset that has the same file layout. All values are made up, so the synthetic results test the software and show how to read its outputs; they are not the paper's numbers.

## 1. Install (once)

You need Python 3.12 or newer (3.13 was used for the paper).

```bash
git clone https://github.com/Montaser126/Rawat_porosity_code.git
cd Rawat_porosity_code/rawat_phie_gen
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

With Anaconda you can use `conda create -n porosity python=3.13 -y` and `conda activate porosity` instead of the two `venv` lines.

## 2. Create the synthetic test dataset

```bash
python make_synthetic_data.py --out ../synthetic_data
```

This writes eight wells, `SYN-1` to `SYN-8`, about 23 000 depth samples in total:

```
synthetic_data/las_raw/SYN-1.las ... SYN-8.las    raw tool curves (GR, RHOB, NPHI, DT, PE, RT_DEEP, RT_SHAL, RXO, SP, CALI, BS, DRHO)
synthetic_data/las_calc/SYN-1.las ... SYN-8.las   interpreted curves (PHIT, VCL, PHIE, SW)
```

Options: `--wells` (number of wells, default 8), `--samples` (samples per well, default 3000) and `--seed` (default 42).

## 3. Use case A: run the whole workflow

```bash
python run_everything.py --las-raw ../synthetic_data/las_raw --las-calc ../synthetic_data/las_calc --out ../synthetic_out
```

Measured run time on a 4-core CPU without a GPU: about 48 minutes (Regime B and Regime A, 30 min; revision analyses, 1 min; five-seed benchmark, 17 min). If the run stops, start the same command again and finished steps are skipped.

### Check your result

Open `synthetic_out/SUMMARY_all_regimes.json` and compare it with [../examples/expected_synthetic/SUMMARY_all_regimes.json](../examples/expected_synthetic/SUMMARY_all_regimes.json). With the default seed you should see:

| Quantity | Expected value (synthetic data) |
|---|---|
| Wells merged | 8 (SYN-1 to SYN-8) |
| Deterministic PHIT × (1 − VCL), pooled R² | 0.995 |
| Pooled blind-well (LOWO) R² of the deployed pipeline | 0.995 |
| Raw-log-only conventional model, random split vs LOWO R² | 0.989 vs 0.974 |

Small differences in the last digit can come from different library versions or CPU types. Because the synthetic interpreter curves are a simple function of the synthetic logs, every model scores high here; on real multi-well data the random-split and blind-well scores differ much more (see the paper).

## 4. Use case B: run only the blind-well (LOWO) evaluation

This is the quickest way to apply the generalization-first pipeline to your own wells (about 1 minute on the synthetic data).

```bash
python run_all_regimes.py --stages b --model lightgbm --las-raw ../synthetic_data/las_raw --las-calc ../synthetic_data/las_calc --out ../lowo_only
```

Read `lowo_only/SUMMARY.json` for the pooled scores and `lowo_only/tables/Table_08_perwell_lowo_metrics.csv` for each held-out well.

## 5. Use case C: compare hyperparameter optimizers on your data

```bash
python regime_a_multiseed.py --las-raw ../synthetic_data/las_raw --las-calc ../synthetic_data/las_calc --out ../optimizers --model xgboost --seeds 1,2,3 --budget 30
```

This runs random search, TPE, AQPSO, GWO and the hybrid AQPSO-GWO at an equal evaluation budget (about 7 minutes on the synthetic data) and writes `optimizers/regime_a_multiseed/Table_12_multiseed.csv` and the convergence figure.

## 6. Use case D: run on your own wells

1. Put one raw LAS file per well in one folder and the matching interpreted LAS file, with the same file name, in a second folder.
2. Check that the curve names match the table in [USER_GUIDE.md](USER_GUIDE.md) (common aliases are recognized).
3. If a well must be left out, add its name to `exclude_wells` in `Config` (`rawat_full_pipeline.py`).
4. Run use case A or B with your folders.
5. Look at Fig. 6 (target audit, `figures/Fig_02_target_audit_crossplot.png`) first. If PHIT × (1 − VCL) already explains most of PHIE, report that deterministic baseline next to any machine-learning score.

## 7. Where each paper item comes from

The table "Output files and paper labels" in [../rawat_phie_gen/README.md](../rawat_phie_gen/README.md) lists the output file behind every table and figure of the article.
