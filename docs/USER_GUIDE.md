# User guide

This guide describes what the workflow reads, what it writes, which options it accepts and what it is expected to do. For step-by-step examples, see [TUTORIAL.md](TUTORIAL.md).

## 1. Inputs

The workflow reads two folders of LAS 2.0 files. Each well has two files with the same name, one in each folder (for example `SYN-1.las` in both).

### 1.1 Raw log folder (`--las-raw`)

The first curve must be depth in metres. Mnemonics are matched case-insensitively, and common aliases are accepted (see `_CANON` in `rawat_full_pipeline.py`).

| Curve | Unit | Aliases accepted | Needed for |
|---|---|---|---|
| GR | gAPI | HCGR, HSGR, CGR, SGR | VSH_GR, PHIE_DERIVED, GR_N |
| RHOB | g/cm³ | RHOZ, ZDEN, DEN | PHID, PHIDN, ND_SEP |
| NPHI | v/v (percent is converted automatically) | TNPH, CNC, NPOR | PHIDN, ND_SEP |
| DT | µs/ft | DTC, AC | PHIS |
| RT_DEEP | ohm·m | RD, RLA5, HLLD, AHF90, RT, ILD | LOG_RD |
| RT_SHAL | ohm·m | RS, RLA1, HLLS, AHF10 | LOG_RS |
| PE, RXO, SP, CALI, BS, DRHO | standard | see `_CANON` | optional (raw-log references, QC) |

### 1.2 Calculated (interpreted) folder (`--las-calc`)

| Curve | Meaning | Required |
|---|---|---|
| PHIE | Interpreter's effective porosity, v/v | Yes (target) |
| PHIT | Total porosity, v/v | Yes for the generalization-first pipeline |
| VCL | Clay volume, v/v | Yes for the generalization-first pipeline |
| SW | Water saturation, v/v | Optional |

Calculated curves are attached to the raw depths by nearest-depth matching within 0.6 of the depth step. Values outside the physical ranges in `Config.valid_range` are set to missing. A well listed in `Config.exclude_wells` is skipped (by default Akaff-1, the well excluded in the paper).

### 1.3 Requirements on the data

- At least three wells are needed for leave-one-well-out (LOWO) evaluation; eight were used in the paper.
- Depths must increase down each file. Null values use the LAS `NULL` entry (usually -999.25).
- No core data are needed.

## 2. Outputs

Everything is written under the folder given by `--out`.

| Path | Contents |
|---|---|
| `merged/` | One merged CSV per well (raw + calculated curves on one depth axis). Treat as confidential if your LAS files are. |
| `tables/` | Every table as CSV (and Markdown in `TABLES.md`) |
| `figures/` | Every figure as 600 dpi PNG |
| `MANIFEST.csv` | Maps every file to the table or figure label it produces |
| `SUMMARY.json`, `SUMMARY_all_regimes.json` | Headline numbers (pooled LOWO R², deterministic baseline, selected features, random-split R²) |
| `revision/` | Raw-log-only references, isolated ablation, contiguous few-shot, outlier and correlation screens |
| `regime_a_multiseed/` | Five-seed optimizer comparison with TPE |

The labels written into `MANIFEST.csv` follow the numbering used while the paper was drafted. The table "Output files and paper labels" in [../rawat_phie_gen/README.md](../rawat_phie_gen/README.md) converts them to the numbering of the published article.

## 3. Options

### 3.1 `run_everything.py` (recommended entry point)

| Option | Default | Meaning |
|---|---|---|
| `--las-raw` | required in practice | Folder of raw LAS files |
| `--las-calc` | required in practice | Folder of calculated LAS files |
| `--out` | required in practice | Output folder |
| `--agents` | 5 | Optimizer population size N (Regime A) |
| `--iters` | 10 | Optimizer iterations T_max (Regime A) |
| `--rs-trials` | 50 | Random-search trials, which is also the evaluation budget |
| `--tune-rows` | 4000 | Training rows used inside the tuning objective |
| `--force` | off | Rerun steps whose outputs already exist |

### 3.2 Individual steps

| Script | Use it when you want | Key options |
|---|---|---|
| `run_all_regimes.py` | Regime B (blind wells) and/or Regime A (random-split optimizer benchmark) | `--stages all|a|b`, `--model auto|lightgbm|gbr|ridge`, `--fast` |
| `rawat_revision_addons.py` | Raw-log-only references, isolated ablation, contiguous few-shot, outlier and correlation screens | `--selected VCL,PHIT,PHIE_DERIVED`, `--mad-threshold 3.0`, `--skip-ablation` |
| `regime_a_multiseed.py` | Five-seed, equal-budget optimizer comparison | `--seeds 1,2,3,4,5`, `--methods`, `--budget`, `--model xgboost` |

Physical constants (matrix and fluid density, sonic matrix and fluid transit times), the feature-selection cap and the few-shot fractions are fields of `Config` in `rawat_full_pipeline.py`.

## 4. Expected behaviour

1. LAS files are merged, cleaned and written to `merged/`.
2. Physics-informed features are built, tool-unit curves are normalized per well, and the target is audited against PHIT × (1 − VCL).
3. Features are screened for a consistent correlation sign across wells and then chosen by sequential forward selection scored with LOWO R².
4. A regularized LightGBM (or ridge) model is scored under LOWO, so the scored well never contributes to any stage. The same pipeline is also scored under an 80/20 random split to measure leakage.
5. Few-shot affine recalibration, SHAP or permutation attribution, LIME examples and a Taylor diagram are computed for the blind predictions.
6. Regime A tunes RF, XGBoost and two MLPs with AQPSO, GWO, the hybrid AQPSO-GWO and random search on a random split.

Each finished step is skipped on a second run, so an interrupted run can be restarted with the same command. Seeds are fixed, so a rerun on the same data gives the same numbers (GPU-free runs; neural networks run on the CPU).

On the synthetic dataset the expected headline values are given in [TUTORIAL.md](TUTORIAL.md). If your own run gives a deterministic PHIT × (1 − VCL) R² far below 0.9, check that PHIE, PHIT and VCL come from the same interpretation.

## 5. Computational requirements

- Python 3.12 or newer (3.13 was used for the paper). Library versions are pinned in `rawat_phie_gen/requirements.txt`.
- CPU only; no GPU is needed.
- Memory: 8 GB RAM is enough for the synthetic dataset; 16 GB was used for the paper's 146 586 samples.
- Paper hardware: AMD Ryzen 7 4800H (8 cores), 16 GB RAM, Windows 11.
- Run time grows with the number of samples and the Regime A budget; see [TUTORIAL.md](TUTORIAL.md) for the measured time on the synthetic dataset.
