# rawat_phie_gen v1.3

This is the code behind the blind-well (leave-one-well-out) porosity results for eight wells in the Rawat Basin, Sudan. One command regenerates every table and figure of the paper from the raw and calculated LAS files.

To try it without the confidential data, first run `python make_synthetic_data.py --out ../synthetic_data` and use those folders below. Step-by-step examples are in [../docs/TUTORIAL.md](../docs/TUTORIAL.md), and the full description of inputs, outputs and options is in [../docs/USER_GUIDE.md](../docs/USER_GUIDE.md).

**No well data are included.** The LAS files and every dataset derived from them are confidential (see the paper's Data availability statement); they can be requested from the data owner.

## Files

| File | Role |
|---|---|
| `run_everything.py` | Runs every step below in order; this is the one command to use |
| `run_all_regimes.py` | Regime B (blind wells) and Regime A (random-split benchmark) |
| `rawat_full_pipeline.py` | Regime B: LAS merge, features, LOWO, ablation, few-shot, SHAP, Taylor |
| `regime_a_benchmark.py` | Regime A: 21-configuration optimizer benchmark, outlier and correlation screens |
| `phie_aqpso_lowo.py` | AQPSO, GWO and hybrid AQPSO-GWO optimizer, plus the model factory |
| `rawat_revision_addons.py` | Raw-log-only LOWO, contiguous few-shot, one-change-per-stage ablation, Figs 2–3 |
| `regime_a_multiseed.py` | Five-seed, equal-budget optimizer comparison with TPE |
| `run_all_regimes.ipynb` | Jupyter version of step 1 |
| `make_synthetic_data.py` | Creates the synthetic eight-well test dataset |
| `requirements.txt` | Library versions |

## Install (once)

You need Python 3.12 or newer (3.13 was used).

```bat
conda create -n porosity python=3.13 -y
conda activate porosity
pip install -r requirements.txt
```

## Run

```bat
conda activate porosity
cd /d <this folder>
python run_everything.py --las-raw "<raw LAS folder>" --las-calc "<calculated LAS folder>" --out "<output folder>"
```

The raw and calculated LAS files are matched by file name. Akaff-1 is excluded automatically. Finished steps are skipped when you run the command again, so you can rerun it after an interruption.

## Outputs (in `--out`)

| Folder / file | Contents |
|---|---|
| `tables/`, `figures/`, `MANIFEST.csv`, `TABLES.md` | Regime B and Regime A tables and figures |
| `revision/` | Raw-log-only references, isolated ablation, contiguous few-shot, outlier and correlation screens; `revision_numbers.json` |
| `regime_a_multiseed/` | Five-seed optimizer comparison |
| `merged/` | Merged per-well data. **Confidential when made from the real LAS files: never upload it.** |

### Output files and paper labels

File names (and the labels inside `MANIFEST.csv`) keep the numbering used while the paper was drafted. This table gives the label of each item in the article.

| Article item | Output file |
|---|---|
| Table 1 | `tables/Table_01_well_inventory_audit.csv` |
| Table 2 | `tables/Table_07_deterministic_ladder.csv` |
| Table 3 | `tables/Table_08_perwell_lowo_metrics.csv` |
| Table 4 | `tables/Table_09_ablation.csv` |
| Table 5 | `revision/Table_10_rawlog_only.csv` |
| Table 6 | `tables/Table_14_fewshot.csv` |
| Table 7 | `tables/Table_15_attribution.csv` |
| Table 8 | `tables/Table_16_taylor_stats.csv` |
| Tables A.1–A.5 | Settings, not outputs: see `Config` in `rawat_full_pipeline.py` and `RegimeAConfig` in `regime_a_benchmark.py` |
| Table B.1 | `tables/Table_07b_perwell_deterministic.csv` |
| Table B.2 | `tables/Table_08b_perwell_leakage.csv` |
| Table B.3 | `tables/Table_S1_sign_consistency.csv` |
| Table B.4 | `tables/Table_S3_outlier_screen.csv` |
| Table B.5 | `tables/Table_S4_correlation_pairs.csv` |
| Table B.6 | Not code-generated (needs the confidential formation tops) |
| Table B.7 | `revision/Table_S7_fewshot_contiguous.csv` |
| Table B.8 | `revision/Table_09_ablation_isolated.csv` |
| Table B.9 | `tables/Table_12_benchmark_all.csv` |
| Table B.10 | `regime_a_multiseed/Table_12_multiseed.csv` |
| Table B.11 | `tables/Table_13_perwell_benchmark_winner.csv` |
| Fig. 1 | Not code-generated (location map) |
| Fig. 2 | `figures/Fig_03_mad_outlier_screen.png` |
| Fig. 3 | `revision/Fig_03_correlation_matrix.png` |
| Fig. 4 | `figures/Fig_05_sign_consistency_screen.png` |
| Fig. 5 | `figures/Fig_06_lowo_sfs_trace.png` |
| Fig. 6 | `figures/Fig_02_target_audit_crossplot.png` |
| Figs. 7–13 | `figures/Fig_07_...` to `figures/Fig_13_...` (same numbers) |
| Fig. 14 | `regime_a_multiseed/Fig_14_optimizer_convergence.png` |
| Figs. 15–20 | `figures/Fig_15_...` to `figures/Fig_20_...` (same numbers) |

Sanity check on the real data: in `SUMMARY.json` the deterministic PHIT × (1 − VCL) value is about 0.936, and the pooled LOWO R² is about 0.96. Expected values for the synthetic data are in [../examples/expected_synthetic/](../examples/expected_synthetic/).
