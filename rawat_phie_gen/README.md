# rawat_phie_gen v1.2

This is the code behind the blind-well (leave-one-well-out) porosity results for eight wells in the Rawat Basin, Sudan. One command regenerates every table and figure of the paper from the raw and calculated LAS files.

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

| Folder / file | Paper items |
|---|---|
| `tables/`, `figures/`, `MANIFEST.csv`, `TABLES.md` | Tables 1, 7, 8, 11, 13, 14, 15, 16, S1, S2; Figs 4–13, 15–20 (`MANIFEST.csv` maps every file to its label) |
| `revision/` | Tables 9, 10, S4, S5, S7; Figs 2, 3; `revision_numbers.json` |
| `regime_a_multiseed/` | Table 12, Fig. 14 |
| `merged/` | Merged per-well LAS data. **Confidential: never upload it.** |

Sanity check: in `SUMMARY.json` the deterministic PHIT × (1 − VCL) value is about 0.936, and the pooled LOWO R² is about 0.96.
