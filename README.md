# Blind-well effective porosity prediction, Rawat Basin

Code for the paper on random-split versus leave-one-well-out (LOWO) evaluation of machine-learning effective-porosity (PHIE) prediction from wireline logs, with a hybrid AQPSO-GWO hyperparameter-optimizer benchmark.

**No real well data are included.** The raw and calculated LAS files, and every dataset derived from them, are confidential and can be requested from the data owner (see the paper's Data availability statement). To test the code without them, the repository includes a generator for a synthetic eight-well dataset with the same file layout ([make_synthetic_data.py](rawat_phie_gen/make_synthetic_data.py)).

## Contents

| Folder / file | What it is |
|---|---|
| `rawat_phie_gen/` | The paper's code (v1.3). One command regenerates every table and figure of the paper, except the location map (Fig. 1) and Table B.6, from the LAS files. |
| `rawat_phie_gen/make_synthetic_data.py` | Creates the synthetic test dataset |
| `docs/TUTORIAL.md` | How-to examples for typical use cases, run on the synthetic dataset |
| `docs/USER_GUIDE.md` | Inputs, outputs, options, expected behaviour and computational requirements |
| `examples/expected_synthetic/` | Results to expect when the tutorial is run with the default seed |
| `AQPSO_GWO_Porosity_prediction.py` | The original single-script version of the optimizer benchmark (v1) |
| `LICENSE` | MIT License |

## Quick start (synthetic data)

Python 3.12 or newer is required (3.13 was used for the paper). No GPU is needed.

```bash
cd rawat_phie_gen
pip install -r requirements.txt
python make_synthetic_data.py --out ../synthetic_data
python run_everything.py --las-raw ../synthetic_data/las_raw --las-calc ../synthetic_data/las_calc --out ../synthetic_out
```

The full run takes about 48 minutes on a 4-core CPU. Compare `synthetic_out/SUMMARY_all_regimes.json` with `examples/expected_synthetic/SUMMARY_all_regimes.json`. For a 1-minute blind-well-only run and other use cases, see [docs/TUTORIAL.md](docs/TUTORIAL.md).

## Running the paper's pipeline on real data

```bash
python rawat_phie_gen/run_everything.py --las-raw "<raw LAS folder>" --las-calc "<calculated LAS folder>" --out "<output folder>"
```

See [rawat_phie_gen/README.md](rawat_phie_gen/README.md) for the output files behind each table and figure of the paper, and [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for the required curves and all options.

## Dependencies and computational requirements

Library versions are pinned in [rawat_phie_gen/requirements.txt](rawat_phie_gen/requirements.txt): numpy, pandas, scikit-learn, scipy, matplotlib, joblib, LightGBM, XGBoost, Optuna, lasio, SHAP, LIME, PyTorch (CPU is enough) and tabulate. The paper was run on an AMD Ryzen 7 4800H (8 cores) with 16 GB RAM; 8 GB RAM is enough for the synthetic dataset.

## Citation

Please cite the archived release: https://doi.org/10.5281/zenodo.23373360
