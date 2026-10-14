# Blind-well effective porosity prediction, Rawat Basin

Code for the paper on random-split versus leave-one-well-out (LOWO) evaluation of machine-learning effective-porosity (PHIE) prediction from wireline logs, with a hybrid AQPSO-GWO hyperparameter-optimizer benchmark.

**No well data are included.** The raw and calculated LAS files, and every dataset derived from them, are confidential. They can be requested from the data owner (see the paper's Data availability statement).

## Contents

| Folder / file | What it is |
|---|---|
| `rawat_phie_gen/` | The paper's code (v1.2). One command regenerates every table and figure of the paper, except the location map (Fig. 1) and Table S6, from the LAS files. |
| `AQPSO_GWO_Porosity_prediction.py` | The original single-script version of the optimizer benchmark (v1). |

## Running the paper's pipeline

Python 3.12 or newer is required (3.13 was used for the paper).

```bash
pip install -r rawat_phie_gen/requirements.txt
python rawat_phie_gen/run_everything.py --las-raw "<raw LAS folder>" --las-calc "<calculated LAS folder>" --out "<output folder>"
```

See [rawat_phie_gen/README.md](rawat_phie_gen/README.md) for the outputs and how they map to the paper's tables and figures.
