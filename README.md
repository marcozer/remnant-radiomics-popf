# 5-rad: CT radiomics of the future pancreatic remnant and CR-POPF after distal pancreatectomy

Code, de-identified data and results of the study *Preoperative Prediction of Postoperative Pancreatic Fistula After Distal
Pancreatectomy Using a Sparse Radiomic Signature of the Pancreatic Remnant* (Beaujon Hospital, AP-HP, Clichy, France).

## Aim

To test whether radiomic features of the future pancreatic remnant (head and isthmus) on preoperative portal venous-phase CT
predict a clinically relevant postoperative pancreatic fistula (ISGPS 2016 grade B or C) after distal pancreatectomy, and to
compare the resulting model (5-rad: five features in an elastic-net logistic regression) with the published preoperative
D-FRS and DISPAIR-FRS. Single center, 251 patients, 50 CR-POPF; internal validation only (bootstrap .632+ and strict nested
cross-validation). The full description of every step is in [PIPELINE.md](PIPELINE.md).

## Repository

| Path | Content |
|---|---|
| `data/` | de-identified input tables, one row per patient (clinical variables, fistula grade, 758 radiomic features, scanner metadata, flow-chart counts); checked against `data/SHA256SUMS`; no images |
| `extraction/` | PyRadiomics settings and the feature-extraction scripts (need the CT images, not shared) |
| `src/` | ComBat harmonization, prefilter, Stabl, elastic net, bootstrap, calibration, DeLong test, cutpoints |
| `analysis/` | one script per analysis (`01` to `12`), run in order by `run_all.sh` |
| `results/` | tables, figures, `key_numbers.json` and the outputs of each analysis, as in the manuscript |
| `Dockerfile`, `requirements.txt`, `requirements-lock.txt` | analysis environment (Python 3.12) |
| `PIPELINE.md` | step-by-step description of the methods, the data dictionary and the script-to-manuscript mapping |
| `CITATION.cff`, `.zenodo.json`, `LICENSE` | citation, archive metadata, license |

## Reproduce

```sh
pip install -r requirements.txt
./run_all.sh          # about 7 hours with 13 workers; WORKERS and BOOTSTRAPS can be set
```

Figures and tables alone: `python3 analysis/11_figures.py` and `python3 analysis/12_tables.py` from the stored results.

## Data and license

Patients carry random identifiers; no names, hospital identifiers, dates or images are included. Code: MIT License
(`src/stabl_official.py`: Clear BSD, notice in the file). Data and results: de-identified patient data from AP-HP, shared to
reproduce the published analyses; any other use requires the agreement of the authors and of AP-HP. Please cite the study
(`CITATION.cff`).
