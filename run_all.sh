#!/bin/sh
# Every analysis from the de-identified inputs in data/ (2,000 bootstrap samples by default). The nested cross-validation
# is cached in results/nested_cv/ and resumed.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-12}
B=${BOOTSTRAPS:-2000}  # samples per series in 06, 07 and 08, outer samples in 10
python3 analysis/01_table1.py
python3 analysis/02_scanner_table.py
python3 analysis/02b_image_type_redundancy.py
NUMBA_NUM_THREADS=$W THREADS=$W python3 analysis/03_feature_selection.py
NUMBA_NUM_THREADS=$W THREADS=$W python3 analysis/03b_stabl_threshold.py
python3 analysis/04_lasso_paths.py
NUMBA_NUM_THREADS=1 python3 analysis/05_final_model.py
NUMBA_NUM_THREADS=1 python3 analysis/05b_model_choice.py --workers $W
NUMBA_NUM_THREADS=1 python3 analysis/06_bootstrap_validation.py --bootstraps $B --workers $W
NUMBA_NUM_THREADS=1 python3 analysis/07_model_families.py --bootstraps $B --workers $W
NUMBA_NUM_THREADS=1 python3 analysis/08_clinical_comparison.py --bootstraps $B --workers $W
NUMBA_NUM_THREADS=1 python3 analysis/09_nested_cv.py --workers $W --threads 1
NUMBA_NUM_THREADS=1 python3 analysis/10_nested_cutpoints.py --outer $B --workers $W
NUMBA_NUM_THREADS=1 python3 analysis/10b_cutpoint_sensitivity.py --workers $W
python3 analysis/11_figures.py
python3 analysis/12_tables.py
