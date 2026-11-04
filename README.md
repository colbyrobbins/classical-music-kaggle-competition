# Classical Music Kaggle Competition

Predict which previous patrons will buy a subscription to the 2014-15 concert season. Scored by AUROC on a public/private split of the test set. Course: Duke AIPI.

## Approach

1. **Data pipeline** (`src/features.py`): cleans the raw tables and builds one feature row per account from history only (seasons up to 2013-14). Train and test go through identical code, and no label information is used.
2. **Modeling** (`src/modeling.py`): imputation, correlation filtering, importance-based selection, scaling and SMOTE all live inside the sklearn pipeline, so they are refit on each CV training fold only (no leakage).
3. **Validation**: repeated stratified 5-fold CV, metric = AUROC.
4. **Final model**: equal-weight rank average of L2 logistic regression, random forest, XGBoost and scikit-learn's HistGradientBoosting on the top-K features by cross-validated permutation importance (K chosen from the accuracy-vs-size curve). Features and tuned hyperparameters are chosen in the notebook and saved to `results/final_config.json`. An XGBoost baseline and a fallback without the account-ID feature are also saved.

The full walkthrough (EDA, cleaning checks, model comparison, feature pruning, tuning, ensemble evidence) is in `notebooks/kaggle_modeling.ipynb`.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md            # expected files and where to put them
│   └── raw/                 # competition CSVs (git-ignored)
├── src/
│   ├── features.py          # cleaning + feature engineering
│   └── modeling.py          # CV harness, selectors, tuning, ensemble
├── notebooks/
│   └── kaggle_modeling.ipynb
├── submissions/
│   ├── submission.csv                 # final submission
│   ├── submission_no_id.csv           # fallback without account-ID feature
│   └── submission_xgb_baseline.csv    # early XGBoost baseline
├── scripts/
│   └── make_submission.py             # rebuild the final model from results/final_config.json
├── results/
│   └── final_config.json              # selected features + tuned hyperparameters (written by the notebook)
├── writeup/                 # 2-4 page modeling writeup goes here
└── docs/
    └── kaggle_instructions.pptx       # assignment instructions
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the competition CSVs in data/raw/ (see data/README.md)
jupyter lab notebooks/kaggle_modeling.ipynb          # run all cells; writes to submissions/
```

Optional: after the notebook has written `results/final_config.json`, rebuild the submission from the command line:

```bash
python scripts/make_submission.py            # submissions/submission.csv  (--no-id for the fallback)
```

The notebook finds the repo root automatically, so it runs from either the root or `notebooks/`.
