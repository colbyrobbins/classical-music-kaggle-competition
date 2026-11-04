"""
Modeling harness for the 2014-15 renewal task.

Every preprocessing step (imputation, correlation filter, importance-based selection,
scaling, SMOTE) lives INSIDE the pipeline, so cross-validation refits them on each
training fold only -- no information from the validation fold leaks in.

Typical use (see the notebook):
    from src.modeling import build_pipeline, make_cv, evaluate, compare
    cv = make_cv()
    evaluate(build_pipeline("logreg", imbalance="weight"), X_train, y_train, cv)

Models: L2 logistic regression ("logreg"), random forest ("rf"), XGBoost ("xgb") and scikit-learn's
HistGradientBoosting ("hgb"). The final model is an equal-weight rank average of the four (`build_ensemble`); its features and
hyperparameters are chosen in the notebook and stored in results/final_config.json.

Pipeline order:  impute -> [corr filter] -> [permutation-importance top-k] -> scale -> [SMOTE] -> model
Imbalance options:
    "none"    : nothing
    "weight"  : class_weight="balanced" (logreg, rf, hgb) / scale_pos_weight (xgb)
    "smote"   : imblearn SMOTE on the training fold only
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # imblearn's Pipeline accepts samplers (SMOTE)
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


# --------------------------------------------------------------------------- #
# Feature-selection transformers (fit on the training fold only)
# --------------------------------------------------------------------------- #
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop constant columns, then greedily drop one of each pair with |corr| > threshold.
    Columns are visited in order of |corr with y| (strongest first), so the more
    predictive member of a correlated pair is the one that survives."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        X = X.loc[:, X.std() > 0]
        order = X.corrwith(y).abs().fillna(0).sort_values(ascending=False).index
        corr = X[order].corr().abs().fillna(0)
        keep = []
        for col in order:
            if all(corr.loc[col, k] <= self.threshold for k in keep):
                keep.append(col)
        self.keep_ = [c for c in X.columns if c in set(keep)]
        self.dropped_ = [c for c in X.columns if c not in set(keep)]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]


class PermutationSelector(BaseEstimator, TransformerMixin):
    """Keep the top_k features by permutation importance (drop in validation AUROC).
    Importances come from a random forest fit on 75% of the training fold and scored on
    the other 25%, so they are measured out-of-sample (impurity importances are biased
    toward high-cardinality / continuous columns)."""

    def __init__(self, top_k: int = 30, n_repeats: int = 3, val_size: float = 0.25,
                 random_state: int = SEED):
        self.top_k = top_k
        self.n_repeats = n_repeats
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=self.val_size, stratify=y,
                                          random_state=self.random_state)
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                    random_state=self.random_state).fit(Xa, ya)
        r = permutation_importance(rf, Xb, yb, scoring="roc_auc", n_repeats=self.n_repeats,
                                   random_state=self.random_state, n_jobs=-1)
        self.importances_ = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        self.keep_ = list(self.importances_.index[: self.top_k])
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def _model(name: str, imbalance: str, pos_weight: float, **params):
    cw = "balanced" if imbalance == "weight" else None
    if name == "logreg":
        p = dict(C=0.1, max_iter=5000, class_weight=cw) | params
        return LogisticRegression(penalty="l2", **p)          # L2-regularised
    if name == "rf":
        p = dict(n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=-1,
                 random_state=SEED, class_weight=cw) | params
        return RandomForestClassifier(**p)
    if name == "xgb":
        p = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, eval_metric="auc", n_jobs=-1, random_state=SEED,
                 scale_pos_weight=pos_weight if imbalance == "weight" else 1.0) | params
        return XGBClassifier(**p)
    if name == "hgb":
        p = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
                 l2_regularization=1.0, random_state=SEED, class_weight=cw) | params
        return HistGradientBoostingClassifier(**p)
    raise ValueError(f"unknown model {name!r}")


def build_pipeline(model: str = "logreg", imbalance: str = "none", corr_threshold: float | None = None,
                   top_k: int | None = None, smote_ratio: float = 0.5, pos_weight: float = 19.9,
                   **model_params) -> Pipeline:
    """pos_weight = (#negatives / #positives); train.csv has 6609 / 332 ~= 19.9."""
    if imbalance not in {"none", "weight", "smote"}:
        raise ValueError("imbalance must be 'none', 'weight' or 'smote'")
    steps = [("impute", SimpleImputer(strategy="median").set_output(transform="pandas"))]
    if corr_threshold is not None:
        steps.append(("corr", CorrelationFilter(corr_threshold)))
    if top_k is not None:
        steps.append(("select", PermutationSelector(top_k)))
    steps.append(("scale", StandardScaler().set_output(transform="pandas")))   # SMOTE/logreg need scaled inputs
    if imbalance == "smote":
        # sampling_strategy = desired minority/majority ratio after oversampling (0.5 = 1 positive per 2 negatives)
        steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, k_neighbors=5, random_state=SEED)))
    steps.append(("model", _model(model, imbalance, pos_weight, **model_params)))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
class RankAverager(ClassifierMixin, BaseEstimator):
    """Average of the members' scores after converting each to a percentile rank (0-1) within the
    batch being scored. AUROC only depends on ordering, so this blends models whose raw probabilities
    live on different scales (e.g. class-weighted vs unweighted)."""

    def __init__(self, estimators, weights=None):
        self.estimators = estimators          # list of (name, unfitted pipeline)
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fitted_ = [(n, clone(e).fit(X, y)) for n, e in self.estimators]
        return self

    def predict_proba(self, X):
        w = np.ones(len(self.fitted_)) if self.weights is None else np.asarray(self.weights, float)
        ranks = [rankdata(e.predict_proba(X)[:, 1]) / len(X) for _, e in self.fitted_]
        s = np.average(ranks, axis=0, weights=w)
        return np.column_stack([1 - s, s])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def fold_scores(pipe, X, y, cv) -> np.ndarray:
    """Per-fold AUROC array (same folds -> paired comparisons between configs)."""
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")


def cv_permutation_importance(pipe, X, y, cv, n_repeats: int = 5) -> pd.Series:
    """Permutation importance (drop in AUROC) measured on each CV validation fold and averaged
    across all folds. Much more stable than a single split: every account is scored out-of-fold
    several times, so correlated features and the ~17 positives per split matter less."""
    imps = []
    for tr_idx, va_idx in cv.split(X, y):
        p = clone(pipe).fit(X.iloc[tr_idx], y[tr_idx])
        r = permutation_importance(p, X.iloc[va_idx], y[va_idx], scoring="roc_auc",
                                   n_repeats=n_repeats, random_state=SEED, n_jobs=1)
        imps.append(r.importances_mean)
    return pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)


def make_cv(n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> RepeatedStratifiedKFold:
    """Stratified 5-fold, repeated. With only ~332 positives (~66 per fold) a single
    5-fold split is noisy; repeating it stabilises the AUROC estimate."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def evaluate(pipe, X, y, cv=None) -> dict:
    cv = cv or make_cv()
    s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return {"auroc_mean": s.mean(), "auroc_std": s.std(), "n_folds": len(s)}


def compare(configs: dict, X, y, cv=None) -> pd.DataFrame:
    """configs = {label: pipeline}. Same folds for every config, so differences are paired."""
    cv = cv or make_cv()
    rows = {name: evaluate(p, X, y, cv) for name, p in configs.items()}
    return pd.DataFrame(rows).T.sort_values("auroc_mean", ascending=False)


# --------------------------------------------------------------------------- #
# Inspection helpers
# --------------------------------------------------------------------------- #
def high_corr_pairs(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    c = X.corr().abs()
    iu = np.triu_indices_from(c, k=1)
    out = pd.DataFrame({"a": c.columns[iu[0]], "b": c.columns[iu[1]], "abs_corr": c.values[iu]})
    return out[out.abs_corr > threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)


def permutation_importance_table(pipe, X, y, n_repeats: int = 5) -> pd.Series:
    """Out-of-sample permutation importances for a final pipeline (single stratified 75/25 split)."""
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    p = clone(pipe).fit(Xa, ya)
    r = permutation_importance(p, Xb, yb, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def write_submission(pipe, X_train, y_train, X_test, path="submissions/submission.csv") -> pd.DataFrame:
    """Fit on all training data and write predicted positive-class probabilities in the
    sample_submission.csv format (columns: ID, Predicted)."""
    p = clone(pipe).fit(X_train, y_train)
    sub = pd.DataFrame({"ID": X_test.index, "Predicted": p.predict_proba(X_test)[:, 1]})
    sub.to_csv(path, index=False)
    return sub


# --------------------------------------------------------------------------- #
# Hyperparameter search (all scored with fold_scores on one shared CV object)
# --------------------------------------------------------------------------- #
def grid_search(model: str, grid: dict, X, y, cv, **fixed) -> pd.DataFrame:
    """Exhaustive grid. `grid` maps parameter -> list of values; the key "imbalance" (none/weight/smote)
    may be included like any other. `fixed` are model parameters held constant. Sorted best-first."""
    keys, rows = list(grid), []
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(keys, vals))
        imb = cfg.pop("imbalance", "none")
        s = fold_scores(build_pipeline(model, imb, **fixed, **cfg), X, y, cv)
        rows.append({"imbalance": imb, **cfg, "auroc_mean": s.mean(), "auroc_std": s.std()})
    return pd.DataFrame(rows).sort_values("auroc_mean", ascending=False).reset_index(drop=True)


def _random_search(model: str, draw, X, y, cv, n_iter: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_iter):
        params = draw(rng)
        imb = str(rng.choice(["none", "weight"]))
        s = fold_scores(build_pipeline(model, imb, **params), X, y, cv)
        rows.append({"config": i, "imbalance": imb, "params": params,
                     "auroc_mean": s.mean(), "auroc_std": s.std()})
    return pd.DataFrame(rows).sort_values("auroc_mean", ascending=False).reset_index(drop=True)


def xgb_random_search(X, y, cv, n_iter: int = 24, seed: int = 0) -> pd.DataFrame:
    """Random search over XGBoost hyperparameters and class weighting; best config first."""
    def draw(rng):
        return dict(n_estimators=int(rng.choice([150, 300, 500])),
                    max_depth=int(rng.choice([2, 3, 4])),
                    learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.1])),
                    min_child_weight=float(rng.choice([1, 3, 5, 10])),
                    reg_lambda=float(rng.choice([0, 1, 5, 20])),
                    subsample=float(rng.choice([0.8, 1.0])),
                    colsample_bytree=float(rng.choice([0.8, 1.0])))
    return _random_search("xgb", draw, X, y, cv, n_iter, seed)


def hgb_random_search(X, y, cv, n_iter: int = 24, seed: int = 0) -> pd.DataFrame:
    """Same idea for scikit-learn's HistGradientBoosting (its own parameter names); best config first."""
    def draw(rng):
        return dict(max_depth=int(rng.choice([2, 3, 4])),
                    learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.1])),
                    max_iter=int(rng.choice([150, 300, 500])),
                    min_samples_leaf=int(rng.choice([10, 20, 40])),
                    l2_regularization=float(rng.choice([0, 1, 5, 20])))
    return _random_search("hgb", draw, X, y, cv, n_iter, seed)


# --------------------------------------------------------------------------- #
# Final model: configuration lives in results/final_config.json (written by the notebook)
# --------------------------------------------------------------------------- #
CONFIG_PATH = Path(__file__).resolve().parents[1] / "results" / "final_config.json"


def build_ensemble(members: dict) -> RankAverager:
    """members = {"logreg": {"imbalance": ..., "params": {...}}, "rf": {...}, "xgb": {...}, "hgb": {...}}
    -> equal-weight rank average of one pipeline per member."""
    return RankAverager([(name, build_pipeline(name, m["imbalance"], **m["params"]))
                         for name, m in members.items()])


def save_config(features: list, members: dict, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": list(features), "members": members}, indent=2))


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())
