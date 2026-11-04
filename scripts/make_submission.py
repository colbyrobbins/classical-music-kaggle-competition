"""Rebuild the final ensemble from results/final_config.json (written by the notebook) and write a Kaggle submission.

    python scripts/make_submission.py             -> submissions/submission.csv
    python scripts/make_submission.py --no-id     -> submissions/submission_no_id.csv (without the account-ID feature)
    python scripts/make_submission.py --out PATH  -> custom output path

Checks the file against data/raw/sample_submission.csv (same IDs in the same order, no NaN, values in 0-1).
Predictions are percentile ranks, which is all AUROC needs.
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_features, load_tables  # noqa: E402
from src.modeling import build_ensemble, load_config, write_submission  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-id", action="store_true", help="drop the account-ID rank feature")
    ap.add_argument("--out", type=Path, help="output path")
    a = ap.parse_args()
    cfg = load_config()
    cols = [c for c in cfg["features"] if not (a.no_id and c == "acct_id_rank")]
    out = a.out or ROOT / "submissions" / ("submission_no_id.csv" if a.no_id else "submission.csv")

    T = load_tables()
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    X_train = build_features(train["account.id"], T)[cols]
    X_test = build_features(test["ID"], T)[cols]
    y = train.set_index("account.id")["label"].loc[X_train.index].to_numpy()

    out.parent.mkdir(parents=True, exist_ok=True)
    sub = write_submission(build_ensemble(cfg["members"]), X_train, y, X_test, path=out)

    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")
    assert (sub["ID"].values == sample["ID"].values).all(), "IDs/order differ from sample_submission"
    assert sub["Predicted"].notna().all() and sub["Predicted"].between(0, 1).all()
    print(f"wrote {out}  {sub.shape}  ({len(cols)} features)")


if __name__ == "__main__":
    main()
