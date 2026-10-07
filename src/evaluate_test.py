import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from thresholds import describe


def main():
    test = pd.read_csv("data/processed/test.csv").fillna({"text": ""})
    y = test["label"].to_numpy()
    p = joblib.load("models/baseline.joblib").predict_proba(test["text"])[:, 1]
    t = json.loads(Path("models/thresholds.json").read_text())

    print(f"TEST SET: {len(y):,} comments, {y.mean():.2%} toxic")
    print(f"Random-guess PR-AUC: {y.mean():.3f}")
    print(f"PR-AUC:  {average_precision_score(y, p):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y, p):.3f}\n")
    print("Frozen thresholds from validation:")
    describe(y, p, t["hold"], t["review"])


if __name__ == "__main__":
    main()