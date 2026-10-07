from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline


def build_model():
    features = FeatureUnion([
        # word pairs: "you idiot", "shut up"
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                 sublinear_tf=True, strip_accents="unicode")),
        # letter chunks: catches misspellings like "id1ot" or "f*ck"
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=200_000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=4.0, solver="liblinear", max_iter=1000)
    return Pipeline([("features", features), ("clf", clf)])


def load(name):
    return pd.read_csv(f"data/processed/{name}.csv").fillna({"text": ""})


def main():
    train, val = load("train"), load("val")

    print("Training... (a few minutes)")
    model = build_model()
    model.fit(train["text"], train["label"])

    y = val["label"].to_numpy()
    p = model.predict_proba(val["text"])[:, 1]

    print(f"\nAccuracy of always saying 'clean': {1 - y.mean():.3f}  <- why accuracy lies")
    print(f"Random-guess PR-AUC:               {y.mean():.3f}")
    print(f"Model PR-AUC:                      {average_precision_score(y, p):.3f}")
    print(f"Model ROC-AUC:                     {roc_auc_score(y, p):.3f}\n")

    print("threshold  precision  recall  flagged")
    for t in [0.3, 0.5, 0.7, 0.9]:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        precision = tp / max(pred.sum(), 1)
        recall = tp / (y == 1).sum()
        print(f"{t:>9.1f}  {precision:>9.3f}  {recall:>6.3f}  {pred.mean():>7.2%}")

    print("\nConfusion matrix @0.5:")
    print("[[clean called clean, clean called toxic]")
    print(" [toxic called clean, toxic called toxic]]")
    print(confusion_matrix(y, p >= 0.5))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/baseline.joblib")
    print("\nSaved models/baseline.joblib")


if __name__ == "__main__":
    main()