from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EMB = Path("data/embeddings")
N = 15  # same worst-error sets as error_analysis.py


def load(name):
    return pd.read_csv(f"data/processed/{name}.csv").fillna({"text": ""})


def summary(label, y, p):
    print(f"\n=== {label} ===")
    print(f"PR-AUC {average_precision_score(y, p):.3f} | ROC-AUC {roc_auc_score(y, p):.3f}")
    print("threshold  precision  recall  flagged")
    for t in [0.3, 0.5, 0.7, 0.9]:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        print(f"{t:>9.1f}  {tp / max(pred.sum(), 1):>9.3f}  {tp / (y == 1).sum():>6.3f}"
              f"  {pred.mean():>7.2%}")


def comparison_rows(rows):
    lines = ["| baseline | embedding | comment |", "|---|---|---|"]
    for _, r in rows.iterrows():
        text = " ".join(str(r["text"]).split())[:120].replace("|", "/")
        lines.append(f"| {r['base']:.3f} | {r['emb']:.3f} | {text} |")
    return "\n".join(lines)


def main():
    train, val = load("train"), load("val")
    y_train, y = train["label"].to_numpy(), val["label"].to_numpy()

    print("Training logistic regression on embeddings...")
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(np.load(EMB / "train.npy"), y_train)

    p_emb = clf.predict_proba(np.load(EMB / "val.npy"))[:, 1]
    p_base = joblib.load("models/baseline.joblib").predict_proba(val["text"])[:, 1]
    p_combo = (p_base + p_emb) / 2

    print(f"\nRandom-guess PR-AUC: {y.mean():.3f}")
    summary("Baseline (TF-IDF)", y, p_base)
    summary("Embedding model", y, p_emb)
    summary("Average of both", y, p_combo)

    # Did the embedding model fix the baseline's worst errors?
    val["base"], val["emb"] = p_base, p_emb
    fp = val[val["label"] == 0].nlargest(N, "base")
    fn = val[val["label"] == 1].nsmallest(N, "base")
    print(f"\nBaseline's {N} worst false positives: embedding scores {(fp['emb'] < 0.5).sum()} below 0.5")
    print(f"Baseline's {N} worst false negatives: embedding scores {(fn['emb'] >= 0.5).sum()} at/above 0.5")

    out = Path("reports/model_comparison.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n\n".join([
        "# Baseline vs embedding on the baseline's worst errors",
        "## False positives (labeled clean)", comparison_rows(fp),
        "## False negatives (labeled toxic)", comparison_rows(fn),
    ]), encoding="utf-8")
    print(f"Wrote {out}")

    joblib.dump(clf, "models/embed_lr.joblib")


if __name__ == "__main__":
    main()