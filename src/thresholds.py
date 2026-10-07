import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

HOLD_PRECISION = 0.95   # held comments must be >= 95% truly toxic
REVIEW_RECALL = 0.90    # hold + review must catch >= 90% of toxic content


def describe(y, p, t_hold, t_review):
    pos = y == 1
    hold = p >= t_hold
    review = (p >= t_review) & ~hold
    publish = p < t_review
    n = len(y)
    print(f"  auto-hold   (score >= {t_hold:.3f}): {hold.mean():6.2%} of comments,"
          f" {pos[hold].mean():.1%} truly toxic, {np.sum(hold & ~pos)} clean wrongly held")
    print(f"  review      ({t_review:.3f}-{t_hold:.3f}): {review.mean():6.2%} of comments,"
          f" {pos[review].mean():.1%} truly toxic")
    print(f"  publish     (score <  {t_review:.3f}): {publish.mean():6.2%} of comments,"
          f" {np.sum(publish & pos)} toxic slipped through")
    print(f"  toxic caught (hold + review): {np.sum(pos & ~publish) / pos.sum():.1%}")
    print(f"  moderator workload: ~{review.mean() * 1000:.0f} reviews per 1,000 comments")


def main():
    val = pd.read_csv("data/processed/val.csv").fillna({"text": ""})
    y = val["label"].to_numpy()
    p = joblib.load("models/baseline.joblib").predict_proba(val["text"])[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds

    t_hold = float(thresholds[np.where(precision >= HOLD_PRECISION)[0][0]])
    t_review = float(thresholds[np.where(recall >= REVIEW_RECALL)[0][-1]])

    print(f"Targets: hold precision >= {HOLD_PRECISION:.0%}, overall recall >= {REVIEW_RECALL:.0%}\n")
    describe(y, p, t_hold, t_review)

    print("\nWhat other recall targets would cost (review queue size):")
    print("  recall target  review cutoff  reviews per 1,000")
    for target in [0.80, 0.85, 0.90, 0.95]:
        t = thresholds[np.where(recall >= target)[0][-1]]
        queue = np.mean((p >= t) & (p < t_hold)) * 1000
        print(f"  {target:>13.0%}  {t:>13.3f}  {queue:>17.0f}")

    out = Path("models/thresholds.json")
    out.write_text(json.dumps({"hold": t_hold, "review": t_review,
                               "targets": {"hold_precision": HOLD_PRECISION,
                                           "review_recall": REVIEW_RECALL}}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()