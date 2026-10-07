from pathlib import Path

import joblib
import numpy as np
import pandas as pd

N = 15  # examples per section


def drivers(model, names, coef, text, direction, k=5):
    """Words/letter-chunks that pushed this comment's score the most.
    direction=+1: pushed toward toxic, -1: pushed toward clean."""
    x = model.named_steps["features"].transform([text]).tocsr()
    contrib = x.data * coef[x.indices]
    order = np.argsort(contrib * direction)[::-1][:k]
    return ", ".join(f"'{names[x.indices[i]]}' ({contrib[i]:+.2f})" for i in order)


def section(rows, title, explanation, model, names, coef, direction):
    lines = [f"## {title}\n", f"_{explanation}_\n"]
    for _, r in rows.iterrows():
        text = " ".join(str(r["text"]).split())[:300]
        lines.append(f"**score {r['score']:.3f}**\n> {text}\n")
        lines.append(f"drivers: {drivers(model, names, coef, r['text'], direction)}\n")
    return "\n".join(lines)


def main():
    val = pd.read_csv("data/processed/val.csv").fillna({"text": ""})
    model = joblib.load("models/baseline.joblib")
    names = model.named_steps["features"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]

    val["score"] = model.predict_proba(val["text"])[:, 1]
    false_pos = val[val["label"] == 0].nlargest(N, "score")
    false_neg = val[val["label"] == 1].nsmallest(N, "score")

    report = "\n\n".join([
        "# Error analysis (validation set, baseline model)",
        section(false_pos, "False positives: clean comments scored as toxic",
                "Drivers = what pushed the score UP.", model, names, coef, +1),
        section(false_neg, "False negatives: toxic comments scored as clean",
                "Drivers = what pulled the score DOWN.", model, names, coef, -1),
    ])

    out = Path("reports/error_analysis.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out} ({N} false positives, {N} false negatives)")


if __name__ == "__main__":
    main()