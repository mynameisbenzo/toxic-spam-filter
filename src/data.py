from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/train.csv")
OUT = Path("data/processed")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42


def main():
    df = pd.read_csv(RAW)
    df["label"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    df = df[["id", "comment_text", "label"]].rename(columns={"comment_text": "text"})

    print(f"Total: {len(df):,} comments, {df['label'].mean():.2%} toxic\n")

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:>5}: {len(split):>7,} rows, {split['label'].mean():.2%} toxic")


if __name__ == "__main__":
    main()