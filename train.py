"""
Training script: generate data, tune, train, evaluate, and save the upsell classifier.
Usage: python train.py [--n_accounts 10000] [--n_trials 50]
"""
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from data.synthetic_accounts import generate_accounts
from models.xgboost_classifier import UpsellClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_accounts", type=int, default=10_000)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)

    print("Generating account data...")
    df = generate_accounts(n=args.n_accounts)
    y = df["upsold"]
    df_features = df.drop(columns=["account_id", "upsold"])

    df_train, df_test, y_train, y_test = train_test_split(
        df_features, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

    model = UpsellClassifier(n_trials=args.n_trials)
    print("\nTuning hyperparameters...")
    model.tune(df_train, y_train)

    print("\nFitting final model...")
    model.fit(df_train, y_train)

    print("\nEvaluation:")
    metrics = model.evaluate(df_test, y_test)

    print("\nDecile Lift Table:")
    lift = model.decile_lift(df_test, y_test)
    print(lift.to_string(float_format="{:.3f}".format))

    model.save(f"{args.output}/upsell_classifier.pkl")
    with open(f"{args.output}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Model and metrics saved to {args.output}/")


if __name__ == "__main__":
    main()