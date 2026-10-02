import argparse
import pandas as pd


def main(log: str) -> None:
    df = pd.read_csv(log)
    print(df.to_string(index=False))
    print("Criterion: pooled-selection recall only. No strict all-layer column in this log. Contexts 958/1474/1990, depths 20/50/80 percent.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="artifacts/logs/experiment_3_niah_summary.csv")
    args = ap.parse_args()
    main(args.log)
