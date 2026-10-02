import argparse
import pandas as pd


def main(log: str) -> None:
    df = pd.read_csv(log)
    valid = df.dropna(subset=["d_jaccard", "uor_empirical", "page_entrokv_uor"])
    print(f"rows={len(df)} valid={len(valid)}")
    print(valid[["d_jaccard", "uor_empirical", "page_entrokv_uor"]].describe().to_string())
    print("Note: supplied pilot log is a consensus null (all D=0.0, UOR=1.0). No bloat claim follows.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="artifacts/logs/experiment_1_structural_bounds_real.csv")
    args = ap.parse_args()
    main(args.log)
