import argparse
import pandas as pd


FAULTED = {"narrativeqa", "hotpotqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en"}


def main(log: str) -> None:
    df = pd.read_csv(log)
    print(df.to_string(index=False))
    completed = df[~df["Task"].isin(FAULTED)]
    print(f"completed={len(completed)} mean={completed['PageEntroKV_Score'].mean():.2f} (completions only, not a benchmark claim)")
    print("Faulted tasks are excluded as harness non-completions, not 0.00 accuracy. No dense baseline in this log.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="artifacts/logs/experiment_5_longbench_summary.csv")
    args = ap.parse_args()
    main(args.log)
