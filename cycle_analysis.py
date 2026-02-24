from __future__ import annotations

import argparse
import pandas as pd

from pozos.analysis.periodic import compute_period_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Calcula métricas por período fijo")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--thr_ls", type=float, default=0.05)
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--days_per_period", type=float, default=2.0)
    ap.add_argument("--out_periods_csv", required=True)
    ap.add_argument("--out_summary_csv", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    periods_df, summary_df = compute_period_metrics(
        df,
        thr_ls=args.thr_ls,
        smooth_window=args.smooth_window,
        days_per_period=args.days_per_period,
    )
    periods_df.to_csv(args.out_periods_csv, index=False)
    if args.out_summary_csv:
        summary_df.to_csv(args.out_summary_csv, index=False)


if __name__ == "__main__":
    main()
