from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pozos.analysis.metricas_por_periodo import compute_period_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera CSV final por períodos de 2 días (o configurable)")
    ap.add_argument("--csv", required=True, help="Entrada con ts, caudal_ls, nivel_m y opcional device_id")
    ap.add_argument("--out_csv", required=True, help="Salida CSV final por período")
    ap.add_argument("--thr_ls", type=float, default=0.05)
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--days_per_period", type=float, default=2.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    periods_df, _ = compute_period_metrics(
        df,
        thr_ls=args.thr_ls,
        smooth_window=args.smooth_window,
        days_per_period=args.days_per_period,
    )

    out_path = Path(args.out_csv)
    periods_df.to_csv(out_path, index=False, float_format="%.2f", decimal=".")
    print(f"CSV final generado: {out_path}")


if __name__ == "__main__":
    main()
