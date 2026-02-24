from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pozos.analysis.metricas_por_periodo import compute_period_metrics


def _process_one_file(
    csv_path: Path,
    thr_ls: float,
    smooth_window: int,
    days_per_period: float,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "device_id" not in df.columns:
        df["device_id"] = csv_path.stem

    periods_df, _ = compute_period_metrics(
        df,
        thr_ls=thr_ls,
        smooth_window=smooth_window,
        days_per_period=days_per_period,
    )
    return periods_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera CSV final por períodos de 2 días (o configurable)")
    ap.add_argument("--csv", help="Entrada única con ts, caudal_ls, nivel_m y opcional device_id")
    ap.add_argument("--input_dir", help="Carpeta con múltiples CSV a procesar")
    ap.add_argument("--out_csv", required=True, help="Salida CSV final por período")
    ap.add_argument("--errors_csv", default="errores_procesamiento.csv", help="CSV de errores por archivo")
    ap.add_argument("--thr_ls", type=float, default=0.05)
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--days_per_period", type=float, default=2.0)
    args = ap.parse_args()

    if bool(args.csv) == bool(args.input_dir):
        raise SystemExit("Debes indicar exactamente uno: --csv o --input_dir")

    out_path = Path(args.out_csv)

    if args.csv:
        periods_df = _process_one_file(
            csv_path=Path(args.csv),
            thr_ls=args.thr_ls,
            smooth_window=args.smooth_window,
            days_per_period=args.days_per_period,
        )
        periods_df.to_csv(out_path, index=False, float_format="%.2f", decimal=".")
        print(f"CSV final generado: {out_path}")
        return

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No se encontraron CSV en: {input_dir}")

    tables: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for csv_path in files:
        try:
            part = _process_one_file(
                csv_path=csv_path,
                thr_ls=args.thr_ls,
                smooth_window=args.smooth_window,
                days_per_period=args.days_per_period,
            )
            tables.append(part)
        except Exception as exc:  # noqa: BLE001
            errors.append({"archivo": csv_path.name, "error": str(exc)})

    if not tables:
        raise SystemExit("No se pudo procesar ningún CSV de la carpeta")

    final_df = pd.concat(tables, ignore_index=True)
    final_df.to_csv(out_path, index=False, float_format="%.2f", decimal=".")

    if errors:
        pd.DataFrame(errors).to_csv(args.errors_csv, index=False)
        print(f"CSV generado: {out_path}. Con errores, ver: {args.errors_csv}")
    else:
        print(f"CSV final generado: {out_path}")


if __name__ == "__main__":
    main()
