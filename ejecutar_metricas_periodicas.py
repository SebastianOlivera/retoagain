from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pozos.analysis.metricas_por_periodo import compute_cycle_metrics, compute_period_metrics


def _parse_well_and_id_from_filename(stem: str) -> tuple[str, str]:
    txt = str(stem).strip()
    if not txt:
        return "", ""
    if "_" not in txt:
        return txt, txt
    well_name, device_id = txt.rsplit("_", 1)
    return well_name.strip(), device_id.strip()


def _is_placeholder_series(sr: pd.Series, placeholder: str) -> bool:
    vals = sr.astype(str).str.strip().str.lower()
    vals = vals[vals != ""]
    if vals.empty:
        return True
    return bool(vals.nunique() == 1 and vals.iloc[0] == placeholder.lower())


def _process_one_file(
    csv_path: Path,
    thr_m3s: float,
    smooth_window: int,
    days_per_period: float,
    level_convention: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    parsed_name, parsed_id = _parse_well_and_id_from_filename(csv_path.stem)

    if "device_id" not in df.columns:
        df["device_id"] = parsed_id or csv_path.stem
    elif _is_placeholder_series(df["device_id"], "device_id"):
        df["device_id"] = parsed_id or csv_path.stem

    if "nombre_pozo" not in df.columns:
        df["nombre_pozo"] = parsed_name or str(df["device_id"].iloc[0])
    elif _is_placeholder_series(df["nombre_pozo"], "nombre_pozo"):
        df["nombre_pozo"] = parsed_name or str(df["device_id"].iloc[0])

    periods_df, _ = compute_period_metrics(
        df,
        thr_m3s=thr_m3s,
        smooth_window=smooth_window,
        days_per_period=days_per_period,
        level_convention=level_convention,
    )
    cycles_df = compute_cycle_metrics(df, thr_m3s=thr_m3s, smooth_window=smooth_window, level_convention=level_convention)
    return periods_df, cycles_df


def _resolve_threshold_m3s(args: argparse.Namespace) -> float:
    if args.thr_m3s is not None:
        return float(args.thr_m3s)
    if args.thr_ls is not None:
        return float(args.thr_ls) / 1000.0
    return 0.00005


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera CSV final por períodos (2 días por defecto)")
    ap.add_argument("--csv", help="Entrada única con ts, nivel_m y caudal_ls/caudal_m3s")
    ap.add_argument("--input_dir", help="Carpeta con múltiples CSV a procesar")
    ap.add_argument("--out_csv", required=True, help="Salida CSV final por período")
    ap.add_argument("--out_cycles_csv", help="Salida opcional con métricas por ciclo")
    ap.add_argument("--errors_csv", default="errores_procesamiento.csv", help="CSV de errores por archivo")
    ap.add_argument("--thr_m3s", type=float, help="Umbral de encendido en m3/s")
    ap.add_argument("--thr_ls", type=float, help="Umbral legacy en L/s (se convierte a m3/s)")
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--days_per_period", type=float, default=2.0)
    ap.add_argument("--level_convention", choices=["depth", "height"], default="depth")
    args = ap.parse_args()

    if bool(args.csv) == bool(args.input_dir):
        raise SystemExit("Debes indicar exactamente uno: --csv o --input_dir")

    thr_m3s = _resolve_threshold_m3s(args)
    out_path = Path(args.out_csv)

    if args.csv:
        periods_df, cycles_df = _process_one_file(
            csv_path=Path(args.csv),
            thr_m3s=thr_m3s,
            smooth_window=args.smooth_window,
            days_per_period=args.days_per_period,
            level_convention=args.level_convention,
        )
        periods_df.to_csv(out_path, index=False, float_format="%.2f", decimal=".")
        if args.out_cycles_csv:
            cycles_df.to_csv(Path(args.out_cycles_csv), index=False, float_format="%.2f", decimal=".")
        print(f"CSV final generado: {out_path}")
        return

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No se encontraron CSV en: {input_dir}")

    period_tables: list[pd.DataFrame] = []
    cycle_tables: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for csv_path in files:
        try:
            part_periods, part_cycles = _process_one_file(
                csv_path=csv_path,
                thr_m3s=thr_m3s,
                smooth_window=args.smooth_window,
                days_per_period=args.days_per_period,
                level_convention=args.level_convention,
            )
            period_tables.append(part_periods)
            if args.out_cycles_csv:
                cycle_tables.append(part_cycles)
        except Exception as exc:  # noqa: BLE001
            errors.append({"archivo": csv_path.name, "error": str(exc)})

    if not period_tables:
        raise SystemExit("No se pudo procesar ningún CSV de la carpeta")

    final_df = pd.concat(period_tables, ignore_index=True)
    final_df.to_csv(out_path, index=False, float_format="%.2f", decimal=".")

    if args.out_cycles_csv and cycle_tables:
        pd.concat(cycle_tables, ignore_index=True).to_csv(Path(args.out_cycles_csv), index=False, float_format="%.2f", decimal=".")

    if errors:
        pd.DataFrame(errors).to_csv(args.errors_csv, index=False)
        print(f"CSV generado: {out_path}. Con errores, ver: {args.errors_csv}")
    else:
        print(f"CSV final generado: {out_path}")


if __name__ == "__main__":
    main()
