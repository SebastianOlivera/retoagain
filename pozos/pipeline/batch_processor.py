from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from pozos.core.profiler import WellProfiler

REQUIRED_COLUMNS = {"ts", "caudal_ls", "nivel_m"}


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["caudal_ls"] = pd.to_numeric(df["caudal_ls"], errors="coerce")
    df["nivel_m"] = pd.to_numeric(df["nivel_m"], errors="coerce")
    if "estado_bomba" in df.columns:
        df["estado_bomba"] = pd.to_numeric(df["estado_bomba"], errors="coerce")
    df = df.dropna(subset=["ts", "caudal_ls", "nivel_m"]).sort_values("ts").reset_index(drop=True)

    if df.empty:
        raise ValueError("El archivo no tiene filas válidas tras limpieza")
    return df


def infer_threshold(df: pd.DataFrame, min_threshold_ls: float) -> float:
    return float(max(min_threshold_ls, float(df["caudal_ls"].max()) * 0.05))


def process_single_file(
    path: Path,
    smooth_window: int,
    min_threshold_ls: float,
    min_segment_points: int,
    period_days: float,
    aggregate_mode: str,
) -> pd.DataFrame:
    df = load_raw_csv(path)
    threshold = infer_threshold(df, min_threshold_ls=min_threshold_ls)

    profiler = WellProfiler.from_dataframe(
        df=df,
        threshold=threshold,
        min_segment_points=min_segment_points,
        smooth_window=smooth_window,
    )
    profiler.extract_segments()
    profiler.compute_segment_metrics()

    if aggregate_mode == "cycle":
        out = profiler.build_cycle_table(device_id=path.stem)
    else:
        out = profiler.aggregate_periods(period_days=period_days, device_id=path.stem)

    out["archivo_origen"] = path.name
    out["umbral_q_usado_ls"] = round(float(threshold), 3)
    return out


def process_folder(
    input_dir: Path,
    smooth_window: int,
    min_threshold_ls: float,
    min_segment_points: int,
    period_days: float,
    aggregate_mode: str,
) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {input_dir}")

    tables: List[pd.DataFrame] = []
    errors: List[dict] = []

    for file in files:
        try:
            tables.append(
                process_single_file(
                    path=file,
                    smooth_window=smooth_window,
                    min_threshold_ls=min_threshold_ls,
                    min_segment_points=min_segment_points,
                    period_days=period_days,
                    aggregate_mode=aggregate_mode,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"archivo_origen": file.name, "error": str(exc)})

    if not tables:
        raise RuntimeError("No se pudo procesar ningún archivo. Revise errores.csv")

    result = pd.concat(tables, ignore_index=True)
    if errors:
        result.attrs["errors_df"] = pd.DataFrame(errors)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Procesa CSV raw y calcula métricas clave por ciclo o período")
    parser.add_argument("--input_dir", required=True, help="Carpeta con CSV raw")
    parser.add_argument("--output_csv", default="metricas_consolidadas.csv", help="CSV final")
    parser.add_argument("--errors_csv", default="errores_procesamiento.csv", help="CSV de errores")
    parser.add_argument("--smooth_window", type=int, default=1, help="Ventana de mediana para suavizado")
    parser.add_argument("--min_threshold_ls", type=float, default=0.05, help="Umbral mínimo ON")
    parser.add_argument("--min_segment_points", type=int, default=5, help="Puntos mínimos por segmento para fit")
    parser.add_argument("--period_days", type=float, default=2.0, help="Días por período")
    parser.add_argument("--aggregate_mode", choices=["period", "cycle"], default="period")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)

    df = process_folder(
        input_dir=input_dir,
        smooth_window=args.smooth_window,
        min_threshold_ls=args.min_threshold_ls,
        min_segment_points=args.min_segment_points,
        period_days=args.period_days,
        aggregate_mode=args.aggregate_mode,
    )
    df.to_csv(args.output_csv, index=False)

    errors_df = df.attrs.get("errors_df")
    if isinstance(errors_df, pd.DataFrame) and not errors_df.empty:
        errors_df.to_csv(args.errors_csv, index=False)
        print(f"Proceso completado con advertencias. Métricas: {args.output_csv}. Errores: {args.errors_csv}")
    else:
        print(f"Proceso completado. Métricas: {args.output_csv}")


if __name__ == "__main__":
    main()
