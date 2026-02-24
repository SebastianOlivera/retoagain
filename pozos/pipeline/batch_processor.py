from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pozos.core.profiler import WellProfiler

REQUIRED_COLUMNS = {"ts", "caudal_ls", "nivel_m"}


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["caudal_ls"] = pd.to_numeric(df["caudal_ls"], errors="coerce")
    df["nivel_m"] = pd.to_numeric(df["nivel_m"], errors="coerce")
    df = df.dropna(subset=["ts", "caudal_ls", "nivel_m"]).sort_values("ts").reset_index(drop=True)

    if df.empty:
        raise ValueError("El archivo no tiene filas válidas tras limpieza")
    return df


def infer_threshold(df: pd.DataFrame, min_threshold_ls: float) -> float:
    return float(max(min_threshold_ls, float(df["caudal_ls"].max()) * 0.05))


def process_single_file(path: Path, smooth_window: int, min_threshold_ls: float, min_cycle_points: int) -> Dict:
    df = load_raw_csv(path)
    threshold = infer_threshold(df, min_threshold_ls=min_threshold_ls)

    profiler = WellProfiler.from_dataframe(
        df=df,
        threshold=threshold,
        min_cycle_points=min_cycle_points,
        smooth_window=smooth_window,
    )
    profiler.extract_cycles()
    profiler.compute_global_metrics()
    profiler.classify_regime()

    features = profiler.get_features()
    features["archivo_origen"] = path.name
    features["fecha_inicio"] = df["ts"].iloc[0]
    features["fecha_fin"] = df["ts"].iloc[-1]
    features["n_muestras"] = int(len(df))
    features["n_ciclos"] = int(0 if profiler.cycles_df is None else len(profiler.cycles_df))
    return features


def process_folder(input_dir: Path, smooth_window: int, min_threshold_ls: float, min_cycle_points: int) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {input_dir}")

    rows: List[Dict] = []
    errors: List[Dict] = []

    for file in files:
        try:
            rows.append(process_single_file(file, smooth_window, min_threshold_ls, min_cycle_points))
        except Exception as exc:  # noqa: BLE001
            errors.append({"archivo_origen": file.name, "error": str(exc)})

    if not rows:
        raise RuntimeError("No se pudo procesar ningún archivo. Revise errores.csv")

    result = pd.DataFrame(rows)
    if errors:
        result.attrs["errors_df"] = pd.DataFrame(errors)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Procesa todos los CSV raw de una carpeta y consolida parámetros")
    parser.add_argument("--input_dir", required=True, help="Carpeta con CSV raw")
    parser.add_argument("--output_csv", default="parametros_consolidados.csv", help="CSV final de parámetros")
    parser.add_argument("--errors_csv", default="errores_procesamiento.csv", help="CSV con archivos fallidos")
    parser.add_argument("--smooth_window", type=int, default=5, help="Ventana de mediana para nivel")
    parser.add_argument("--min_threshold_ls", type=float, default=0.5, help="Umbral mínimo para bomba ON")
    parser.add_argument("--min_cycle_points", type=int, default=5, help="Puntos mínimos por ciclo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)

    df = process_folder(
        input_dir=input_dir,
        smooth_window=args.smooth_window,
        min_threshold_ls=args.min_threshold_ls,
        min_cycle_points=args.min_cycle_points,
    )
    df.to_csv(args.output_csv, index=False)

    errors_df = df.attrs.get("errors_df")
    if isinstance(errors_df, pd.DataFrame) and not errors_df.empty:
        errors_df.to_csv(args.errors_csv, index=False)
        print(f"Proceso completado con advertencias. Parámetros: {args.output_csv}. Errores: {args.errors_csv}")
    else:
        print(f"Proceso completado. Parámetros: {args.output_csv}")


if __name__ == "__main__":
    main()
