from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pozos.core.profiler import WellProfiler

REQUIRED_COLUMNS = {"ts", "nivel_m"}


def _to_m3s(series: pd.Series, unit: str) -> pd.Series:
    if unit == "m3s":
        return series
    if unit == "ls":
        return series / 1000.0
    if unit == "lmin":
        return series / 1000.0 / 60.0
    if unit == "lh":
        return series / 1000.0 / 3600.0
    raise ValueError(f"Unidad de caudal no soportada: {unit}")


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["nivel_m"] = pd.to_numeric(df["nivel_m"], errors="coerce")

    flow_unit = None
    flow_col = None
    for candidate, unit in (("caudal_m3s", "m3s"), ("caudal_ls", "ls"), ("caudal_lmin", "lmin"), ("caudal_lh", "lh")):
        if candidate in df.columns:
            flow_col = candidate
            flow_unit = unit
            break

    if flow_col is None or flow_unit is None:
        raise ValueError("Falta columna de caudal. Se esperaba una de: caudal_m3s, caudal_ls, caudal_lmin, caudal_lh")

    df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")
    df["caudal_m3s"] = _to_m3s(df[flow_col], flow_unit)

    if flow_col == "caudal_ls":
        valid = df[["caudal_ls", "caudal_m3s"]].dropna()
        if not valid.empty and not np.allclose(valid["caudal_m3s"].to_numpy(), valid["caudal_ls"].to_numpy() / 1000.0):
            raise AssertionError("Conversión inválida: caudal_m3s debe ser caudal_ls/1000")

    df = df.dropna(subset=["ts", "caudal_m3s", "nivel_m"]).sort_values("ts").reset_index(drop=True)

    if df.empty:
        raise ValueError("El archivo no tiene filas válidas tras limpieza")
    return df


def infer_threshold(df: pd.DataFrame, min_threshold_m3s: float) -> float:
    return float(max(min_threshold_m3s, float(df["caudal_m3s"].max()) * 0.05))


def infer_well_identity(path: Path) -> tuple[str, str]:
    stem = path.stem.strip()
    for sep in ("_", "-", " "):
        if sep in stem:
            well_id, well_name = stem.split(sep, 1)
            well_id = well_id.strip()
            well_name = well_name.strip()
            if well_id and well_name:
                return well_id, well_name
    return stem, stem


def process_single_file(
    path: Path,
    smooth_window: int,
    min_threshold_m3s: float,
    min_cycle_points: int,
) -> Dict:
    df = load_raw_csv(path)
    threshold = infer_threshold(df, min_threshold_m3s=min_threshold_m3s)

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
    well_id, well_name = infer_well_identity(path)
    features["pozo_id"] = well_id
    features["pozo_nombre"] = well_name
    features["archivo_origen"] = path.name
    features["fecha_inicio"] = df["ts"].iloc[0]
    features["fecha_fin"] = df["ts"].iloc[-1]
    features["n_muestras"] = int(len(df))
    features["n_ciclos"] = int(0 if profiler.cycles_df is None else len(profiler.cycles_df))
    return features


def process_folder(
    input_dir: Path,
    smooth_window: int,
    min_threshold_m3s: float,
    min_cycle_points: int,
) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {input_dir}")

    rows: List[Dict] = []
    errors: List[Dict] = []

    for file in files:
        try:
            rows.append(process_single_file(file, smooth_window, min_threshold_m3s, min_cycle_points))
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
    parser.add_argument("--min_threshold_m3s", type=float, default=0.0005, help="Umbral mínimo para bomba ON en m3/s")
    parser.add_argument("--min_cycle_points", type=int, default=5, help="Puntos mínimos por ciclo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)

    df = process_folder(
        input_dir=input_dir,
        smooth_window=args.smooth_window,
        min_threshold_m3s=args.min_threshold_m3s,
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
