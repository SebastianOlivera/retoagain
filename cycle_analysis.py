#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def robust_median_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).median()


def detect_depth_like(nivel: np.ndarray, pump_on: np.ndarray) -> bool:
    """Si durante ON el nivel numérico sube, normalmente es profundidad."""
    if pump_on.sum() < 10 or (~pump_on).sum() < 10:
        return False
    return np.nanmedian(nivel[pump_on]) > np.nanmedian(nivel[~pump_on])


def integrate_volume(c_m3s: np.ndarray, dt_s: np.ndarray) -> float:
    dt_s = np.nan_to_num(dt_s, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(c_m3s * dt_s))


def mean_std(series: pd.Series) -> Tuple[float, float]:
    """Media y desviación estándar muestral (ddof=1)."""
    values = pd.to_numeric(series, errors="coerce")
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


@dataclass
class PeriodRow:
    periodo: int
    inicio: str
    fin: str
    n_muestras: int
    duracion_s: float

    # Métricas globales del período.
    porcentaje_on: float
    volumen_bombeado_m3: float

    # Promedio + std de variables base.
    caudal_prom_ls: float
    caudal_std_ls: float
    caudal_on_prom_ls: float
    caudal_on_std_ls: float
    caudal_off_prom_ls: float
    caudal_off_std_ls: float

    nivel_prom_m: float
    nivel_std_m: float
    nivel_f_prom_m: float
    nivel_f_std_m: float

    # h invierte signo si se detectó profundidad.
    h_prom_m: float
    h_std_m: float


def calc_stats(df_part: pd.DataFrame, col: str) -> Tuple[float, float]:
    return mean_std(df_part[col])


def summarize_periods(periods_df: pd.DataFrame, days_per_period: float, depth_like: bool) -> pd.DataFrame:
    summary: Dict[str, float] = {
        "n_periodos": int(len(periods_df)),
        "dias_por_periodo": float(days_per_period),
        "nivel_es_profundidad_detectado": bool(depth_like),
    }

    numeric_cols = periods_df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"periodo"}
    for col in numeric_cols:
        if col in excluded:
            continue
        col_mean, col_std = mean_std(periods_df[col])
        summary[f"{col}_prom"] = col_mean
        summary[f"{col}_std"] = col_std

    return pd.DataFrame([summary])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Agrupa la serie temporal en períodos fijos de N días (por ejemplo 2 días), "
            "calcula métricas por período y exporta promedios + desvíos estándar."
        )
    )
    ap.add_argument("--csv", required=True, help="CSV con columnas ts, caudal_ls, nivel_m")
    ap.add_argument("--thr_ls", type=float, default=0.05, help="Umbral L/s para marcar bomba ON")
    ap.add_argument("--smooth_window", type=int, default=5, help="Ventana de mediana para nivel")
    ap.add_argument(
        "--days_per_period",
        type=float,
        default=2.0,
        help="Cantidad de días por período. Ejemplo: 2 para ventanas de 2 días.",
    )
    ap.add_argument("--out_periods_csv", required=True, help="Salida CSV con métricas por período")
    ap.add_argument(
        "--out_summary_csv",
        default="",
        help="Salida CSV resumen (promedio y desviación estándar de cada métrica)",
    )
    args = ap.parse_args()

    if args.days_per_period <= 0:
        raise SystemExit("--days_per_period debe ser > 0")

    df = pd.read_csv(args.csv)
    required_cols = {"ts", "caudal_ls", "nivel_m"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"Faltan columnas. Requiere {required_cols}")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    # dt para integraciones de caudal.
    df["dt_s"] = df["ts"].diff().dt.total_seconds()
    if len(df) > 1:
        default_dt = np.nanmedian(df["dt_s"].iloc[1:].to_numpy())
    else:
        default_dt = 0.0
    df.loc[0, "dt_s"] = default_dt

    # Variables derivadas de soporte.
    df["nivel_f"] = robust_median_smooth(df["nivel_m"], window=args.smooth_window)
    df["pump_on"] = df["caudal_ls"] > args.thr_ls

    depth_like = detect_depth_like(df["nivel_f"].to_numpy(float), df["pump_on"].to_numpy(bool))
    df["h"] = -df["nivel_f"] if depth_like else df["nivel_f"]

    # Agrupación temporal por períodos fijos elegidos por el usuario.
    start_ts = df["ts"].min()
    period_s = args.days_per_period * 86400.0
    elapsed_s = (df["ts"] - start_ts).dt.total_seconds()
    df["periodo"] = (elapsed_s // period_s).astype(int) + 1

    rows: List[PeriodRow] = []
    grouped = df.groupby("periodo", sort=True)

    for periodo, part in grouped:
        part = part.copy()
        on_part = part[part["pump_on"]]
        off_part = part[~part["pump_on"]]

        caudal_mean, caudal_std = calc_stats(part, "caudal_ls")
        nivel_mean, nivel_std = calc_stats(part, "nivel_m")
        nivel_f_mean, nivel_f_std = calc_stats(part, "nivel_f")
        h_mean, h_std = calc_stats(part, "h")

        caudal_on_mean, caudal_on_std = calc_stats(on_part, "caudal_ls") if len(on_part) else (np.nan, np.nan)
        caudal_off_mean, caudal_off_std = calc_stats(off_part, "caudal_ls") if len(off_part) else (np.nan, np.nan)

        porcentaje_on = float(part["pump_on"].mean() * 100.0)
        volumen_m3 = integrate_volume(part["caudal_ls"].to_numpy(float) / 1000.0, part["dt_s"].to_numpy(float))

        rows.append(
            PeriodRow(
                periodo=int(periodo),
                inicio=str(part["ts"].iloc[0]),
                fin=str(part["ts"].iloc[-1]),
                n_muestras=int(len(part)),
                duracion_s=float(np.nansum(part["dt_s"].to_numpy())),
                porcentaje_on=porcentaje_on,
                volumen_bombeado_m3=float(volumen_m3),
                caudal_prom_ls=float(caudal_mean),
                caudal_std_ls=float(caudal_std),
                caudal_on_prom_ls=float(caudal_on_mean),
                caudal_on_std_ls=float(caudal_on_std),
                caudal_off_prom_ls=float(caudal_off_mean),
                caudal_off_std_ls=float(caudal_off_std),
                nivel_prom_m=float(nivel_mean),
                nivel_std_m=float(nivel_std),
                nivel_f_prom_m=float(nivel_f_mean),
                nivel_f_std_m=float(nivel_f_std),
                h_prom_m=float(h_mean),
                h_std_m=float(h_std),
            )
        )

    periods_df = pd.DataFrame([asdict(r) for r in rows])
    periods_df.to_csv(args.out_periods_csv, index=False, encoding="utf-8")

    if args.out_summary_csv:
        summary_df = summarize_periods(periods_df, days_per_period=args.days_per_period, depth_like=depth_like)
        summary_df.to_csv(args.out_summary_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
