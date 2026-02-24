#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def robust_median_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).median()


def detect_depth_like(nivel: np.ndarray, pump_on: np.ndarray) -> bool:
    """Si durante ON el nivel numérico sube, normalmente es profundidad."""
    if pump_on.sum() < 10 or (~pump_on).sum() < 10:
        return False
    return np.nanmedian(nivel[pump_on]) > np.nanmedian(nivel[~pump_on])


def detect_pump_state(caudal_ls: pd.Series, thr_ls: float) -> Tuple[pd.Series, float, bool]:
    """
    Detecta estado ON/OFF con fallback adaptativo.

    Caso normal:
      - ON si caudal > thr_ls.

    Caso borde solicitado:
      - Si todo queda ON (p. ej. caudal siempre > 0), se define un umbral
        adaptativo entre cuantiles 20% y 80% para separar caudal bajo/alto.
    """
    base_state = caudal_ls > thr_ls
    if base_state.any() and (~base_state).any():
        return base_state, float(thr_ls), False

    q20 = float(np.nanquantile(caudal_ls.to_numpy(float), 0.20))
    q80 = float(np.nanquantile(caudal_ls.to_numpy(float), 0.80))
    if not np.isfinite(q20) or not np.isfinite(q80) or q80 <= q20:
        return base_state, float(thr_ls), False

    adaptive_thr = q20 + 0.5 * (q80 - q20)
    adaptive_state = caudal_ls > adaptive_thr
    if not (adaptive_state.any() and (~adaptive_state).any()):
        return base_state, float(thr_ls), False

    return adaptive_state, float(adaptive_thr), True


def detect_segments(state: np.ndarray) -> List[Tuple[int, int, bool]]:
    segs: List[Tuple[int, int, bool]] = []
    s = 0
    cur = bool(state[0])
    for i in range(1, len(state)):
        if bool(state[i]) != cur:
            segs.append((s, i - 1, cur))
            s = i
            cur = bool(state[i])
    segs.append((s, len(state) - 1, cur))
    return segs


def fit_exp_closed_form(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, bool]:
    """Ajusta y(t)=y_inf+(y0-y_inf)*exp(-(t-t0)/tau) y retorna (y_inf, tau, ok)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 5 or len(y) < 5 or not np.all(np.isfinite(y)):
        return np.nan, np.nan, False

    y0 = float(y[0])
    y_inf0 = float(np.nanmedian(y[-max(5, len(y) // 5) :]))
    tau0 = max(float(t[-1] - t[0]) / 3.0, 1.0)
    x0 = np.array([y_inf0, np.log(tau0)], dtype=float)

    def residuals(x):
        y_inf, logtau = x
        tau = np.exp(logtau)
        y_hat = y_inf + (y0 - y_inf) * np.exp(-(t - t[0]) / tau)
        return y - y_hat

    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    rng = abs(ymax - ymin) + 1e-6
    lb = np.array([ymin - 10 * rng, np.log(1e-3)], dtype=float)
    ub = np.array([ymax + 10 * rng, np.log(1e9)], dtype=float)

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            loss="huber",
            f_scale=1.0,
            max_nfev=10000,
        )
    except Exception:
        return np.nan, np.nan, False

    y_inf_fit, logtau_fit = res.x
    tau_fit = float(np.exp(logtau_fit))
    ok = bool(res.success and np.isfinite(tau_fit))
    return float(y_inf_fit), tau_fit, ok


def integrate_volume(c_m3s: np.ndarray, dt_s: np.ndarray) -> float:
    dt_s = np.nan_to_num(dt_s, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(c_m3s * dt_s))


def mean_std(series: pd.Series) -> Tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce")
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


@dataclass
class PeriodRow:
    periodo: int
    inicio: str
    fin: str
    n_muestras: int
    duracion_s: float

    porcentaje_on: float
    volumen_bombeado_m3: float

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
    h_prom_m: float
    h_std_m: float

    # Parámetros solicitados
    h_static_nivel_m: float
    tau_s: float
    k_m2_s: float
    A_m2: float


def calc_stats(df_part: pd.DataFrame, col: str) -> Tuple[float, float]:
    return mean_std(df_part[col])


def estimate_model_params_for_period(part: pd.DataFrame, depth_like: bool) -> Tuple[float, float, float, float]:
    """Calcula h_static, tau, k y A para un período."""
    if len(part) < 10:
        return np.nan, np.nan, np.nan, np.nan

    segs = detect_segments(part["pump_on"].to_numpy(bool))
    if not segs:
        return np.nan, np.nan, np.nan, np.nan

    # Elegimos los segmentos OFF y ON más largos dentro del período.
    off_cands: List[Tuple[float, int, int]] = []
    on_cands: List[Tuple[float, int, int]] = []
    for s, e, on_state in segs:
        dur = float(np.nansum(part.iloc[s : e + 1]["dt_s"].to_numpy()))
        if on_state:
            on_cands.append((dur, s, e))
        else:
            off_cands.append((dur, s, e))

    if not off_cands or not on_cands:
        return np.nan, np.nan, np.nan, np.nan

    _, off_s, off_e = max(off_cands, key=lambda x: x[0])
    _, on_s, on_e = max(on_cands, key=lambda x: x[0])

    off_seg = part.iloc[off_s : off_e + 1]
    on_seg = part.iloc[on_s : on_e + 1]

    t_off = (off_seg["ts"] - off_seg["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
    t_on = (on_seg["ts"] - on_seg["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
    h_off = off_seg["h"].to_numpy(float)
    h_on = on_seg["h"].to_numpy(float)

    h_static_fit, tau_off, ok_off = fit_exp_closed_form(t_off, h_off)
    h_dyn_fit, tau_on, ok_on = fit_exp_closed_form(t_on, h_on)

    # tau representativa del período
    if ok_off and np.isfinite(tau_off):
        tau = float(tau_off)
    elif ok_on and np.isfinite(tau_on):
        tau = float(tau_on)
    else:
        tau = np.nan

    # h_static en la convención original de nivel
    h_static_nivel = -h_static_fit if depth_like and np.isfinite(h_static_fit) else h_static_fit

    # k = C / (h_static - h_d)
    c_on_ls = float(np.nanmedian(on_seg["caudal_ls"].to_numpy(float)))
    c_on_m3s = c_on_ls / 1000.0
    delta_h = h_static_fit - h_dyn_fit
    if np.isfinite(delta_h) and delta_h > 0 and np.isfinite(c_on_m3s) and c_on_m3s > 0:
        k = float(c_on_m3s / delta_h)
    else:
        k = np.nan

    A = float(tau * k) if np.isfinite(tau) and np.isfinite(k) and k > 0 else np.nan
    return float(h_static_nivel), float(tau), float(k), float(A)


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

    summary_df = pd.DataFrame([summary])
    summary_num_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[summary_num_cols] = summary_df[summary_num_cols].round(3)
    return summary_df


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

    df["dt_s"] = df["ts"].diff().dt.total_seconds()
    default_dt = np.nanmedian(df["dt_s"].iloc[1:].to_numpy()) if len(df) > 1 else 0.0
    df.loc[0, "dt_s"] = default_dt

    df["nivel_f"] = robust_median_smooth(df["nivel_m"], window=args.smooth_window)
    pump_on, umbral_usado_ls, uso_umbral_adaptativo = detect_pump_state(df["caudal_ls"], args.thr_ls)
    df["pump_on"] = pump_on

    depth_like = detect_depth_like(df["nivel_f"].to_numpy(float), df["pump_on"].to_numpy(bool))
    df["h"] = -df["nivel_f"] if depth_like else df["nivel_f"]

    start_ts = df["ts"].min()
    period_s = args.days_per_period * 86400.0
    elapsed_s = (df["ts"] - start_ts).dt.total_seconds()
    df["periodo"] = (elapsed_s // period_s).astype(int) + 1

    rows: List[PeriodRow] = []
    for periodo, part in df.groupby("periodo", sort=True):
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

        h_static_nivel, tau_s, k_m2_s, A_m2 = estimate_model_params_for_period(part, depth_like)

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
                h_static_nivel_m=float(h_static_nivel),
                tau_s=float(tau_s),
                k_m2_s=float(k_m2_s),
                A_m2=float(A_m2),
            )
        )

    periods_df = pd.DataFrame([asdict(r) for r in rows])
    periods_num_cols = periods_df.select_dtypes(include=[np.number]).columns
    periods_df[periods_num_cols] = periods_df[periods_num_cols].round(3)
    periods_df.to_csv(args.out_periods_csv, index=False, encoding="utf-8")

    if args.out_summary_csv:
        summary_df = summarize_periods(periods_df, days_per_period=args.days_per_period, depth_like=depth_like)
        summary_df["thr_on_usado_ls"] = round(float(umbral_usado_ls), 3)
        summary_df["uso_umbral_adaptativo"] = bool(uso_umbral_adaptativo)
        summary_df.to_csv(args.out_summary_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
