from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pozos.analysis.ajuste_fisico import fit_period_cycles

LITERS_TO_M3 = 1.0 / 1000.0


def robust_median_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).median()


def _compute_stats(values: Any) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n == 0:
        return {"median": np.nan, "mean": np.nan, "std": np.nan, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    median = float(np.median(arr))
    return {"median": median, "mean": mean, "std": std, "n": n}


def _add_stats_cols(row: dict[str, Any], base_name: str, values: Any) -> None:
    st = _compute_stats(values)
    row[f"{base_name}_median"] = st["median"]
    row[f"{base_name}_mean"] = st["mean"]
    row[f"{base_name}_std"] = st["std"]
    row[f"{base_name}_n"] = int(st["n"])


def _count_on_segments(pump_on: pd.Series) -> int:
    p = pump_on.astype(bool)
    prev = p.shift(1)
    starts = p & (prev == False)
    return int(starts.fillna(False).sum())


def _mean_on_duration_seconds(part: pd.DataFrame, ts_col: str = "ts") -> float:
    on = part["pump_on"].astype(bool).to_numpy()
    if len(on) == 0:
        return np.nan

    starts = []
    ends = []
    s = None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif (not v) and s is not None:
            starts.append(s)
            ends.append(i - 1)
            s = None
    if s is not None:
        starts.append(s)
        ends.append(len(on) - 1)

    if not starts:
        return np.nan

    durs = []
    for s_idx, e_idx in zip(starts, ends):
        t0 = part.iloc[s_idx][ts_col]
        t1 = part.iloc[e_idx][ts_col]
        durs.append(float((t1 - t0).total_seconds()))
    return float(np.mean(durs)) if durs else np.nan


def _prepare_df(df: pd.DataFrame, smooth_window: int, thr_m3s: float) -> pd.DataFrame:
    dfx = df.copy()
    dfx["ts"] = pd.to_datetime(dfx["ts"], errors="coerce")
    dfx["nivel_m"] = pd.to_numeric(dfx["nivel_m"], errors="coerce")

    if "caudal_m3s" in dfx.columns:
        dfx["caudal_m3s"] = pd.to_numeric(dfx["caudal_m3s"], errors="coerce")
    elif "caudal_ls" in dfx.columns:
        dfx["caudal_ls"] = pd.to_numeric(dfx["caudal_ls"], errors="coerce")
        dfx["caudal_m3s"] = dfx["caudal_ls"] * LITERS_TO_M3
    else:
        raise ValueError("Falta columna de caudal: se requiere caudal_ls o caudal_m3s")

    dfx = dfx.dropna(subset=["ts", "caudal_m3s", "nivel_m"]).sort_values("ts").reset_index(drop=True)
    if dfx.empty:
        return dfx

    dfx["nivel_use"] = robust_median_smooth(dfx["nivel_m"], window=max(1, int(smooth_window)))
    if "estado_bomba" in dfx.columns:
        dfx["pump_on"] = pd.to_numeric(dfx["estado_bomba"], errors="coerce").fillna(0).astype(int).astype(bool)
    else:
        dfx["pump_on"] = dfx["caudal_m3s"] > float(thr_m3s)
    return dfx


def compute_cycle_metrics(
    df: pd.DataFrame,
    thr_m3s: float = 0.00005,
    smooth_window: int = 5,
) -> pd.DataFrame:
    required = {"ts", "nivel_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas. Requiere {required} + caudal_ls/caudal_m3s")

    dfx = _prepare_df(df, smooth_window=smooth_window, thr_m3s=thr_m3s)
    if dfx.empty:
        return pd.DataFrame()

    fit = fit_period_cycles(dfx, thr_m3s=thr_m3s, ts_col="ts", flow_col="caudal_m3s", level_col="nivel_use")
    rows: list[dict[str, Any]] = []
    for cyc in fit.get("cycles", []):
        rows.append(
            {
                "device_id": str(dfx["device_id"].iloc[0]) if "device_id" in dfx.columns else "",
                "cycle_idx": cyc["cycle_idx"],
                "inicio_on": cyc["inicio_on"],
                "fin_on": cyc["fin_on"],
                "h_static": cyc["h_static"],
                "hd_fit": cyc["hd_fit"],
                "tau_fit": cyc["tau_fit"],
                "C_const_m3s": cyc["C_const_m3s"],
                "k": cyc["k"],
                "ok_k": cyc["ok_k"],
                "rmse": cyc["rmse"],
                "r2": cyc["r2"],
                "ok_fit": cyc["ok_fit"],
                "tiempo_entre_encendidos_s": cyc["tiempo_entre_encendidos_s"],
                "tiempo_entre_error": cyc["tiempo_entre_error"],
            }
        )
    return pd.DataFrame(rows)


def compute_period_metrics(
    df: pd.DataFrame,
    thr_m3s: float = 0.00005,
    smooth_window: int = 5,
    days_per_period: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if days_per_period <= 0:
        raise ValueError("days_per_period debe ser > 0")

    required = {"ts", "nivel_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas. Requiere {required} + caudal_ls/caudal_m3s")

    dfx = _prepare_df(df, smooth_window=smooth_window, thr_m3s=thr_m3s)
    if dfx.empty:
        return pd.DataFrame(), None

    start_ts = dfx["ts"].min()
    period_s = float(days_per_period) * 86400.0
    dfx["periodo"] = (((dfx["ts"] - start_ts).dt.total_seconds()) // period_s).astype(int) + 1

    rows: list[dict[str, Any]] = []

    for periodo, part in dfx.groupby("periodo", sort=True):
        part = part.copy().reset_index(drop=True)
        on_mask = part["pump_on"].astype(bool)

        fit = fit_period_cycles(part, thr_m3s=thr_m3s, ts_col="ts", flow_col="caudal_m3s", level_col="nivel_use")

        hs_values = part.loc[~on_mask, "nivel_use"].to_numpy(float)
        hd_values = part.loc[on_mask, "nivel_use"].to_numpy(float)
        c_values = part.loc[on_mask, "caudal_m3s"].to_numpy(float)
        tau_values = np.asarray(fit.get("tau_values", []), dtype=float)
        k_values = np.asarray(fit.get("k_values", []), dtype=float)
        t_between_values = np.asarray(fit.get("tiempo_entre_encendidos_values", []), dtype=float)

        inicio = part["ts"].iloc[0]
        fin = part["ts"].iloc[-1]
        dur_days = max((fin - inicio).total_seconds() / 86400.0, days_per_period)
        n_on = _count_on_segments(part["pump_on"])
        freq_day = float(n_on / dur_days) if dur_days > 0 else 0.0
        t_on_prom_s = _mean_on_duration_seconds(part, ts_col="ts")

        row: dict[str, Any] = {
            "device_id": str(part["device_id"].iloc[0]) if "device_id" in part.columns else "",
            "periodo": int(periodo),
            "inicio": inicio,
            "fin": fin,
            "n_on": n_on,
            "frecuencia_encendido_por_dia": freq_day,
            "tiempo_on_prom_s": t_on_prom_s,
            "ok_fit_global": bool(fit["ok_fit_global"]),
            "rmse_global": float(fit["rmse_global"]) if np.isfinite(fit["rmse_global"]) else np.nan,
            "r2_global": float(fit["r2_global"]) if np.isfinite(fit["r2_global"]) else np.nan,
            "umbral_q_usado_m3s": float(thr_m3s),
        }

        _add_stats_cols(row, "h_static_nivel", hs_values)
        _add_stats_cols(row, "h_dinamico_nivel", hd_values)
        _add_stats_cols(row, "tau_s", tau_values)
        _add_stats_cols(row, "C_const_m3s", c_values)
        _add_stats_cols(row, "k", k_values)
        _add_stats_cols(row, "tiempo_entre_encendidos", t_between_values)
        rows.append(row)

    periods_df = pd.DataFrame(rows)

    desired_order = [
        "device_id",
        "periodo",
        "inicio",
        "fin",
        "n_on",
        "frecuencia_encendido_por_dia",
        "tiempo_on_prom_s",
        "ok_fit_global",
        "rmse_global",
        "r2_global",
        "h_static_nivel_median",
        "h_static_nivel_mean",
        "h_static_nivel_std",
        "h_static_nivel_n",
        "h_dinamico_nivel_median",
        "h_dinamico_nivel_mean",
        "h_dinamico_nivel_std",
        "h_dinamico_nivel_n",
        "tau_s_median",
        "tau_s_mean",
        "tau_s_std",
        "tau_s_n",
        "C_const_m3s_median",
        "C_const_m3s_mean",
        "C_const_m3s_std",
        "C_const_m3s_n",
        "k_median",
        "k_mean",
        "k_std",
        "k_n",
        "tiempo_entre_encendidos_median",
        "tiempo_entre_encendidos_mean",
        "tiempo_entre_encendidos_std",
        "tiempo_entre_encendidos_n",
        "umbral_q_usado_m3s",
    ]

    cols = [c for c in desired_order if c in periods_df.columns] + [c for c in periods_df.columns if c not in desired_order]
    periods_df = periods_df[cols]

    for col in periods_df.columns:
        if col.endswith("_median") or col.endswith("_mean") or col.endswith("_std") or col.endswith("_n") or col in {
            "frecuencia_encendido_por_dia",
            "tiempo_on_prom_s",
            "rmse_global",
            "r2_global",
            "umbral_q_usado_m3s",
        }:
            periods_df[col] = pd.to_numeric(periods_df[col], errors="coerce")

    for col in periods_df.columns:
        if col.endswith("_n") or col == "n_on":
            periods_df[col] = periods_df[col].fillna(0).astype("Int64")

    return periods_df, None
