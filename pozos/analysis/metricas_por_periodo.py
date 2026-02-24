from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pozos.analysis.ajuste_fisico import fit_period_cycles


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


def compute_period_metrics(
    df: pd.DataFrame,
    thr_ls: float = 0.05,
    smooth_window: int = 5,
    days_per_period: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if days_per_period <= 0:
        raise ValueError("days_per_period debe ser > 0")

    required = {"ts", "caudal_ls", "nivel_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas. Requiere {required}")

    dfx = df.copy()
    dfx["ts"] = pd.to_datetime(dfx["ts"], errors="coerce")
    dfx["caudal_ls"] = pd.to_numeric(dfx["caudal_ls"], errors="coerce")
    dfx["nivel_m"] = pd.to_numeric(dfx["nivel_m"], errors="coerce")
    dfx = dfx.dropna(subset=["ts", "caudal_ls", "nivel_m"]).sort_values("ts").reset_index(drop=True)

    if dfx.empty:
        return pd.DataFrame(), None

    dfx["nivel_use"] = robust_median_smooth(dfx["nivel_m"], window=max(1, int(smooth_window)))
    dfx["pump_on"] = dfx["caudal_ls"] > float(thr_ls)

    start_ts = dfx["ts"].min()
    period_s = float(days_per_period) * 86400.0
    dfx["periodo"] = (((dfx["ts"] - start_ts).dt.total_seconds()) // period_s).astype(int) + 1

    rows: list[dict[str, Any]] = []

    for periodo, part in dfx.groupby("periodo", sort=True):
        part = part.copy().reset_index(drop=True)
        on_mask = part["pump_on"].astype(bool)

        # Fitting físico por ciclos ON del período
        fit = fit_period_cycles(part, thr_ls=thr_ls, ts_col="ts", flow_col="caudal_ls", level_col="nivel_use")

        hs_values = part.loc[~on_mask, "nivel_use"].to_numpy(float)  # SOLO OFF
        hd_values = part.loc[on_mask, "nivel_use"].to_numpy(float)   # SOLO ON
        c_values = part.loc[on_mask, "caudal_ls"].to_numpy(float)    # SOLO ON

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
        }

        _add_stats_cols(row, "h_static_nivel", hs_values)
        _add_stats_cols(row, "h_dinamico_nivel", hd_values)
        _add_stats_cols(row, "C_const_ls", c_values)

        tau_values = np.asarray(fit.get("tau_values", []), dtype=float)
        tau_values = tau_values[np.isfinite(tau_values)]
        row["tau_s_median"] = float(np.median(tau_values)) if len(tau_values) else np.nan
        row["tau_s_mean"] = float(fit["tau_s_mean"]) if np.isfinite(fit["tau_s_mean"]) else np.nan
        row["tau_s_std"] = float(fit["tau_s_std"]) if np.isfinite(fit["tau_s_std"]) else np.nan
        row["tau_s_n"] = int(fit["tau_s_n"])

        row["umbral_q_usado_ls"] = float(thr_ls)
        rows.append(row)

    periods_df = pd.DataFrame(rows)

    # Normalización defensiva por compatibilidad con nombres legacy
    periods_df = periods_df.rename(columns={
        "h_static_nivel_m": "h_static_nivel_median",
        "h_dinamico_nivel_m": "h_dinamico_nivel_median",
        "tau_s": "tau_s_median",
        "C_const_ls": "C_const_ls_median",
    })
    # Eliminar cualquier columna agregada ambigua legacy
    periods_df = periods_df.drop(
        columns=["h_static_nivel_m", "h_dinamico_nivel_m", "tau_s", "C_const_ls"],
        errors="ignore",
    )

    desired_order = [
        "device_id", "periodo", "inicio", "fin",
        "n_on", "frecuencia_encendido_por_dia", "tiempo_on_prom_s",
        "ok_fit_global", "rmse_global", "r2_global",
        "h_static_nivel_median", "h_static_nivel_mean", "h_static_nivel_std", "h_static_nivel_n",
        "h_dinamico_nivel_median", "h_dinamico_nivel_mean", "h_dinamico_nivel_std", "h_dinamico_nivel_n",
        "tau_s_median", "tau_s_mean", "tau_s_std", "tau_s_n",
        "C_const_ls_median", "C_const_ls_mean", "C_const_ls_std", "C_const_ls_n",
        "umbral_q_usado_ls",
    ]

    cols = [c for c in desired_order if c in periods_df.columns] + [c for c in periods_df.columns if c not in desired_order]
    periods_df = periods_df[cols]

    # Tipos numéricos
    for col in periods_df.columns:
        if col.endswith("_median") or col.endswith("_mean") or col.endswith("_std") or col.endswith("_n") or col in {
            "frecuencia_encendido_por_dia", "tiempo_on_prom_s", "rmse_global", "r2_global", "umbral_q_usado_ls"
        }:
            periods_df[col] = pd.to_numeric(periods_df[col], errors="coerce")

    for col in periods_df.columns:
        if col.endswith("_n") or col == "n_on":
            periods_df[col] = periods_df[col].fillna(0).astype("Int64")

    return periods_df, None
