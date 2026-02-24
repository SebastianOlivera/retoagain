from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= 0:
        return np.nan
    return float(1 - sse / sst)


def _extract_on_segments(df: pd.DataFrame, on_col: str = "pump_on") -> list[tuple[int, int]]:
    if df.empty:
        return []
    on = df[on_col].astype(bool).to_numpy()
    segs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(on) - 1))
    return segs


def fit_tau_hd_by_segment(
    seg_df: pd.DataFrame,
    ts_col: str = "ts",
    level_col: str = "nivel_m",
    min_points: int = 10,
    min_tspan_s: float = 60.0,
) -> dict[str, Any] | None:
    if len(seg_df) < min_points:
        return None

    dfx = seg_df.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce")
    dfx[level_col] = pd.to_numeric(dfx[level_col], errors="coerce")
    dfx = dfx.dropna(subset=[ts_col, level_col]).sort_values(ts_col).reset_index(drop=True)

    if len(dfx) < min_points:
        return None

    t = (dfx[ts_col] - dfx[ts_col].iloc[0]).dt.total_seconds().to_numpy(float)
    h = dfx[level_col].to_numpy(float)

    t_span = float(t[-1] - t[0]) if len(t) else 0.0
    if t_span < min_tspan_s:
        return None

    h0 = float(h[0])

    def model(tvec: np.ndarray, h_d: float, tau: float) -> np.ndarray:
        tau_eff = max(float(tau), 1e-6)
        return h_d + (h0 - h_d) * np.exp(-np.maximum(tvec, 0.0) / tau_eff)

    h_d0 = float(np.nanmedian(h[-max(3, len(h) // 5) :]))
    tau0 = max(t_span / 3.0, 1.0)

    try:
        (h_d_fit, tau_fit), _ = curve_fit(
            model,
            t,
            h,
            p0=[h_d0, tau0],
            bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
            maxfev=20000,
        )
        tau_fit = max(float(tau_fit), 1e-6)
        pred = model(t, float(h_d_fit), tau_fit)
        rmse = float(np.sqrt(np.mean((h - pred) ** 2)))
        r2 = _safe_r2(h, pred)

        return {
            "tau": tau_fit,
            "h_d": float(h_d_fit),
            "rmse": rmse,
            "r2": r2,
            "n_points": int(len(dfx)),
            "t_span_s": t_span,
        }
    except Exception:  # noqa: BLE001
        return None


def fit_period_cycles(
    df_period: pd.DataFrame,
    thr_ls: float,
    ts_col: str = "ts",
    flow_col: str = "caudal_ls",
    level_col: str = "nivel_m",
) -> dict[str, Any]:
    dfx = df_period.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce")
    dfx[flow_col] = pd.to_numeric(dfx[flow_col], errors="coerce")
    dfx[level_col] = pd.to_numeric(dfx[level_col], errors="coerce")
    dfx = dfx.dropna(subset=[ts_col, flow_col, level_col]).sort_values(ts_col).reset_index(drop=True)

    if dfx.empty:
        return {
            "tau_s_mean": np.nan,
            "tau_s_std": np.nan,
            "tau_s_n": 0,
            "rmse_global": np.nan,
            "r2_global": np.nan,
            "ok_fit_global": False,
            "tau_values": np.array([], dtype=float),
            "hd_values": np.array([], dtype=float),
            "fit_count": 0,
        }

    dfx["pump_on"] = dfx[flow_col] > float(thr_ls)
    on_segments = _extract_on_segments(dfx, on_col="pump_on")

    tau_vals = []
    rmse_vals = []
    r2_vals = []
    hd_vals = []

    for s, e in on_segments:
        seg = dfx.loc[s:e, [ts_col, level_col]]
        fit = fit_tau_hd_by_segment(seg, ts_col=ts_col, level_col=level_col)
        if fit is None:
            continue
        tau_vals.append(float(fit["tau"]))
        rmse_vals.append(float(fit["rmse"]))
        r2_vals.append(float(fit["r2"]) if np.isfinite(fit["r2"]) else np.nan)
        hd_vals.append(float(fit["h_d"]))

    tau_arr = np.asarray(tau_vals, dtype=float)
    rmse_arr = np.asarray(rmse_vals, dtype=float)
    r2_arr = np.asarray(r2_vals, dtype=float)

    tau_valid = tau_arr[np.isfinite(tau_arr)]
    tau_n = int(len(tau_valid))
    if tau_n == 0:
        tau_mean = np.nan
        tau_std = np.nan
    else:
        tau_mean = float(np.mean(tau_valid))
        tau_std = float(np.std(tau_valid, ddof=1)) if tau_n >= 2 else 0.0

    rmse_global = float(np.nanmean(rmse_arr)) if len(rmse_arr) else np.nan
    r2_global = float(np.nanmean(r2_arr)) if len(r2_arr) else np.nan
    ok_fit_global = bool(tau_n >= 1 and np.isfinite(r2_global) and r2_global >= 0.7)

    return {
        "tau_s_mean": tau_mean,
        "tau_s_std": tau_std,
        "tau_s_n": tau_n,
        "rmse_global": rmse_global,
        "r2_global": r2_global,
        "ok_fit_global": ok_fit_global,
        "tau_values": tau_valid,
        "hd_values": np.asarray(hd_vals, dtype=float),
        "fit_count": tau_n,
    }

