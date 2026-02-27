from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= 1e-12:
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


def _off_segment_before(df: pd.DataFrame, on_start: int, on_col: str = "pump_on") -> tuple[int, int] | None:
    if on_start <= 0:
        return None
    i = on_start - 1
    if bool(df.iloc[i][on_col]):
        return None
    end = i
    while i >= 0 and (not bool(df.iloc[i][on_col])):
        i -= 1
    start = i + 1
    return start, end


def _estimate_h_static_for_cycle(
    df: pd.DataFrame,
    on_start: int,
    ts_col: str,
    level_col: str,
    on_col: str = "pump_on",
    off_window_s: float = 30 * 60,
    min_points: int = 3,
) -> float:
    off_seg = _off_segment_before(df, on_start, on_col=on_col)
    if off_seg is None:
        return np.nan

    s_off, e_off = off_seg
    off_df = df.loc[s_off:e_off, [ts_col, level_col]].copy()
    if off_df.empty:
        return np.nan

    t_end = pd.to_datetime(off_df[ts_col].iloc[-1], errors="coerce")
    if pd.isna(t_end):
        return np.nan

    ts = pd.to_datetime(off_df[ts_col], errors="coerce")
    vals = pd.to_numeric(off_df[level_col], errors="coerce")
    mask = (t_end - ts).dt.total_seconds() <= float(off_window_s)
    vals = vals[mask]
    vals = vals[np.isfinite(vals)]

    if len(vals) < min_points:
        return np.nan
    return float(np.median(vals.to_numpy(float)))


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
        t_eff = np.maximum(tvec, 0.0)
        return h_d - (h_d - h0) * np.exp(-t_eff / tau_eff)

    h_d0 = float(np.nanmedian(h[-max(3, len(h) // 5):]))
    tau0 = max(t_span / 3.0, 1.0)
    h_min = float(np.nanmin(h))
    h_max = float(np.nanmax(h))
    h_margin = max(0.1, 0.5 * (h_max - h_min + 1e-6))

    try:
        (h_d_fit, tau_fit), _ = curve_fit(
            model,
            t,
            h,
            p0=[h_d0, tau0],
            bounds=([h_min - h_margin, 1e-6], [h_max + h_margin, np.inf]),
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
    eps_denom: float = 1e-6,
) -> dict[str, Any]:
    dfx = df_period.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce")
    dfx[flow_col] = pd.to_numeric(dfx[flow_col], errors="coerce")
    dfx[level_col] = pd.to_numeric(dfx[level_col], errors="coerce")
    dfx = dfx.dropna(subset=[ts_col, flow_col, level_col]).sort_values(ts_col).reset_index(drop=True)

    empty = {
        "tau_s_mean": np.nan,
        "tau_s_std": np.nan,
        "tau_s_n": 0,
        "rmse_global": np.nan,
        "r2_global": np.nan,
        "ok_fit_global": False,
        "tau_values": np.array([], dtype=float),
        "hd_values": np.array([], dtype=float),
        "k_values": np.array([], dtype=float),
        "tiempo_entre_encendidos_values": np.array([], dtype=float),
        "fit_count": 0,
        "cycles": [],
    }
    if dfx.empty:
        return empty

    if "estado_bomba" in dfx.columns:
        dfx["pump_on"] = pd.to_numeric(dfx["estado_bomba"], errors="coerce").fillna(0).astype(int).astype(bool)
    else:
        dfx["pump_on"] = dfx[flow_col] > float(thr_ls)

    on_segments = _extract_on_segments(dfx, on_col="pump_on")

    tau_vals: list[float] = []
    rmse_vals: list[float] = []
    r2_vals: list[float] = []
    hd_vals: list[float] = []
    k_vals: list[float] = []
    t_between_vals: list[float] = []
    cycles: list[dict[str, Any]] = []

    for idx, (s, e) in enumerate(on_segments):
        seg = dfx.loc[s:e, [ts_col, level_col, flow_col]].copy()
        fit = fit_tau_hd_by_segment(seg, ts_col=ts_col, level_col=level_col)

        t_start = dfx.loc[s, ts_col]
        t_end = dfx.loc[e, ts_col]
        c_seg = float(np.nanmean(pd.to_numeric(seg[flow_col], errors="coerce").to_numpy(float)))

        h_static = _estimate_h_static_for_cycle(dfx, s, ts_col=ts_col, level_col=level_col)
        hd_fit = float(fit["h_d"]) if fit is not None else np.nan
        tau_fit = float(fit["tau"]) if fit is not None else np.nan
        rmse = float(fit["rmse"]) if fit is not None else np.nan
        r2 = float(fit["r2"]) if (fit is not None and np.isfinite(fit["r2"])) else np.nan
        ok_fit = bool(fit is not None and np.isfinite(r2) and r2 >= 0.7)

        den = hd_fit - h_static if np.isfinite(hd_fit) and np.isfinite(h_static) else np.nan
        ok_k = False
        k_val = np.nan
        if np.isfinite(den) and abs(den) >= eps_denom and np.isfinite(c_seg) and c_seg > 0 and ok_fit:
            k_val = float(c_seg / den)
            ok_k = bool(np.isfinite(k_val) and k_val > 0)

        if fit is not None:
            tau_vals.append(tau_fit)
            rmse_vals.append(rmse)
            r2_vals.append(r2)
            hd_vals.append(hd_fit)
        if np.isfinite(k_val):
            k_vals.append(k_val)

        t_between = np.nan
        between_error = False
        if idx + 1 < len(on_segments):
            next_s, _ = on_segments[idx + 1]
            delta = float((dfx.loc[next_s, ts_col] - t_end).total_seconds())
            if delta < 0:
                between_error = True
                t_between = np.nan
            else:
                t_between = delta
                t_between_vals.append(delta)

        cycles.append({
            "cycle_idx": int(idx + 1),
            "inicio_on": t_start,
            "fin_on": t_end,
            "h_static": h_static,
            "hd_fit": hd_fit,
            "tau_fit": tau_fit,
            "C_const_ls": c_seg,
            "k": k_val,
            "ok_k": ok_k,
            "rmse": rmse,
            "r2": r2,
            "ok_fit": ok_fit,
            "tiempo_entre_encendidos_s": t_between,
            "tiempo_entre_error": between_error,
        })

    tau_arr = np.asarray(tau_vals, dtype=float)
    rmse_arr = np.asarray(rmse_vals, dtype=float)
    r2_arr = np.asarray(r2_vals, dtype=float)

    tau_valid = tau_arr[np.isfinite(tau_arr)]
    tau_n = int(len(tau_valid))
    tau_mean = float(np.mean(tau_valid)) if tau_n else np.nan
    tau_std = float(np.std(tau_valid, ddof=1)) if tau_n >= 2 else (0.0 if tau_n == 1 else np.nan)

    rmse_global = float(np.nanmean(rmse_arr)) if len(rmse_arr) else np.nan
    r2_valid = r2_arr[np.isfinite(r2_arr)]
    r2_global = float(np.mean(r2_valid)) if len(r2_valid) else np.nan
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
        "k_values": np.asarray(k_vals, dtype=float),
        "tiempo_entre_encendidos_values": np.asarray(t_between_vals, dtype=float),
        "fit_count": tau_n,
        "cycles": cycles,
    }
