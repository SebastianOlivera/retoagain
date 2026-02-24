from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fit_hd_k_A_from_cycle(
    df: pd.DataFrame,
    cycle: int = 1,
    ts_col: str = "ts",
    flow_col: str = "caudal_ls",
    level_col: str = "nivel_m",
    on_threshold: float = 0.0,
) -> dict:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas DataFrame")

    for col in (ts_col, flow_col, level_col):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en el DataFrame")

    dfx = df.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce")
    dfx = dfx.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    is_on = dfx[flow_col] > on_threshold
    start_cycle = is_on & (~is_on.shift(1, fill_value=False))
    cycle_id = start_cycle.cumsum()

    dfx["_is_on"] = is_on
    dfx["_cycle_id"] = cycle_id

    on_df = dfx[dfx["_is_on"]].copy()
    if on_df.empty:
        raise ValueError("No hay registros con bomba encendida.")

    available = sorted(on_df["_cycle_id"].unique())
    if cycle not in available:
        raise ValueError(f"Ciclo {cycle} no existe. Ciclos disponibles: {available}")

    seg = on_df[on_df["_cycle_id"] == cycle].copy().reset_index(drop=True)
    t = (seg[ts_col] - seg[ts_col].iloc[0]).dt.total_seconds().to_numpy()
    h = seg[level_col].to_numpy(dtype=float)

    if len(h) < 3:
        raise ValueError("Muy pocos puntos en el ciclo para ajustar.")

    h0 = float(h[0])
    C = float(seg[flow_col].mean())

    def model(tvec: np.ndarray, h_s: float, k: float, tau: float) -> np.ndarray:
        h_d = h_s + C / k
        return h_d - (h_d - h0) * np.exp(-tvec / tau)

    h_s0 = float(np.median(h))
    k0 = max(abs(C) / max(abs(h_s0 - h[-1]), 1e-6), 1e-3)
    tau0 = max((t[-1] - t[0]) / 5, 1e-6)

    eps = 1e-9
    (h_s_fit, k_fit, tau_fit), _ = curve_fit(
        model,
        t,
        h,
        p0=[h_s0, k0, tau0],
        bounds=([-np.inf, eps, eps], [np.inf, np.inf, np.inf]),
        maxfev=50000,
    )

    hd_fit = float(h_s_fit + C / k_fit)
    A_fit = float(k_fit * tau_fit)

    resid = h - model(t, h_s_fit, k_fit, tau_fit)
    rmse = float(np.sqrt(np.mean(resid**2)))

    return {
        "cycle": cycle,
        "period_start": seg[ts_col].iloc[0],
        "period_end": seg[ts_col].iloc[-1],
        "C_mean_ls": C,
        "h_s": float(h_s_fit),
        "h_d": hd_fit,
        "k": float(k_fit),
        "A": A_fit,
        "tau": float(tau_fit),
        "rmse": rmse,
    }
