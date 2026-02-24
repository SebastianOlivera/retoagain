from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def robust_median_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).median()


def detect_depth_like(nivel: np.ndarray, pump_on: np.ndarray) -> bool:
    if pump_on.sum() < 10 or (~pump_on).sum() < 10:
        return False
    return bool(np.nanmedian(nivel[pump_on]) > np.nanmedian(nivel[~pump_on]))


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


def summarize_periods(periods_df: pd.DataFrame, days_per_period: float, depth_like: bool) -> pd.DataFrame:
    summary: Dict[str, float | int | bool] = {
        "n_periodos": int(len(periods_df)),
        "dias_por_periodo": float(days_per_period),
        "nivel_es_profundidad_detectado": bool(depth_like),
    }
    for col in periods_df.select_dtypes(include=[np.number]).columns:
        if col == "periodo":
            continue
        m, s = mean_std(periods_df[col])
        summary[f"{col}_prom"] = m
        summary[f"{col}_std"] = s
    return pd.DataFrame([summary])


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
    dfx = dfx.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    dfx["dt_s"] = dfx["ts"].diff().dt.total_seconds()
    default_dt = float(np.nanmedian(dfx["dt_s"].iloc[1:].to_numpy())) if len(dfx) > 1 else 0.0
    dfx.loc[0, "dt_s"] = default_dt

    dfx["nivel_f"] = robust_median_smooth(dfx["nivel_m"], window=smooth_window)
    dfx["pump_on"] = dfx["caudal_ls"] > thr_ls
    depth_like = detect_depth_like(dfx["nivel_f"].to_numpy(float), dfx["pump_on"].to_numpy(bool))
    dfx["h"] = -dfx["nivel_f"] if depth_like else dfx["nivel_f"]

    start_ts = dfx["ts"].min()
    period_s = days_per_period * 86400.0
    dfx["periodo"] = (((dfx["ts"] - start_ts).dt.total_seconds()) // period_s).astype(int) + 1

    rows: List[PeriodRow] = []
    for periodo, part in dfx.groupby("periodo", sort=True):
        on = part[part["pump_on"]]
        off = part[~part["pump_on"]]
        cm, cs = mean_std(part["caudal_ls"])
        nm, ns = mean_std(part["nivel_m"])
        nfm, nfs = mean_std(part["nivel_f"])
        hm, hs = mean_std(part["h"])
        con_m, con_s = mean_std(on["caudal_ls"]) if len(on) else (np.nan, np.nan)
        cof_m, cof_s = mean_std(off["caudal_ls"]) if len(off) else (np.nan, np.nan)

        rows.append(
            PeriodRow(
                periodo=int(periodo),
                inicio=str(part["ts"].iloc[0]),
                fin=str(part["ts"].iloc[-1]),
                n_muestras=int(len(part)),
                duracion_s=float(np.nansum(part["dt_s"].to_numpy())),
                porcentaje_on=float(part["pump_on"].mean() * 100.0),
                volumen_bombeado_m3=integrate_volume(part["caudal_ls"].to_numpy(float) / 1000.0, part["dt_s"].to_numpy(float)),
                caudal_prom_ls=cm,
                caudal_std_ls=cs,
                caudal_on_prom_ls=con_m,
                caudal_on_std_ls=con_s,
                caudal_off_prom_ls=cof_m,
                caudal_off_std_ls=cof_s,
                nivel_prom_m=nm,
                nivel_std_m=ns,
                nivel_f_prom_m=nfm,
                nivel_f_std_m=nfs,
                h_prom_m=hm,
                h_std_m=hs,
            )
        )

    periods_df = pd.DataFrame([asdict(r) for r in rows])
    summary_df = summarize_periods(periods_df, days_per_period=days_per_period, depth_like=depth_like)
    return periods_df, summary_df
