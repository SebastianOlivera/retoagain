from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class FitResult:
    y_inf: float
    tau_s: float
    rmse: float
    r2: float
    ok: bool


class WellProfiler:
    """Analiza ciclos de bombeo y calcula parámetros operativos e hidrogeológicos."""

    def __init__(self, min_cycle_points: int = 5, smooth_window: int = 5) -> None:
        self.min_cycle_points = min_cycle_points
        self.smooth_window = smooth_window
        self.df: Optional[pd.DataFrame] = None
        self.cycles_df: Optional[pd.DataFrame] = None
        self.stats: Dict[str, float | int | str | bool | None] = {}
        self.regime = "Desconocido"
        self.threshold = 0.5
        self.depth_like = False

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        threshold: float,
        min_cycle_points: int = 5,
        smooth_window: int = 5,
    ) -> "WellProfiler":
        instance = cls(min_cycle_points=min_cycle_points, smooth_window=smooth_window)
        instance.threshold = threshold
        instance.df = df.copy()
        instance.df["pump_on"] = instance.df["caudal_ls"] > threshold
        instance._prepare_level_column()
        return instance

    @staticmethod
    def _detect_depth_like(nivel: np.ndarray, pump_on: np.ndarray) -> bool:
        if pump_on.sum() < 10 or (~pump_on).sum() < 10:
            return False
        return bool(np.nanmedian(nivel[pump_on]) > np.nanmedian(nivel[~pump_on]))

    def _prepare_level_column(self) -> None:
        assert self.df is not None
        self.df["nivel_f"] = self.df["nivel_m"].rolling(
            window=self.smooth_window, center=True, min_periods=1
        ).median()
        self.depth_like = self._detect_depth_like(
            self.df["nivel_f"].to_numpy(float),
            self.df["pump_on"].to_numpy(bool),
        )
        self.df["h"] = -self.df["nivel_f"] if self.depth_like else self.df["nivel_f"]

    @staticmethod
    def _fit_exp(t: np.ndarray, y: np.ndarray) -> FitResult:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = np.isfinite(t) & np.isfinite(y)
        t, y = t[valid], y[valid]
        if len(t) < 3:
            return FitResult(np.nan, np.nan, np.nan, np.nan, False)

        y0 = float(y[0])
        y_inf0 = float(np.nanmedian(y[-max(3, len(y) // 5) :]))
        tau0 = max(float(t[-1] - t[0]) / 3.0, 1.0)
        x0 = np.array([y_inf0, np.log(tau0)])

        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        rng = abs(ymax - ymin) + 1e-6
        lb = np.array([ymin - 10 * rng, np.log(1e-3)])
        ub = np.array([ymax + 10 * rng, np.log(1e9)])

        def residuals(x: np.ndarray) -> np.ndarray:
            y_inf, logtau = x
            return y - (y_inf + (y0 - y_inf) * np.exp(-(t - t[0]) / np.exp(logtau)))

        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), loss="huber", max_nfev=10000)
            y_inf_fit = float(res.x[0])
            tau_fit = float(np.exp(res.x[1]))
            r = residuals(res.x)
            rmse = float(np.sqrt(np.mean(r**2)))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1 - np.sum(r**2) / ss_tot) if ss_tot > 0 else float("nan")
            return FitResult(y_inf_fit, tau_fit, rmse, r2, bool(res.success))
        except Exception:
            return FitResult(np.nan, np.nan, np.nan, np.nan, False)

    def extract_cycles(self) -> None:
        if self.df is None:
            raise ValueError("El DataFrame no está cargado")

        df = self.df.copy()
        df["change"] = df["pump_on"].astype(int).diff().fillna(0)
        starts = df[df["change"] == 1].index.tolist()
        if not df.empty and bool(df["pump_on"].iloc[0]):
            starts = [int(df.index[0])] + starts

        cycles_data = []

        for start_idx in starts:
            ends = df[(df.index > start_idx) & (df["change"] == -1)].index
            if len(ends) == 0:
                end_idx = int(df.index[-1]) + 1
            else:
                end_idx = int(ends[0])

            on_window = df.loc[start_idx : end_idx - 1]
            if len(on_window) < self.min_cycle_points:
                continue

            q = on_window["caudal_ls"]
            duration_min = (on_window["ts"].iloc[-1] - on_window["ts"].iloc[0]).total_seconds() / 60.0

            dt_arr = on_window["ts"].diff().dt.total_seconds().to_numpy()
            if len(dt_arr) > 1:
                dt_arr[0] = float(np.nanmedian(dt_arr[1:]))
            else:
                dt_arr[0] = 0.0
            vol_m3 = float(np.nansum((q.to_numpy() / 1000.0) * np.nan_to_num(dt_arr)))

            t_on = (on_window["ts"] - on_window["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
            fit_on = self._fit_exp(t_on, on_window["h"].to_numpy(float))
            h_d_nivel = -fit_on.y_inf if self.depth_like else fit_on.y_inf

            prev_offs = df[(df.index < start_idx) & (df["change"] == -1)].index
            off_start_idx = int(prev_offs[-1]) if len(prev_offs) > 0 else int(df.index[0])
            off_window = df.loc[off_start_idx : start_idx - 1]

            h_static_nivel = np.nan
            fit_off = FitResult(np.nan, np.nan, np.nan, np.nan, False)
            k = np.nan
            A = np.nan

            if len(off_window) >= self.min_cycle_points:
                t_off = (off_window["ts"] - off_window["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
                fit_off = self._fit_exp(t_off, off_window["h"].to_numpy(float))
                h_static_nivel = -fit_off.y_inf if self.depth_like else fit_off.y_inf

                q_m3s = float(q.median()) / 1000.0
                delta_h = fit_off.y_inf - fit_on.y_inf
                if np.isfinite(delta_h) and delta_h > 0 and q_m3s > 0:
                    k = float(q_m3s / delta_h)
                    tau_use = fit_off.tau_s if (fit_off.ok and np.isfinite(fit_off.tau_s)) else fit_on.tau_s
                    if np.isfinite(tau_use) and tau_use > 0:
                        A = float(tau_use * k)

            cycles_data.append(
                {
                    "start_ts": on_window["ts"].iloc[0],
                    "duration_min": duration_min,
                    "q_median": q.median(),
                    "q_std": q.std(),
                    "h_median": on_window["nivel_m"].median(),
                    "h_static_m": h_static_nivel,
                    "h_dinamico_m": h_d_nivel,
                    "tau_off_s": fit_off.tau_s,
                    "tau_on_s": fit_on.tau_s,
                    "volumen_bombeado_m3": vol_m3,
                    "k_m2_s": k,
                    "A_m2": A,
                    "rmse_off": fit_off.rmse,
                    "r2_off": fit_off.r2,
                    "rmse_on": fit_on.rmse,
                    "r2_on": fit_on.r2,
                    "ok_off": fit_off.ok,
                    "ok_on": fit_on.ok,
                }
            )

        self.cycles_df = pd.DataFrame(cycles_data)

    def compute_global_metrics(self) -> None:
        if self.df is None:
            raise ValueError("El DataFrame no está cargado")

        t_total_days = (self.df["ts"].max() - self.df["ts"].min()).total_seconds() / 86400
        t_total_days = max(t_total_days, 1e-6)

        if self.cycles_df is not None and not self.cycles_df.empty:
            total_cycles = len(self.cycles_df)
            freq_cycles_day = total_cycles / t_total_days
            avg_duration = self.cycles_df["duration_min"].mean()
            median_q = self.cycles_df["q_median"].median()
            variability_q = self.cycles_df["q_std"].mean()
            total_on_minutes = self.cycles_df["duration_min"].sum()
            duty_cycle_pct = (total_on_minutes / (t_total_days * 1440)) * 100

            vol_total_m3 = float(self.cycles_df["volumen_bombeado_m3"].sum())
            h_static_mean = float(np.nanmean(self.cycles_df["h_static_m"]))
            h_d_mean = float(np.nanmean(self.cycles_df["h_dinamico_m"]))
            k_mean = float(np.nanmean(self.cycles_df["k_m2_s"]))
            A_mean = float(np.nanmean(self.cycles_df["A_m2"]))
        else:
            if self.df["pump_on"].mean() > 0.95:
                duty_cycle_pct = 100.0
                freq_cycles_day = 0.0
                avg_duration = t_total_days * 1440
                median_q = self.df["caudal_ls"].median()
                variability_q = self.df["caudal_ls"].std()
            else:
                duty_cycle_pct = 0.0
                freq_cycles_day = 0.0
                avg_duration = 0.0
                median_q = 0.0
                variability_q = 0.0

            vol_total_m3 = 0.0
            h_static_mean = np.nan
            h_d_mean = np.nan
            k_mean = np.nan
            A_mean = np.nan

        def _safe_round(v: float, decimals: int) -> Optional[float]:
            return round(float(v), decimals) if np.isfinite(v) else None

        self.stats = {
            "duty_cycle_pct": round(float(duty_cycle_pct), 2),
            "freq_cycles_day": round(float(freq_cycles_day), 2),
            "avg_cycle_duration_min": round(float(avg_duration), 1),
            "typical_flow_ls": round(float(median_q), 2),
            "flow_stability_std": round(float(variability_q), 2),
            "vol_total_m3": round(float(vol_total_m3), 3),
            "h_static_mean_m": _safe_round(h_static_mean, 3),
            "h_dinamico_mean_m": _safe_round(h_d_mean, 3),
            "k_mean_m2_s": _safe_round(k_mean, 6),
            "A_mean_m2": _safe_round(A_mean, 3),
            "nivel_es_profundidad": self.depth_like,
            "dynamic_threshold_used": round(float(self.threshold), 3),
        }

    def classify_regime(self) -> None:
        s = self.stats
        if float(s["duty_cycle_pct"]) > 98.0:
            self.regime = "Siempre ON"
        elif float(s["duty_cycle_pct"]) < 1.0:
            self.regime = "Siempre OFF"
        elif float(s["freq_cycles_day"]) > 10:
            self.regime = "Mixto Pulso"
        elif float(s["avg_cycle_duration_min"]) > 360:
            self.regime = "Mixto Largo"
        else:
            self.regime = "Mixto Medio"

    def get_features(self) -> Dict[str, float | int | str | bool | None]:
        features = dict(self.stats)
        features["regime_label"] = self.regime
        return features
