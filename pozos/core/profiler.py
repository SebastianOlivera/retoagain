from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


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
        self.period_aggregations: Dict[str, pd.DataFrame] = {}
        self.stats: Dict[str, float | int | str | bool | None] = {}
        self.regime = "Desconocido"
        self.threshold = 0.5
        self.depth_like = False

        self.static_window_points = 30
        self.fit_min_r2 = 0.2
        self.eps_den = 1e-6

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
        instance.df["pump_on"] = instance.df["caudal_m3s"] > threshold
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

    def _select_static_level(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        off_prev = df[(df.index < start_idx) & (~df["pump_on"])].tail(self.static_window_points)
        if len(off_prev) >= self.min_cycle_points:
            return float(np.nanmedian(off_prev["h"].to_numpy(float)))

        off_next = df[(df.index >= end_idx) & (~df["pump_on"])].head(self.static_window_points)
        if len(off_next) >= self.min_cycle_points:
            return float(np.nanmedian(off_next["h"].to_numpy(float)))

        return np.nan

    def _fit_on_segment(self, t_on: np.ndarray, h_on: np.ndarray, h_static: float) -> FitResult:
        t = np.asarray(t_on, dtype=float)
        h = np.asarray(h_on, dtype=float)
        valid = np.isfinite(t) & np.isfinite(h)
        t, h = t[valid], h[valid]
        if len(t) < 3:
            return FitResult(np.nan, np.nan, np.nan, np.nan, False)

        h0 = float(h[0])
        h_min = float(np.nanmin(h))
        h_max = float(np.nanmax(h))
        h_rng = max(abs(h_max - h_min), 1e-6)

        hd0 = float(np.nanmedian(h[-max(3, len(h) // 4) :]))
        tau0 = max(float(t[-1] - t[0]) / 3.0, 1.0)

        if np.isfinite(h_static):
            hd_upper = h_static + 2.0 * h_rng
        else:
            hd_upper = h_max + 3.0 * h_rng
        hd_lower = h_min - 0.5 * h_rng

        x0 = np.array([hd0, np.log(tau0)], dtype=float)
        lb = np.array([hd_lower, np.log(1e-3)], dtype=float)
        ub = np.array([hd_upper, np.log(1e9)], dtype=float)

        def residuals(x: np.ndarray) -> np.ndarray:
            hd_fit, logtau = x
            tau = np.exp(logtau)
            model = hd_fit - (hd_fit - h0) * np.exp(-t / tau)
            return h - model

        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), loss="huber", max_nfev=10000)
            hd_fit = float(res.x[0])
            tau_fit = float(np.exp(res.x[1]))
            r = residuals(res.x)
            rmse = float(np.sqrt(np.mean(r**2)))
            ss_tot = float(np.sum((h - np.mean(h)) ** 2))
            r2 = float(1 - np.sum(r**2) / ss_tot) if ss_tot > 0 else float("nan")
            return FitResult(hd_fit, tau_fit, rmse, r2, bool(res.success))
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
            end_idx = int(ends[0]) if len(ends) > 0 else int(df.index[-1]) + 1

            on_window = df.loc[start_idx : end_idx - 1]
            if len(on_window) < self.min_cycle_points:
                continue

            q = on_window["caudal_m3s"]
            duration_min = (on_window["ts"].iloc[-1] - on_window["ts"].iloc[0]).total_seconds() / 60.0
            n_points = int(len(on_window))

            dt_arr = on_window["ts"].diff().dt.total_seconds().to_numpy()
            dt_arr[0] = float(np.nanmedian(dt_arr[1:])) if len(dt_arr) > 1 else 0.0
            volumen_bombeado_l = float(np.nansum((q.to_numpy() * 1000.0) * np.nan_to_num(dt_arr)))
            vol_m3 = float(volumen_bombeado_l / 1000.0)
            if not np.isclose(vol_m3, volumen_bombeado_l / 1000.0):
                raise AssertionError("Conversión inválida: volumen_bombeado_m3 debe ser volumen_bombeado_l/1000")

            t_on = (on_window["ts"] - on_window["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
            h_on = on_window["h"].to_numpy(float)
            h_static = self._select_static_level(df, start_idx=start_idx, end_idx=end_idx)
            fit_on = self._fit_on_segment(t_on=t_on, h_on=h_on, h_static=h_static)

            hd_fit = fit_on.y_inf
            tau_fit = fit_on.tau_s
            c_mean = float(np.nanmean(q.to_numpy(float)))
            den = hd_fit - h_static if np.isfinite(h_static) and np.isfinite(hd_fit) else np.nan

            ok_fit = bool(fit_on.ok and np.isfinite(fit_on.r2) and fit_on.r2 >= self.fit_min_r2)
            ok_k = False
            k = np.nan

            if np.isfinite(tau_fit) and tau_fit > 1e7:
                logger.warning("tau extremadamente grande detectado en ciclo (%s s)", tau_fit)

            if ok_fit and np.isfinite(den) and abs(den) >= self.eps_den and c_mean > 0:
                k = float(c_mean / den)
                ok_k = bool(np.isfinite(k) and k > 0)

            if np.isfinite(k) and abs(k) < 1e-12:
                logger.warning("k cercano a cero detectado (%.3e). Posible comportamiento cuasi-estático.", k)

            h_static_nivel = -h_static if self.depth_like and np.isfinite(h_static) else h_static
            h_d_nivel = -hd_fit if self.depth_like and np.isfinite(hd_fit) else hd_fit

            cycles_data.append(
                {
                    "start_ts": on_window["ts"].iloc[0],
                    "end_ts": on_window["ts"].iloc[-1],
                    "duration_min": duration_min,
                    "q_median_m3s": q.median(),
                    "q_std_m3s": q.std(),
                    "h_median": on_window["nivel_m"].median(),
                    "h_static_m": h_static_nivel,
                    "h_dinamico_m": h_d_nivel,
                    "hd_fit": hd_fit,
                    "tau_fit": tau_fit,
                    "tau_on_s": tau_fit,
                    "h_static": h_static,
                    "C_m3s": c_mean,
                    "k": k,
                    "k_m2_s": k,
                    "ok_fit": ok_fit,
                    "ok_k": ok_k,
                    "rmse": fit_on.rmse,
                    "r2": fit_on.r2,
                    "n_points": n_points,
                    "volumen_bombeado_m3": vol_m3,
                }
            )

        self.cycles_df = pd.DataFrame(cycles_data)
        self._add_cycle_derived_metrics()
        self._compute_period_aggregations()

    def _add_cycle_derived_metrics(self) -> None:
        if self.cycles_df is None or self.cycles_df.empty:
            return

        self.cycles_df = self.cycles_df.sort_values("start_ts").reset_index(drop=True)
        next_start = self.cycles_df["start_ts"].shift(-1)
        self.cycles_df["tiempo_entre_encendidos_s"] = (next_start - self.cycles_df["end_ts"]).dt.total_seconds()

        invalid_gap = self.cycles_df["tiempo_entre_encendidos_s"].dropna() < 0
        if bool(invalid_gap.any()):
            raise ValueError("Se detectaron tiempos entre encendidos negativos")

    def _compute_period_aggregations(self) -> None:
        self.period_aggregations = {}
        if self.cycles_df is None or self.cycles_df.empty:
            return

        for freq_name, freq in (("day", "D"), ("week", "W"), ("month", "M")):
            period_df = self.cycles_df.copy()
            period_df["period_start"] = period_df["start_ts"].dt.to_period(freq).dt.start_time
            agg = (
                period_df.groupby("period_start", dropna=False)
                .agg(
                    tiempo_entre_encendidos_mean=("tiempo_entre_encendidos_s", "mean"),
                    tiempo_entre_encendidos_std=("tiempo_entre_encendidos_s", "std"),
                    tiempo_entre_encendidos_n=("tiempo_entre_encendidos_s", "count"),
                    k_m2_s_mean=("k_m2_s", "mean"),
                    k_m2_s_std=("k_m2_s", "std"),
                    k_m2_s_n=("k_m2_s", "count"),
                )
                .reset_index()
            )
            self.period_aggregations[freq_name] = agg

    @staticmethod
    def _series_stats(series: pd.Series) -> tuple[float, float, int]:
        clean = series.dropna()
        if clean.empty:
            return np.nan, np.nan, 0
        return float(clean.mean()), float(clean.std(ddof=1)), int(clean.count())

    def compute_global_metrics(self) -> None:
        if self.df is None:
            raise ValueError("El DataFrame no está cargado")

        t_total_days = (self.df["ts"].max() - self.df["ts"].min()).total_seconds() / 86400
        t_total_days = max(t_total_days, 1e-6)

        if self.cycles_df is not None and not self.cycles_df.empty:
            total_cycles = len(self.cycles_df)
            freq_cycles_day = total_cycles / t_total_days
            avg_duration = self.cycles_df["duration_min"].mean()
            median_q = self.cycles_df["q_median_m3s"].median()
            variability_q = self.cycles_df["q_std_m3s"].mean()
            total_on_minutes = self.cycles_df["duration_min"].sum()
            duty_cycle_pct = (total_on_minutes / (t_total_days * 1440)) * 100

            vol_total_m3 = float(self.cycles_df["volumen_bombeado_m3"].sum())
            h_static_mean = float(np.nanmean(self.cycles_df["h_static_m"]))
            h_d_mean = float(np.nanmean(self.cycles_df["h_dinamico_m"]))
            k_mean = float(np.nanmean(self.cycles_df["k_m2_s"]))
            tiempo_entre_mean, tiempo_entre_std, tiempo_entre_n = self._series_stats(
                self.cycles_df["tiempo_entre_encendidos_s"]
            )
            k_metric_mean, k_metric_std, k_metric_n = self._series_stats(self.cycles_df["k_m2_s"])
        else:
            if self.df["pump_on"].mean() > 0.95:
                duty_cycle_pct = 100.0
                freq_cycles_day = 0.0
                avg_duration = t_total_days * 1440
                median_q = self.df["caudal_m3s"].median()
                variability_q = self.df["caudal_m3s"].std()
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
            tiempo_entre_mean, tiempo_entre_std, tiempo_entre_n = np.nan, np.nan, 0
            k_metric_mean, k_metric_std, k_metric_n = np.nan, np.nan, 0

        def _safe_round(v: float, decimals: int) -> Optional[float]:
            return round(float(v), decimals) if np.isfinite(v) else None

        self.stats = {
            "duty_cycle_pct": round(float(duty_cycle_pct), 2),
            "freq_cycles_day": round(float(freq_cycles_day), 2),
            "avg_cycle_duration_min": round(float(avg_duration), 1),
            "typical_flow_m3s": round(float(median_q), 6),
            "flow_stability_m3s_std": round(float(variability_q), 6),
            "vol_total_m3": round(float(vol_total_m3), 3),
            "h_static_mean_m": _safe_round(h_static_mean, 3),
            "h_dinamico_mean_m": _safe_round(h_d_mean, 3),
            "k_mean_m2_s": _safe_round(k_mean, 6),
            "tiempo_entre_encendidos_mean": _safe_round(tiempo_entre_mean, 3),
            "tiempo_entre_encendidos_std": _safe_round(tiempo_entre_std, 3),
            "tiempo_entre_encendidos_n": int(tiempo_entre_n),
            "k_m2_s_mean": _safe_round(k_metric_mean, 6),
            "k_m2_s_std": _safe_round(k_metric_std, 6),
            "k_m2_s_n": int(k_metric_n),
            "nivel_es_profundidad": self.depth_like,
            "dynamic_threshold_m3s_used": round(float(self.threshold), 6),
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
