from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class FitResult:
    h_inf: float
    tau_s: float
    rmse: float
    r2: float
    ok: bool


class WellProfiler:
    """Perfilador de pozo con métricas por segmento, ciclo y período."""

    def __init__(self, min_segment_points: int = 5, smooth_window: int = 1, flat_std_eps: float = 1e-6) -> None:
        self.min_segment_points = min_segment_points
        self.smooth_window = smooth_window
        self.flat_std_eps = flat_std_eps
        self.df: Optional[pd.DataFrame] = None
        self.threshold: float = 0.05
        self.segments_df: Optional[pd.DataFrame] = None
        self.segment_metrics_df: Optional[pd.DataFrame] = None

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        threshold: float = 0.05,
        min_segment_points: int = 5,
        smooth_window: int = 1,
    ) -> "WellProfiler":
        instance = cls(min_segment_points=min_segment_points, smooth_window=smooth_window)
        instance.threshold = float(threshold)
        instance.df = instance._prepare_dataframe(df)
        return instance

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"ts", "nivel_m", "caudal_ls"}
        if not required.issubset(df.columns):
            raise ValueError(f"Faltan columnas requeridas: {sorted(required)}")

        dfx = df.copy()
        dfx["ts"] = pd.to_datetime(dfx["ts"], errors="coerce", utc=True)
        dfx["nivel_m"] = pd.to_numeric(dfx["nivel_m"], errors="coerce")
        dfx["caudal_ls"] = pd.to_numeric(dfx["caudal_ls"], errors="coerce")
        dfx = dfx.dropna(subset=["ts", "nivel_m", "caudal_ls"]).sort_values("ts").reset_index(drop=True)

        if dfx.empty:
            raise ValueError("No hay datos válidos tras limpieza")

        if "estado_bomba" in dfx.columns:
            dfx["state"] = pd.to_numeric(dfx["estado_bomba"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        else:
            dfx["state"] = (dfx["caudal_ls"] > self.threshold).astype(int)

        if self.smooth_window > 1:
            dfx["nivel_use"] = dfx["nivel_m"].rolling(self.smooth_window, center=True, min_periods=1).median()
            dfx["caudal_use"] = dfx["caudal_ls"].rolling(self.smooth_window, center=True, min_periods=1).median()
        else:
            dfx["nivel_use"] = dfx["nivel_m"]
            dfx["caudal_use"] = dfx["caudal_ls"]

        return dfx

    @staticmethod
    def _std_ddof1_zero(values: np.ndarray) -> float:
        vals = values[np.isfinite(values)]
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals, ddof=1))

    def extract_segments(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("DataFrame no cargado")

        dfx = self.df
        segments = []
        start = 0
        current = int(dfx.loc[0, "state"])

        for idx in range(1, len(dfx)):
            st = int(dfx.loc[idx, "state"])
            if st != current:
                segments.append((start, idx - 1, current))
                start = idx
                current = st
        segments.append((start, len(dfx) - 1, current))

        rows = []
        for seg_id, (s, e, is_on) in enumerate(segments, start=1):
            part = dfx.loc[s:e]
            rows.append(
                {
                    "segment_id": seg_id,
                    "start_idx": int(s),
                    "end_idx": int(e),
                    "is_on": int(is_on),
                    "inicio": part["ts"].iloc[0],
                    "fin": part["ts"].iloc[-1],
                    "h0": float(part["nivel_use"].iloc[0]),
                    "n_points": int(len(part)),
                    "duracion_s": float((part["ts"].iloc[-1] - part["ts"].iloc[0]).total_seconds()),
                }
            )

        self.segments_df = pd.DataFrame(rows)
        return self.segments_df

    @staticmethod
    def _tail_median(values: np.ndarray, tail_ratio: float = 0.2) -> float:
        vals = values[np.isfinite(values)]
        if len(vals) == 0:
            return np.nan
        n_tail = max(3, int(np.ceil(len(vals) * tail_ratio)))
        return float(np.nanmedian(vals[-n_tail:]))

    def _fit_segment(self, t_rel_s: np.ndarray, h: np.ndarray) -> FitResult:
        valid = np.isfinite(t_rel_s) & np.isfinite(h)
        t = t_rel_s[valid]
        y = h[valid]
        if len(y) < self.min_segment_points or float(np.nanstd(y)) <= self.flat_std_eps:
            return FitResult(self._tail_median(y), np.nan, np.nan, np.nan, False)

        h0 = float(y[0])
        h_inf0 = self._tail_median(y)
        tau0 = max(float(t[-1] - t[0]) / 3.0, 1.0)
        x0 = np.array([h_inf0, np.log(tau0)])

        p5, p95 = np.nanpercentile(y, [5, 95])
        margin = max(0.1, abs(p95 - p5) * 2.0)
        lb = np.array([p5 - margin, np.log(1.0)])
        ub = np.array([p95 + margin, np.log(1e9)])

        def residuals(x: np.ndarray) -> np.ndarray:
            h_inf, logtau = x
            tau = np.exp(logtau)
            return y - (h_inf + (h0 - h_inf) * np.exp(-(t - t[0]) / tau))

        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), loss="huber", max_nfev=10000)
            h_inf = float(res.x[0])
            tau = float(np.exp(res.x[1]))
            r = residuals(res.x)
            rmse = float(np.sqrt(np.mean(r**2)))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1 - np.sum(r**2) / ss_tot) if ss_tot > 0 else np.nan
            ok = bool(res.success) and np.isfinite(rmse)
            if ok:
                return FitResult(h_inf, tau, rmse, r2, True)
            return FitResult(self._tail_median(y), np.nan, rmse, r2, False)
        except Exception:
            return FitResult(self._tail_median(y), np.nan, np.nan, np.nan, False)

    def compute_segment_metrics(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("DataFrame no cargado")
        if self.segments_df is None:
            self.extract_segments()

        rows = []
        assert self.segments_df is not None

        for _, seg in self.segments_df.iterrows():
            part = self.df.loc[int(seg["start_idx"]) : int(seg["end_idx"])].copy()
            t_rel_s = (part["ts"] - part["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
            h = part["nivel_use"].to_numpy(float)
            fit = self._fit_segment(t_rel_s, h)

            row = {
                "segment_id": int(seg["segment_id"]),
                "is_on": int(seg["is_on"]),
                "inicio": seg["inicio"],
                "fin": seg["fin"],
                "n_points": int(seg["n_points"]),
                "duracion_s": float(seg["duracion_s"]),
                "ok_fit": bool(fit.ok),
                "rmse": fit.rmse,
                "r2": fit.r2,
                "h_inf_nivel_m": fit.h_inf,
                "tau_s": fit.tau_s,
                "C_const_ls": float(np.nanmedian(part["caudal_use"].to_numpy(float))) if int(seg["is_on"]) == 1 else np.nan,
            }
            rows.append(row)

        self.segment_metrics_df = pd.DataFrame(rows)
        return self.segment_metrics_df

    def build_cycle_table(self, device_id: str = "") -> pd.DataFrame:
        if self.segment_metrics_df is None:
            self.compute_segment_metrics()
        assert self.segment_metrics_df is not None

        seg = self.segment_metrics_df.reset_index(drop=True)
        on_idx = seg.index[seg["is_on"] == 1].tolist()
        rows = []

        for cyc_id, idx in enumerate(on_idx, start=1):
            on = seg.loc[idx]
            off_prev = seg.loc[idx - 1] if idx - 1 >= 0 and int(seg.loc[idx - 1, "is_on"]) == 0 else None
            off_next = seg.loc[idx + 1] if idx + 1 < len(seg) and int(seg.loc[idx + 1, "is_on"]) == 0 else None

            hs_candidates = []
            tau_off_candidates = []
            for off in (off_prev, off_next):
                if off is not None:
                    hs_candidates.append(float(off["h_inf_nivel_m"]))
                    tau_off_candidates.append(float(off["tau_s"]))

            hs = float(np.nanmedian(hs_candidates)) if len(hs_candidates) else np.nan
            tau_off = float(np.nanmedian(tau_off_candidates)) if len(tau_off_candidates) else np.nan

            rows.append(
                {
                    "device_id": device_id,
                    "ciclo_id": cyc_id,
                    "inicio": on["inicio"],
                    "fin": on["fin"],
                    "h_static_nivel_m": hs,
                    "h_dinamico_nivel_m": float(on["h_inf_nivel_m"]),
                    "tau_off_s": tau_off,
                    "tau_on_s": float(on["tau_s"]),
                    "C_const_ls": float(on["C_const_ls"]),
                    "tiempo_on_prom_s": float(on["duracion_s"]),
                    "ok_on": bool(on["ok_fit"]),
                    "rmse_on": float(on["rmse"]) if np.isfinite(on["rmse"]) else np.nan,
                    "r2_on": float(on["r2"]) if np.isfinite(on["r2"]) else np.nan,
                }
            )

        return pd.DataFrame(rows)

    def aggregate_periods(self, period_days: float = 2.0, device_id: str = "") -> pd.DataFrame:
        if period_days <= 0:
            raise ValueError("period_days debe ser > 0")
        if self.df is None:
            raise ValueError("DataFrame no cargado")
        if self.segment_metrics_df is None:
            self.compute_segment_metrics()
        assert self.segment_metrics_df is not None

        seg = self.segment_metrics_df.copy()
        seg["inicio"] = pd.to_datetime(seg["inicio"], utc=True)
        start_ts = self.df["ts"].min()
        period_s = period_days * 86400.0
        seg["periodo"] = (((seg["inicio"] - start_ts).dt.total_seconds()) // period_s).astype(int) + 1

        period_rows = []
        for periodo, part in seg.groupby("periodo", sort=True):
            on = part[part["is_on"] == 1]
            off = part[part["is_on"] == 0]

            inicio = part["inicio"].min()
            fin = part["fin"].max()
            dur_days = max((fin - inicio).total_seconds() / 86400.0, period_days)
            n_on = int(len(on))
            freq = float(n_on / dur_days) if dur_days > 0 else 0.0

            hs_vals = off["h_inf_nivel_m"].to_numpy(float)
            hd_vals = on["h_inf_nivel_m"].to_numpy(float)
            tau_off_vals = off["tau_s"].to_numpy(float)
            tau_on_vals = on["tau_s"].to_numpy(float)
            c_vals = on["C_const_ls"].to_numpy(float)
            on_dur_vals = on["duracion_s"].to_numpy(float)

            period_rows.append(
                {
                    "device_id": device_id,
                    "periodo": int(periodo),
                    "inicio": inicio,
                    "fin": fin,
                    "n_on": n_on,
                    "h_static_nivel_m": float(np.nanmedian(hs_vals)) if len(hs_vals) else np.nan,
                    "h_dinamico_nivel_m": float(np.nanmedian(hd_vals)) if len(hd_vals) else np.nan,
                    "tau_off_s": float(np.nanmedian(tau_off_vals)) if len(tau_off_vals) else np.nan,
                    "tau_on_s": float(np.nanmedian(tau_on_vals)) if len(tau_on_vals) else np.nan,
                    "C_const_ls": float(np.nanmedian(c_vals)) if len(c_vals) else np.nan,
                    "frecuencia_encendido_por_dia": freq,
                    "tiempo_on_prom_s": float(np.nanmean(on_dur_vals)) if len(on_dur_vals) else np.nan,
                    "hs_std": self._std_ddof1_zero(hs_vals),
                    "hd_std": self._std_ddof1_zero(hd_vals),
                    "tau_off_std": self._std_ddof1_zero(tau_off_vals),
                    "tau_on_std": self._std_ddof1_zero(tau_on_vals),
                    "C_std": self._std_ddof1_zero(c_vals),
                    "rmse_on": float(np.nanmean(on["rmse"].to_numpy(float))) if len(on) else np.nan,
                    "r2_on": float(np.nanmean(on["r2"].to_numpy(float))) if len(on) else np.nan,
                    "ok_on": float(np.nanmean(on["ok_fit"].to_numpy(float))) if len(on) else np.nan,
                    "rmse_off": float(np.nanmean(off["rmse"].to_numpy(float))) if len(off) else np.nan,
                    "r2_off": float(np.nanmean(off["r2"].to_numpy(float))) if len(off) else np.nan,
                    "ok_off": float(np.nanmean(off["ok_fit"].to_numpy(float))) if len(off) else np.nan,
                }
            )

        return pd.DataFrame(period_rows)
