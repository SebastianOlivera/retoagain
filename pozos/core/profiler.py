from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
    """Perfilador de pozo con tau global compartido entre segmentos."""

    def __init__(
        self,
        min_segment_points: int = 10,
        smooth_window: int = 1,
        flat_std_eps: float = 1e-6,
        min_seg_dur_s: float = 1800.0,
        min_delta_h: float = 0.01,
        tau_min_s: float = 60.0,
        tau_max_s: float = 30.0 * 86400.0,
        h_bounds_margin: float = 0.5,
    ) -> None:
        self.min_segment_points = min_segment_points
        self.smooth_window = smooth_window
        self.flat_std_eps = flat_std_eps
        self.min_seg_dur_s = min_seg_dur_s
        self.min_delta_h = min_delta_h
        self.tau_min_s = tau_min_s
        self.tau_max_s = tau_max_s
        self.h_bounds_margin = h_bounds_margin

        self.df: Optional[pd.DataFrame] = None
        self.threshold: float = 0.05
        self.segments_df: Optional[pd.DataFrame] = None
        self.segment_metrics_df: Optional[pd.DataFrame] = None
        self.global_fit_metrics: dict[str, Any] = {}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        threshold: float = 0.05,
        min_segment_points: int = 10,
        smooth_window: int = 1,
        min_seg_dur_s: float = 1800.0,
        min_delta_h: float = 0.01,
        tau_min_s: float = 60.0,
        tau_max_s: float = 30.0 * 86400.0,
        h_bounds_margin: float = 0.5,
    ) -> "WellProfiler":
        instance = cls(
            min_segment_points=min_segment_points,
            smooth_window=smooth_window,
            min_seg_dur_s=min_seg_dur_s,
            min_delta_h=min_delta_h,
            tau_min_s=tau_min_s,
            tau_max_s=tau_max_s,
            h_bounds_margin=h_bounds_margin,
        )
        instance.threshold = float(threshold)
        instance.df = instance._prepare_dataframe(df)
        return instance

    @staticmethod
    def _round_float(value: float, ndigits: int = 3) -> float:
        if value is None or not np.isfinite(value):
            return np.nan
        return round(float(value), ndigits)

    @staticmethod
    def _round_df_numeric(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].round(ndigits)
        return out

    @staticmethod
    def _std_ddof1_zero(values: np.ndarray) -> float:
        vals = values[np.isfinite(values)]
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals, ddof=1))

    @staticmethod
    def _tail_median(values: np.ndarray, tail_ratio: float = 0.2) -> float:
        vals = values[np.isfinite(values)]
        if len(vals) == 0:
            return np.nan
        n_tail = max(3, int(np.ceil(len(vals) * tail_ratio)))
        return float(np.nanmedian(vals[-n_tail:]))

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
            h_seg = part["nivel_use"].to_numpy(float)
            rows.append(
                {
                    "segment_id": seg_id,
                    "start_idx": int(s),
                    "end_idx": int(e),
                    "is_on": int(is_on),
                    "inicio": part["ts"].iloc[0],
                    "fin": part["ts"].iloc[-1],
                    "h0": self._round_float(float(h_seg[0])),
                    "n_points": int(len(part)),
                    "duracion_s": self._round_float(float((part["ts"].iloc[-1] - part["ts"].iloc[0]).total_seconds())),
                    "delta_h": self._round_float(float(np.nanmax(h_seg) - np.nanmin(h_seg))),
                }
            )

        self.segments_df = pd.DataFrame(rows)
        return self.segments_df

    def fit_global_tau_and_hinf(
        self,
        segments: pd.DataFrame,
        ts: pd.Series,
        nivel_m: pd.Series,
    ) -> tuple[float, dict[int, float], dict[str, Any]]:
        tsv = pd.to_datetime(ts, utc=True)
        hv = pd.to_numeric(nivel_m, errors="coerce").to_numpy(float)

        valid_rows: list[dict[str, Any]] = []
        for _, seg in segments.iterrows():
            s = int(seg["start_idx"])
            e = int(seg["end_idx"])
            n_points = int(seg["n_points"])
            dur_s = float(seg["duracion_s"])
            h_seg = hv[s : e + 1]
            delta_h = float(np.nanmax(h_seg) - np.nanmin(h_seg)) if len(h_seg) else np.nan
            if (
                n_points >= self.min_segment_points
                and dur_s >= self.min_seg_dur_s
                and np.isfinite(delta_h)
                and delta_h >= self.min_delta_h
            ):
                valid_rows.append(
                    {
                        "segment_id": int(seg["segment_id"]),
                        "start_idx": s,
                        "end_idx": e,
                        "is_on": int(seg["is_on"]),
                    }
                )

        if len(valid_rows) == 0 or np.nanstd(hv) <= self.flat_std_eps or pd.Series(hv).nunique(dropna=True) <= 1:
            return np.nan, {}, {
                "ok_fit_global": False,
                "n_valid_segments": 0,
                "rmse_global": np.nan,
                "r2_global": np.nan,
                "reason": "Sin segmentos válidos o nivel sin variación",
            }

        tau_init = max(self.min_seg_dur_s, 3600.0)
        x0 = [np.log(max(min(tau_init, self.tau_max_s), self.tau_min_s))]
        lb = [np.log(self.tau_min_s)]
        ub = [np.log(self.tau_max_s)]

        segment_arrays = []
        for row in valid_rows:
            s = row["start_idx"]
            e = row["end_idx"]
            t = (tsv.iloc[s : e + 1] - tsv.iloc[s]).dt.total_seconds().to_numpy(float)
            y = hv[s : e + 1]
            h_inf_init = self._tail_median(y)
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))

            x0.append(h_inf_init)
            lb.append(y_min - self.h_bounds_margin)
            ub.append(y_max + self.h_bounds_margin)
            segment_arrays.append((row["segment_id"], t, y, float(y[0])))

        x0_arr = np.asarray(x0, dtype=float)
        lb_arr = np.asarray(lb, dtype=float)
        ub_arr = np.asarray(ub, dtype=float)

        def residuals(p: np.ndarray) -> np.ndarray:
            tau = np.exp(p[0])
            all_res = []
            for j, (_, t, y, h0) in enumerate(segment_arrays):
                h_inf = p[1 + j]
                pred = h_inf + (h0 - h_inf) * np.exp(-t / tau)
                all_res.append(y - pred)
            return np.concatenate(all_res)

        try:
            res = least_squares(residuals, x0_arr, bounds=(lb_arr, ub_arr), loss="huber", max_nfev=20000)
            p_opt = res.x
            tau_s = float(np.exp(p_opt[0]))
            resid = residuals(p_opt)
            rmse = float(np.sqrt(np.mean(resid**2)))

            y_all = np.concatenate([arr[2] for arr in segment_arrays])
            sse = float(np.sum(resid**2))
            sst = float(np.sum((y_all - np.mean(y_all)) ** 2))
            r2 = float(1 - sse / sst) if sst > 0 else np.nan

            h_inf_per_segment = {
                int(segment_arrays[j][0]): float(p_opt[1 + j]) for j in range(len(segment_arrays))
            }
            ok = bool(res.success) and np.isfinite(rmse)
            return tau_s, h_inf_per_segment, {
                "ok_fit_global": ok,
                "n_valid_segments": len(valid_rows),
                "rmse_global": rmse,
                "r2_global": r2,
                "status": int(res.status),
                "message": str(res.message),
            }
        except Exception as exc:  # noqa: BLE001
            return np.nan, {}, {
                "ok_fit_global": False,
                "n_valid_segments": len(valid_rows),
                "rmse_global": np.nan,
                "r2_global": np.nan,
                "reason": str(exc),
            }

    def compute_segment_metrics(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("DataFrame no cargado")
        if self.segments_df is None:
            self.extract_segments()
        assert self.segments_df is not None

        tau_s, h_inf_map, metrics = self.fit_global_tau_and_hinf(
            segments=self.segments_df,
            ts=self.df["ts"],
            nivel_m=self.df["nivel_use"],
        )
        self.global_fit_metrics = metrics

        rows = []
        for _, seg in self.segments_df.iterrows():
            seg_id = int(seg["segment_id"])
            s = int(seg["start_idx"])
            e = int(seg["end_idx"])
            part = self.df.loc[s:e]
            h = part["nivel_use"].to_numpy(float)
            h0 = float(h[0])
            t = (part["ts"] - part["ts"].iloc[0]).dt.total_seconds().to_numpy(float)

            h_inf = h_inf_map.get(seg_id, self._tail_median(h))
            tau_local = tau_s if np.isfinite(tau_s) else np.nan
            ok_fit = bool(metrics.get("ok_fit_global", False) and seg_id in h_inf_map)

            if np.isfinite(tau_local):
                pred = h_inf + (h0 - h_inf) * np.exp(-t / tau_local)
                resid = h - pred
                rmse = float(np.sqrt(np.mean(resid**2)))
                ss_tot = float(np.sum((h - np.mean(h)) ** 2))
                r2 = float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan
            else:
                rmse = np.nan
                r2 = np.nan

            rows.append(
                {
                    "segment_id": seg_id,
                    "is_on": int(seg["is_on"]),
                    "inicio": seg["inicio"],
                    "fin": seg["fin"],
                    "n_points": int(seg["n_points"]),
                    "duracion_s": float(seg["duracion_s"]),
                    "ok_fit": ok_fit,
                    "rmse": rmse,
                    "r2": r2,
                    "h_inf_nivel_m": float(h_inf) if np.isfinite(h_inf) else np.nan,
                    "tau_s": float(tau_local) if np.isfinite(tau_local) else np.nan,
                    "C_const_ls": float(np.nanmedian(part["caudal_use"].to_numpy(float))) if int(seg["is_on"]) == 1 else np.nan,
                }
            )

        self.segment_metrics_df = self._round_df_numeric(pd.DataFrame(rows))
        return self.segment_metrics_df

    def build_cycle_table(self, device_id: str = "") -> pd.DataFrame:
        if self.segment_metrics_df is None:
            self.compute_segment_metrics()
        assert self.segment_metrics_df is not None

        seg = self.segment_metrics_df.reset_index(drop=True)
        tau_global = self._round_float(float(seg["tau_s"].dropna().median())) if seg["tau_s"].notna().any() else np.nan
        on_idx = seg.index[seg["is_on"] == 1].tolist()
        rows = []

        for cyc_id, idx in enumerate(on_idx, start=1):
            on = seg.loc[idx]
            off_prev = seg.loc[idx - 1] if idx - 1 >= 0 and int(seg.loc[idx - 1, "is_on"]) == 0 else None
            off_next = seg.loc[idx + 1] if idx + 1 < len(seg) and int(seg.loc[idx + 1, "is_on"]) == 0 else None
            hs_candidates = []
            for off in (off_prev, off_next):
                if off is not None:
                    hs_candidates.append(float(off["h_inf_nivel_m"]))

            rows.append(
                {
                    "device_id": device_id,
                    "ciclo_id": cyc_id,
                    "inicio": on["inicio"],
                    "fin": on["fin"],
                    "h_static_nivel_m": self._round_float(float(np.nanmedian(hs_candidates))) if len(hs_candidates) else np.nan,
                    "h_dinamico_nivel_m": self._round_float(float(on["h_inf_nivel_m"])),
                    "tau_s": tau_global,
                    "C_const_ls": self._round_float(float(on["C_const_ls"])),
                    "tiempo_on_prom_s": self._round_float(float(on["duracion_s"])),
                    "ok_fit_global": bool(self.global_fit_metrics.get("ok_fit_global", False)),
                    "rmse_global": self._round_float(float(self.global_fit_metrics.get("rmse_global", np.nan))),
                    "r2_global": self._round_float(float(self.global_fit_metrics.get("r2_global", np.nan))),
                }
            )

        return self._round_df_numeric(pd.DataFrame(rows))

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

        tau_global = self._round_float(float(seg["tau_s"].dropna().median())) if seg["tau_s"].notna().any() else np.nan

        rows = []
        for periodo, part in seg.groupby("periodo", sort=True):
            on = part[part["is_on"] == 1]
            off = part[part["is_on"] == 0]

            inicio = part["inicio"].min()
            fin = part["fin"].max()
            dur_days = max((fin - inicio).total_seconds() / 86400.0, period_days)
            n_on = int(len(on))
            freq = self._round_float(float(n_on / dur_days)) if dur_days > 0 else 0.0

            hs_vals = off["h_inf_nivel_m"].to_numpy(float)
            hd_vals = on["h_inf_nivel_m"].to_numpy(float)
            c_vals = on["C_const_ls"].to_numpy(float)
            on_dur_vals = on["duracion_s"].to_numpy(float)

            rows.append(
                {
                    "device_id": device_id,
                    "periodo": int(periodo),
                    "inicio": inicio,
                    "fin": fin,
                    "n_on": n_on,
                    "h_static_nivel_m": self._round_float(float(np.nanmedian(hs_vals))) if len(hs_vals) else np.nan,
                    "h_dinamico_nivel_m": self._round_float(float(np.nanmedian(hd_vals))) if len(hd_vals) else np.nan,
                    "tau_s": tau_global,
                    "C_const_ls": self._round_float(float(np.nanmedian(c_vals))) if len(c_vals) else np.nan,
                    "frecuencia_encendido_por_dia": freq,
                    "tiempo_on_prom_s": self._round_float(float(np.nanmean(on_dur_vals))) if len(on_dur_vals) else np.nan,
                    "hs_std": self._round_float(self._std_ddof1_zero(hs_vals)),
                    "hd_std": self._round_float(self._std_ddof1_zero(hd_vals)),
                    "C_std": self._round_float(self._std_ddof1_zero(c_vals)),
                    "ok_fit_global": bool(self.global_fit_metrics.get("ok_fit_global", False)),
                    "rmse_global": self._round_float(float(self.global_fit_metrics.get("rmse_global", np.nan))),
                    "r2_global": self._round_float(float(self.global_fit_metrics.get("r2_global", np.nan))),
                }
            )

        return self._round_df_numeric(pd.DataFrame(rows))
