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
    def compute_stats(values: Any) -> dict[str, float | int]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = int(len(arr))
        if n == 0:
            return {"median": np.nan, "mean": np.nan, "std": np.nan, "n": 0}

        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
        return {"median": median, "mean": mean, "std": std, "n": n}

    def add_stats_cols(self, row: dict[str, Any], base_name: str, values: Any) -> None:
        stats = self.compute_stats(values)
        row[base_name] = self._round_float(stats["median"])
        row[f"{base_name}_mean"] = self._round_float(stats["mean"])
        row[f"{base_name}_std"] = self._round_float(stats["std"])
        row[f"{base_name}_n"] = int(stats["n"])

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


    def _extract_segments_from_df(self, dfx: pd.DataFrame) -> pd.DataFrame:
        if dfx.empty:
            return pd.DataFrame(columns=["segment_id", "start_idx", "end_idx", "is_on", "inicio", "fin", "h0", "n_points", "duracion_s", "delta_h"])

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
        return pd.DataFrame(rows)

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

            used_in_global_fit = seg_id in h_inf_map
            h_inf = h_inf_map.get(seg_id, self._tail_median(h))
            tau_local = tau_s if np.isfinite(tau_s) else np.nan

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
                    "used_in_global_fit": int(used_in_global_fit),
                    "ok_fit": bool(metrics.get("ok_fit_global", False) and used_in_global_fit),
                    "rmse": rmse,
                    "r2": r2,
                    "h_inf_nivel_m": float(h_inf) if np.isfinite(h_inf) else np.nan,
                    "tau_s": float(tau_local) if np.isfinite(tau_local) else np.nan,
                    "C_const_ls": float(np.nanmedian(part["caudal_use"].to_numpy(float))) if int(seg["is_on"]) == 1 else np.nan,
                }
            )

        self.segment_metrics_df = self._round_df_numeric(pd.DataFrame(rows))
        return self.segment_metrics_df

    def _stats_order_columns(self, row: dict[str, Any], base_names: list[str]) -> dict[str, Any]:
        ordered: dict[str, Any] = {}
        for key in row:
            if key in base_names or any(key == f"{b}_{s}" for b in base_names for s in ["mean", "std", "n"]):
                continue
            ordered[key] = row[key]

        for b in base_names:
            ordered[b] = row.get(b, np.nan)
            ordered[f"{b}_mean"] = row.get(f"{b}_mean", np.nan)
            ordered[f"{b}_std"] = row.get(f"{b}_std", np.nan)
            ordered[f"{b}_n"] = row.get(f"{b}_n", 0)

        for key in row:
            if key not in ordered:
                ordered[key] = row[key]

        return ordered

    def build_cycle_table(self, device_id: str = "") -> pd.DataFrame:
        if self.segment_metrics_df is None:
            self.compute_segment_metrics()
        assert self.segment_metrics_df is not None

        seg = self.segment_metrics_df.reset_index(drop=True)
        fit_seg = seg[seg["used_in_global_fit"] == 1]
        hs_values_all = fit_seg.loc[fit_seg["is_on"] == 0, "h_inf_nivel_m"].to_numpy(float)
        hd_values_all = fit_seg.loc[fit_seg["is_on"] == 1, "h_inf_nivel_m"].to_numpy(float)
        c_values_all = fit_seg.loc[fit_seg["is_on"] == 1, "C_const_ls"].to_numpy(float)

        tau_global = self.global_fit_metrics.get("tau_s", np.nan)
        if not np.isfinite(tau_global):
            tau_candidates = fit_seg["tau_s"].to_numpy(float)
            tau_global = float(np.nanmedian(tau_candidates)) if len(tau_candidates) else np.nan

        on_idx = seg.index[seg["is_on"] == 1].tolist()
        rows = []
        for cyc_id, idx in enumerate(on_idx, start=1):
            on = seg.loc[idx]
            row = {
                "device_id": device_id,
                "ciclo_id": cyc_id,
                "inicio": on["inicio"],
                "fin": on["fin"],
                "n_on": 1,
                "frecuencia_encendido_por_dia": np.nan,
                "tiempo_on_prom_s": self._round_float(float(on["duracion_s"])),
                "ok_fit_global": bool(self.global_fit_metrics.get("ok_fit_global", False)),
                "rmse_global": self._round_float(float(self.global_fit_metrics.get("rmse_global", np.nan))),
                "r2_global": self._round_float(float(self.global_fit_metrics.get("r2_global", np.nan))),
            }
            self.add_stats_cols(row, "h_static_nivel_m", hs_values_all)
            self.add_stats_cols(row, "h_dinamico_nivel_m", hd_values_all)
            self.add_stats_cols(row, "C_const_ls", c_values_all)

            tau_n = int(self.global_fit_metrics.get("n_valid_segments", 0))
            row["tau_s"] = self._round_float(float(tau_global)) if np.isfinite(tau_global) else np.nan
            row["tau_s_mean"] = row["tau_s"]
            row["tau_s_std"] = 0.0 if tau_n > 0 and np.isfinite(tau_global) else np.nan
            row["tau_s_n"] = tau_n

            rows.append(self._stats_order_columns(row, ["h_static_nivel_m", "h_dinamico_nivel_m", "tau_s", "C_const_ls"]))

        out = pd.DataFrame(rows)
        for c in ["h_static_nivel_m_n", "h_dinamico_nivel_m_n", "tau_s_n", "C_const_ls_n", "n_on"]:
            if c in out.columns:
                out[c] = out[c].fillna(0).astype(int)
        return self._round_df_numeric(out)

    def aggregate_periods(self, period_days: float = 2.0, device_id: str = "") -> pd.DataFrame:
        if period_days <= 0:
            raise ValueError("period_days debe ser > 0")
        if self.df is None:
            raise ValueError("DataFrame no cargado")

        dfx_all = self.df.copy().reset_index(drop=True)
        start_ts = dfx_all["ts"].min()
        period_s = period_days * 86400.0
        dfx_all["periodo"] = (((dfx_all["ts"] - start_ts).dt.total_seconds()) // period_s).astype(int) + 1

        rows = []
        for periodo, dfp in dfx_all.groupby("periodo", sort=True):
            dfp = dfp.reset_index(drop=True)
            seg_local = self._extract_segments_from_df(dfp)

            if seg_local.empty:
                continue

            tau_s, h_inf_map, fit_metrics = self.fit_global_tau_and_hinf(
                segments=seg_local,
                ts=dfp["ts"],
                nivel_m=dfp["nivel_use"],
            )

            seg_rows = []
            for _, seg in seg_local.iterrows():
                seg_id = int(seg["segment_id"])
                sidx = int(seg["start_idx"])
                eidx = int(seg["end_idx"])
                part = dfp.loc[sidx:eidx]
                h = part["nivel_use"].to_numpy(float)
                h0 = float(h[0])
                t = (part["ts"] - part["ts"].iloc[0]).dt.total_seconds().to_numpy(float)

                used_in_fit = seg_id in h_inf_map
                h_inf = h_inf_map.get(seg_id, self._tail_median(h))

                if np.isfinite(tau_s):
                    pred = h_inf + (h0 - h_inf) * np.exp(-t / tau_s)
                    resid = h - pred
                    rmse = float(np.sqrt(np.mean(resid**2)))
                    ss_tot = float(np.sum((h - np.mean(h)) ** 2))
                    r2 = float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan
                else:
                    rmse = np.nan
                    r2 = np.nan

                seg_rows.append(
                    {
                        "segment_id": seg_id,
                        "is_on": int(seg["is_on"]),
                        "inicio": seg["inicio"],
                        "fin": seg["fin"],
                        "duracion_s": float(seg["duracion_s"]),
                        "used_in_global_fit": int(used_in_fit),
                        "h_inf_nivel_m": float(h_inf) if np.isfinite(h_inf) else np.nan,
                        "tau_s": float(tau_s) if np.isfinite(tau_s) else np.nan,
                        "rmse": rmse,
                        "r2": r2,
                        "C_const_ls": float(np.nanmedian(part["caudal_use"].to_numpy(float))) if int(seg["is_on"]) == 1 else np.nan,
                    }
                )

            seg_metrics_local = pd.DataFrame(seg_rows)
            fit_seg = seg_metrics_local[seg_metrics_local["used_in_global_fit"] == 1]
            on_all = seg_metrics_local[seg_metrics_local["is_on"] == 1]
            on_fit = fit_seg[fit_seg["is_on"] == 1]
            off_fit = fit_seg[fit_seg["is_on"] == 0]

            inicio = seg_metrics_local["inicio"].min()
            fin = seg_metrics_local["fin"].max()
            dur_days = max((fin - inicio).total_seconds() / 86400.0, period_days)
            n_on = int(len(on_all))
            freq = self._round_float(float(n_on / dur_days)) if dur_days > 0 else 0.0

            hs_values = off_fit["h_inf_nivel_m"].to_numpy(float)
            hd_values = on_fit["h_inf_nivel_m"].to_numpy(float)
            c_values = on_fit["C_const_ls"].to_numpy(float)

            tau_n = int(fit_metrics.get("n_valid_segments", 0)) if np.isfinite(tau_s) else 0

            row = {
                "device_id": device_id,
                "periodo": int(periodo),
                "inicio": inicio,
                "fin": fin,
                "n_on": n_on,
                "frecuencia_encendido_por_dia": freq,
                "tiempo_on_prom_s": self._round_float(float(np.nanmean(on_all["duracion_s"].to_numpy(float)))) if len(on_all) else np.nan,
                "ok_fit_global": bool(fit_metrics.get("ok_fit_global", False)),
                "rmse_global": self._round_float(float(fit_metrics.get("rmse_global", np.nan))),
                "r2_global": self._round_float(float(fit_metrics.get("r2_global", np.nan))),
            }
            self.add_stats_cols(row, "h_static_nivel_m", hs_values)
            self.add_stats_cols(row, "h_dinamico_nivel_m", hd_values)
            self.add_stats_cols(row, "C_const_ls", c_values)

            row["tau_s"] = self._round_float(float(tau_s)) if np.isfinite(tau_s) else np.nan
            row["tau_s_mean"] = row["tau_s"]
            row["tau_s_std"] = 0.0 if tau_n > 0 and np.isfinite(tau_s) else np.nan
            row["tau_s_n"] = tau_n

            rows.append(self._stats_order_columns(row, ["h_static_nivel_m", "h_dinamico_nivel_m", "tau_s", "C_const_ls"]))

        out = pd.DataFrame(rows)
        for c in ["h_static_nivel_m_n", "h_dinamico_nivel_m_n", "tau_s_n", "C_const_ls_n", "n_on"]:
            if c in out.columns:
                out[c] = out[c].fillna(0).astype(int)
        return self._round_df_numeric(out)

