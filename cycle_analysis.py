#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def robust_median_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).median()


def detect_segments(state: np.ndarray) -> List[Tuple[int, int, bool]]:
    segs: List[Tuple[int, int, bool]] = []
    s = 0
    cur = bool(state[0])
    for i in range(1, len(state)):
        if bool(state[i]) != cur:
            segs.append((s, i - 1, cur))
            s = i
            cur = bool(state[i])
    segs.append((s, len(state) - 1, cur))
    return segs


def detect_depth_like(nivel: np.ndarray, pump_on: np.ndarray) -> bool:
    # Si durante ON el "nivel" numérico sube, suele ser profundidad.
    if pump_on.sum() < 10 or (~pump_on).sum() < 10:
        return False
    return np.nanmedian(nivel[pump_on]) > np.nanmedian(nivel[~pump_on])


def fit_exp_closed_form(t: np.ndarray, y: np.ndarray, mode: str):
    """
    Ajuste por mínimos cuadrados del modelo:
        y(t) = y_inf + (y0 - y_inf)*exp(-t/tau)

    mode: "off" o "on" (solo cambia nombres; fórmula es igual).
    Ajustamos parámetros: y_inf y tau.
    y0 se toma como y[0] (dato inicial) para estabilizar.
    """
    _ = mode
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])

    # Inicial: y_inf ~ mediana cola, tau ~ t_final/3.
    y_inf0 = float(np.nanmedian(y[-max(5, len(y) // 5) :]))
    tau0 = max(float(t[-1] - t[0]) / 3.0, 1.0)

    # Parametrización: logtau > 0.
    x0 = np.array([y_inf0, np.log(tau0)], dtype=float)

    def residuals(x):
        y_inf, logtau = x
        tau = np.exp(logtau)
        y_hat = y_inf + (y0 - y_inf) * np.exp(-(t - t[0]) / tau)
        return y - y_hat

    # Bounds suaves.
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    rng = abs(ymax - ymin) + 1e-6
    lb = np.array([ymin - 10 * rng, np.log(1e-3)], dtype=float)
    ub = np.array([ymax + 10 * rng, np.log(1e9)], dtype=float)

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        loss="huber",
        f_scale=1.0,
        max_nfev=20000,
    )

    y_inf_fit, logtau_fit = res.x
    tau_fit = float(np.exp(logtau_fit))

    # Métricas.
    r = residuals(res.x)
    rmse = float(np.sqrt(np.mean(r**2)))
    ss_res = float(np.sum(r**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return float(y_inf_fit), float(tau_fit), rmse, r2, bool(res.success)


@dataclass
class CycleRow:
    ciclo: int
    inicio_off: str
    fin_off: str
    duracion_off_s: float
    inicio_on: str
    fin_on: str
    duracion_on_s: float
    periodo_desde_ciclo_anterior_s: float

    # Ajustes cerrados.
    h_static_nivel_m: float  # h'
    h_dinamico_nivel_m: float  # h_d
    tau_off_s: float
    tau_on_s: float

    # Caudal.
    C_const_ls: float
    volumen_bombeado_m3: float

    # Derivados del modelo del PDF.
    k_m2_s: float
    A_m2: float
    tau_usada_s: float

    # Calidad.
    rmse_off: float
    r2_off: float
    rmse_on: float
    r2_on: float
    ok_off: bool
    ok_on: bool


def integrate_volume(C_m3s: np.ndarray, dt_s: np.ndarray) -> float:
    dt_s = np.nan_to_num(dt_s, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(C_m3s * dt_s))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def exp_model(t: np.ndarray, y0: float, y_inf: float, tau: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return y_inf + (y0 - y_inf) * np.exp(-(t - t[0]) / tau)


def plot_cycle_h(
    cidx: int,
    out_dir: str,
    t_off: np.ndarray,
    h_off: np.ndarray,
    t_on: np.ndarray,
    h_on: np.ndarray,
    h_static_fit: float,
    tau_off: float,
    ok_off: bool,
    h_d_fit: float,
    tau_on: float,
    ok_on: bool,
    depth_like: bool,
) -> None:
    ensure_dir(out_dir)
    plt.figure()

    # OFF.
    plt.plot(t_off, h_off, label="h OFF (datos)")
    if ok_off and np.isfinite(tau_off):
        yhat_off = exp_model(t_off, float(h_off[0]), h_static_fit, tau_off)
        plt.plot(t_off, yhat_off, label="h OFF (ajuste)")

    # ON (desplazado en tiempo para que sea continuo).
    t_on_shift = t_on + (t_off[-1] if len(t_off) else 0.0)
    plt.plot(t_on_shift, h_on, label="h ON (datos)")
    if ok_on and np.isfinite(tau_on):
        yhat_on = exp_model(t_on, float(h_on[0]), h_d_fit, tau_on)
        plt.plot(t_on_shift, yhat_on, label="h ON (ajuste)")

    plt.xlabel("t (s)")
    plt.ylabel("nivel (m)" if not depth_like else "profundidad (m)")
    plt.title(f"Ciclo {cidx} - h(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cycle_{cidx:03d}_h.png"), dpi=150)
    plt.close()


def plot_cycle_q(
    cidx: int,
    out_dir: str,
    t_off: np.ndarray,
    q_off: np.ndarray,
    t_on: np.ndarray,
    q_on: np.ndarray,
) -> None:
    ensure_dir(out_dir)
    plt.figure()
    plt.plot(t_off, q_off, label="caudal OFF (L/s)")
    t_on_shift = t_on + (t_off[-1] if len(t_off) else 0.0)
    plt.plot(t_on_shift, q_on, label="caudal ON (L/s)")
    plt.xlabel("t (s)")
    plt.ylabel("caudal (L/s)")
    plt.title(f"Ciclo {cidx} - caudal(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cycle_{cidx:03d}_q.png"), dpi=150)
    plt.close()


def mean_std(series: pd.Series) -> Tuple[float, float]:
    return float(np.nanmean(series)), float(np.nanstd(series, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con columnas ts, caudal_ls, nivel_m")
    ap.add_argument("--thr_ls", type=float, default=0.05, help="Umbral L/s para ON")
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--min_off_min", type=float, default=30)
    ap.add_argument("--min_on_min", type=float, default=5)
    ap.add_argument(
        "--C_ls",
        type=float,
        default=None,
        help="Si lo pasás, usa este C constante (L/s) para todos los ciclos. Si no, usa mediana del ON de cada ciclo.",
    )
    ap.add_argument("--out_cycles_csv", required=True)
    ap.add_argument("--out_summary_csv", default="")
    ap.add_argument(
        "--out_plots_dir",
        default="",
        help="Carpeta para guardar PNGs por ciclo (si vacío, no grafica)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"ts", "caudal_ls", "nivel_m"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"Faltan columnas. Requiere {needed}")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    df["dt_s"] = df["ts"].diff().dt.total_seconds()
    df.loc[0, "dt_s"] = np.nanmedian(df["dt_s"].iloc[1:].to_numpy())

    df["nivel_f"] = robust_median_smooth(df["nivel_m"], window=args.smooth_window)
    df["pump_on"] = df["caudal_ls"] > args.thr_ls

    # Detectar profundidad.
    depth_like = detect_depth_like(df["nivel_f"].to_numpy(float), df["pump_on"].to_numpy(bool))

    # Variable física h (altura).
    df["h"] = -df["nivel_f"] if depth_like else df["nivel_f"]

    segs = detect_segments(df["pump_on"].to_numpy(bool))

    min_off_s = args.min_off_min * 60.0
    min_on_s = args.min_on_min * 60.0

    cycles: List[Tuple[int, int, int, int]] = []
    for i in range(1, len(segs)):
        s_prev, e_prev, prev_on = segs[i - 1]
        s_cur, e_cur, cur_on = segs[i]
        if (prev_on is False) and (cur_on is True):
            off_dur = float(np.nansum(df.loc[s_prev:e_prev, "dt_s"].to_numpy()))
            on_dur = float(np.nansum(df.loc[s_cur:e_cur, "dt_s"].to_numpy()))
            if off_dur >= min_off_s and on_dur >= min_on_s:
                cycles.append((s_prev, e_prev, s_cur, e_cur))

    if not cycles:
        raise SystemExit("No se encontraron ciclos OFF->ON con mínimos. Ajustá --thr_ls / --min_*.")

    rows: List[CycleRow] = []
    prev_on_start: Optional[pd.Timestamp] = None

    for cidx, (off_s, off_e, on_s, on_e) in enumerate(cycles, start=1):
        off_seg = df.loc[off_s:off_e].copy()
        on_seg = df.loc[on_s:on_e].copy()

        # Tiempos relativos.
        t_off = (off_seg["ts"] - off_seg["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
        t_on = (on_seg["ts"] - on_seg["ts"].iloc[0]).dt.total_seconds().to_numpy(float)
        h_off = off_seg["h"].to_numpy(float)
        h_on = on_seg["h"].to_numpy(float)

        # Ajuste OFF: y_inf = h' (estático), tau_off.
        h_static_fit, tau_off, rmse_off, r2_off, ok_off = fit_exp_closed_form(t_off, h_off, mode="off")

        # Ajuste ON: y_inf = h_d (dinámico), tau_on.
        h_d_fit, tau_on, rmse_on, r2_on, ok_on = fit_exp_closed_form(t_on, h_on, mode="on")

        # C constante (por ciclo o global).
        if args.C_ls is not None:
            C_ls = float(args.C_ls)
        else:
            C_ls = float(np.nanmedian(on_seg["caudal_ls"].to_numpy(float)))
        C_m3s = C_ls / 1000.0

        # k desde h_d = h' - C/k => k = C/(h' - h_d).
        delta = h_static_fit - h_d_fit
        if np.isfinite(delta) and delta > 0 and np.isfinite(C_m3s) and C_m3s > 0:
            k = float(C_m3s / delta)
        else:
            k = np.nan

        # tau usada: idealmente tau_off ~ tau_on ~ A/k.
        # Usamos la mejor disponible (prioridad: OFF si ok).
        tau_use = tau_off if ok_off and np.isfinite(tau_off) else tau_on
        A = float(tau_use * k) if (np.isfinite(tau_use) and np.isfinite(k) and k > 0) else np.nan

        # Convertir a "nivel" original.
        h_static_nivel = -h_static_fit if depth_like else h_static_fit
        h_d_nivel = -h_d_fit if depth_like else h_d_fit

        # Volumen bombeado.
        vol_m3 = integrate_volume(
            (on_seg["caudal_ls"].to_numpy(float) / 1000.0),
            on_seg["dt_s"].to_numpy(float),
        )

        # Período entre arranques.
        on_start_ts = df.loc[on_s, "ts"]
        periodo_prev = float((on_start_ts - prev_on_start).total_seconds()) if prev_on_start is not None else np.nan
        prev_on_start = on_start_ts

        if args.out_plots_dir:
            plot_cycle_h(
                cidx=cidx,
                out_dir=args.out_plots_dir,
                t_off=t_off,
                h_off=h_off,
                t_on=t_on,
                h_on=h_on,
                h_static_fit=h_static_fit,
                tau_off=tau_off,
                ok_off=ok_off,
                h_d_fit=h_d_fit,
                tau_on=tau_on,
                ok_on=ok_on,
                depth_like=depth_like,
            )

            q_off = off_seg["caudal_ls"].to_numpy(float)
            q_on = on_seg["caudal_ls"].to_numpy(float)
            plot_cycle_q(
                cidx=cidx,
                out_dir=args.out_plots_dir,
                t_off=t_off,
                q_off=q_off,
                t_on=t_on,
                q_on=q_on,
            )

        rows.append(
            CycleRow(
                ciclo=cidx,
                inicio_off=str(df.loc[off_s, "ts"]),
                fin_off=str(df.loc[off_e, "ts"]),
                duracion_off_s=float(np.nansum(off_seg["dt_s"].to_numpy())),
                inicio_on=str(df.loc[on_s, "ts"]),
                fin_on=str(df.loc[on_e, "ts"]),
                duracion_on_s=float(np.nansum(on_seg["dt_s"].to_numpy())),
                periodo_desde_ciclo_anterior_s=periodo_prev,
                h_static_nivel_m=float(h_static_nivel),
                h_dinamico_nivel_m=float(h_d_nivel),
                tau_off_s=float(tau_off),
                tau_on_s=float(tau_on),
                C_const_ls=float(C_ls),
                volumen_bombeado_m3=float(vol_m3),
                k_m2_s=float(k),
                A_m2=float(A),
                tau_usada_s=float(tau_use),
                rmse_off=float(rmse_off),
                r2_off=float(r2_off),
                rmse_on=float(rmse_on),
                r2_on=float(r2_on),
                ok_off=bool(ok_off),
                ok_on=bool(ok_on),
            )
        )

    cycles_df = pd.DataFrame([asdict(r) for r in rows])
    cycles_df.to_csv(args.out_cycles_csv, index=False, encoding="utf-8")

    if args.out_summary_csv:
        hs_mean, hs_std = mean_std(cycles_df["h_static_nivel_m"])
        hd_mean, hd_std = mean_std(cycles_df["h_dinamico_nivel_m"])
        tau_off_mean, tau_off_std = mean_std(cycles_df["tau_off_s"])
        tau_on_mean, tau_on_std = mean_std(cycles_df["tau_on_s"])
        k_mean, k_std = mean_std(cycles_df["k_m2_s"])
        A_mean, A_std = mean_std(cycles_df["A_m2"])
        C_mean, C_std = mean_std(cycles_df["C_const_ls"])
        rmse_off_mean, rmse_off_std = mean_std(cycles_df["rmse_off"])
        rmse_on_mean, rmse_on_std = mean_std(cycles_df["rmse_on"])
        r2_off_mean, r2_off_std = mean_std(cycles_df["r2_off"])
        r2_on_mean, r2_on_std = mean_std(cycles_df["r2_on"])

        summary = pd.DataFrame(
            [
                {
                    "n_ciclos": int(len(cycles_df)),
                    "nivel_es_profundidad_detectado": bool(depth_like),
                    "hs_prom": hs_mean,
                    "hs_std": hs_std,
                    "hd_prom": hd_mean,
                    "hd_std": hd_std,
                    "tau_off_prom_s": tau_off_mean,
                    "tau_off_std_s": tau_off_std,
                    "tau_on_prom_s": tau_on_mean,
                    "tau_on_std_s": tau_on_std,
                    "k_prom_m2_s": k_mean,
                    "k_std_m2_s": k_std,
                    "A_prom_m2": A_mean,
                    "A_std_m2": A_std,
                    "C_prom_ls": C_mean,
                    "C_std_ls": C_std,
                    "rmse_off_prom": rmse_off_mean,
                    "rmse_off_std": rmse_off_std,
                    "rmse_on_prom": rmse_on_mean,
                    "rmse_on_std": rmse_on_std,
                    "r2_off_prom": r2_off_mean,
                    "r2_off_std": r2_off_std,
                    "r2_on_prom": r2_on_mean,
                    "r2_on_std": r2_on_std,
                }
            ]
        )
        summary.to_csv(args.out_summary_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
