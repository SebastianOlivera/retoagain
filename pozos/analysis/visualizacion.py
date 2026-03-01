from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_flow(df: pd.DataFrame, ts_col: str = "ts", flow_col: str = "caudal_m3s") -> None:
    plt.figure()
    plt.plot(df[ts_col], df[flow_col])
    plt.title("Caudal en el tiempo")
    plt.xlabel("Tiempo")
    plt.ylabel("Caudal (m³/s)")
    plt.tight_layout()
    plt.show()


def smooth_flow(df: pd.DataFrame, flow_col: str = "caudal_m3s", window: int = 25) -> pd.Series:
    return df[flow_col].rolling(window, center=True).mean()


def plot_smoothed_flow(df: pd.DataFrame, ts_col: str = "ts", flow_col: str = "caudal_m3s", window: int = 25) -> pd.Series:
    smoothed = smooth_flow(df, flow_col=flow_col, window=window)
    plt.figure()
    plt.plot(df[ts_col], df[flow_col], alpha=0.35, label="Original")
    plt.plot(df[ts_col], smoothed, linewidth=2, label=f"Media móvil ({window})")
    plt.title("Caudal suavizado")
    plt.xlabel("Tiempo")
    plt.ylabel("Caudal (m³/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return smoothed
