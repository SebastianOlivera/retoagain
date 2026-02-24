import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def ajustar_hd_k_A_desde_df(
    df: pd.DataFrame,
    ciclo: int = 1,
    col_timestamp: str = "Timestamp",
    col_caudal: str = "velocidad de salida de agua",
    col_nivel: str = "nivel de agua",
):
    """
    - Detecta ciclos con caudal>0 (bomba encendida)
    - Toma el ciclo N (1 = primer ciclo)
    - Ajusta por mínimos cuadrados (ajusta h_s, k, tau):
        h_d = h_s + C/k
        h(t)= h_d - (h_d - h0)*exp(-t/tau)
    - Deriva:
        hd = h_d
        A  = k*tau

    Devuelve dict con h_s, hd, k, A, tau, rmse, periodo y C_prom.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas DataFrame")

    for c in (col_timestamp, col_caudal, col_nivel):
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en el DataFrame")

    df = df.copy()

    # timestamp y orden
    df[col_timestamp] = pd.to_datetime(df[col_timestamp])
    df = df.sort_values(col_timestamp).reset_index(drop=True)

    # bomba encendida
    is_on = df[col_caudal] > 0

    # detectar arranques (False -> True)
    start_cycle = is_on & (~is_on.shift(1, fill_value=False))

    # id de ciclo (1,2,3...) solo para estados ON
    cycle_id = start_cycle.cumsum()
    df["__is_on"] = is_on
    df["__cycle_id"] = cycle_id

    on_df = df[df["__is_on"]].copy()
    if on_df.empty:
        raise ValueError("No hay registros con bomba encendida (caudal > 0).")

    # elegir ciclo
    ciclos_disponibles = sorted(on_df["__cycle_id"].unique())
    if ciclo not in ciclos_disponibles:
        raise ValueError(f"Ciclo {ciclo} no existe. Ciclos disponibles: {ciclos_disponibles}")

    seg = on_df[on_df["__cycle_id"] == ciclo].copy().reset_index(drop=True)

    # construir t desde inicio del ciclo (en segundos)
    t = (seg[col_timestamp] - seg[col_timestamp].iloc[0]).dt.total_seconds().to_numpy()
    h = seg[col_nivel].to_numpy(dtype=float)

    if len(h) < 3:
        raise ValueError("Muy pocos puntos en el ciclo para ajustar.")

    # h0 (nivel inicial del ciclo)
    h0 = float(h[0])

    # C constante aproximado: promedio durante el ciclo
    C = float(seg[col_caudal].mean())

    # modelo: ajustamos h_s (h'), k y tau
    def modelo(t, h_s, k, tau):
        h_d = h_s + C / k
        return h_d - (h_d - h0) * np.exp(-t / tau)

    # iniciales (sin hardcode fijo)
    h_s0 = float(np.median(h))                # arranque razonable
    k0 = max(abs(C) / max(abs(h_s0 - h[-1]), 1e-6), 1e-3)  # positivo
    tau0 = max((t[-1] - t[0]) / 5, 1e-6)

    eps = 1e-9
    (h_s_fit, k_fit, tau_fit), _ = curve_fit(
        modelo, t, h,
        p0=[h_s0, k0, tau0],
        bounds=([-np.inf, eps, eps], [np.inf, np.inf, np.inf]),
        maxfev=50000
    )

    hd_fit = float(h_s_fit + C / k_fit)
    A_fit = float(k_fit * tau_fit)

    resid = h - modelo(t, h_s_fit, k_fit, tau_fit)
    rmse = float(np.sqrt(np.mean(resid**2)))
    
    
    # --- curva ajustada para graficar ---
    h_fit = modelo(t, h_s_fit, k_fit, tau_fit)

    plt.figure()
    plt.plot(seg[col_timestamp], h, label="Nivel real")
    plt.plot(seg[col_timestamp], h_fit, label="Ajuste")
    plt.title(f"Ajuste nivel - ciclo {ciclo}")
    plt.xlabel("Tiempo")
    plt.ylabel("Nivel")
    plt.legend()
    plt.show()

    return {
        "ciclo": ciclo,
        "periodo_inicio": seg[col_timestamp].iloc[0],
        "periodo_fin": seg[col_timestamp].iloc[-1],
        "C_prom": C,
        "h_s": float(h_s_fit),
        "hd": hd_fit,
        "k": float(k_fit),
        "A": A_fit,
        "tau": float(tau_fit),
        "rmse": rmse,
    }