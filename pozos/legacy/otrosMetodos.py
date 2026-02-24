import pandas as pd

def promedio_caudal(df, intervalos):

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)

    intervalos = intervalos.copy()
    intervalos["inicio"] = pd.to_datetime(intervalos["inicio"]).dt.tz_localize(None)
    intervalos["fin"] = pd.to_datetime(intervalos["fin"]).dt.tz_localize(None)

    mask_total = pd.Series(False, index=df.index)
    for _, row in intervalos.iterrows():
        mask_total |= (df["ts"] >= row["inicio"]) & (df["ts"] <= row["fin"])

    prom = df.loc[mask_total, "caudal_ls"].mean()
    print("Promedio caudal (solo periodos con caudal != 0):", prom)
    return prom



import matplotlib.pyplot as plt

def graficar_caudal_suavizado(df, ventana=25):
    caudal_suave = df["caudal_ls"].rolling(ventana, center=True).mean()

    plt.figure()
    plt.plot(df["ts"], df["caudal_ls"], alpha=0.35, label="Original")
    plt.plot(df["ts"], caudal_suave, linewidth=2, label=f"Media móvil ({ventana})")
    plt.title("Caudal suavizado")
    plt.xlabel("Tiempo")
    plt.ylabel("Caudal (L/s)")
    plt.legend()
    plt.show()

    return caudal_suave