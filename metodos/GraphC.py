import pandas as pd
import matplotlib.pyplot as plt

def graficar_caudal_y_detectar_intervalos(df: pd.DataFrame):
    # Gráfica de caudal
    plt.figure()
    plt.plot(df["ts"], df["caudal_ls"])
    plt.title("Caudal en el tiempo")
    plt.xlabel("Tiempo")
    plt.ylabel("Caudal (L/s)")
    plt.show()

    activo = df["caudal_ls"] != 0
    cambios = activo.astype(int).diff()

    inicios = df.loc[cambios == 1, "ts"]
    finales = df.loc[cambios == -1, "ts"]

    if activo.iloc[0]:
        inicios = pd.concat([pd.Series([df["ts"].iloc[0]]), inicios], ignore_index=True)

    if activo.iloc[-1]:
        finales = pd.concat([finales, pd.Series([df["ts"].iloc[-1]])], ignore_index=True)

    intervalos = pd.DataFrame({"inicio": inicios.values, "fin": finales.values})

    print("Cantidad de periodos con caudal distinto de 0:", len(intervalos))
    return intervalos
