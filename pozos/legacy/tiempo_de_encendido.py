import pandas as pd

def promedio_bomba_encendida(path, columna_velocidad, columna_timestamp="timestamp"):
    # Detectar formato
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Convertir timestamp a datetime
    df[columna_timestamp] = pd.to_datetime(df[columna_timestamp])

    # Ordenar por tiempo
    df = df.sort_values(columna_timestamp)

    primer_timestamp = df[columna_timestamp].iloc[0]
    ultimo_timestamp = df[columna_timestamp].iloc[-1]

    total_datos = len(df)
    encendida = (df[columna_velocidad] > 0).sum()

    promedio = encendida / total_datos

    print(
        f"Promedio de tiempo encendido entre {primer_timestamp} y {ultimo_timestamp} es de {promedio*100:.2f}%"
    )

    return promedio
