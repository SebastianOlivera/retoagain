import os
import glob
import pandas as pd
from profiler import WellProfiler

# --- CONFIGURACIÓN ---
CARPETA_DATOS = "../../Datos1/"
ARCHIVO_SALIDA_EXCEL = "Resumen_Clasificacion_Pozos.xlsx"
ARCHIVO_SALIDA_CSV = "Dataset_Features_Clustering.csv"
VENTANA_DIAS = 2
SMOOTH_WINDOW = 5


def procesar_pozo_en_ventanas(ruta_archivo):
    """
    Carga un CSV de un pozo y calcula las métricas en ventanas de VENTANA_DIAS días.
    Retorna una lista de dicts, uno por ventana.
    """
    nombre_archivo = os.path.basename(ruta_archivo)
    resultados = []

    df_full = pd.read_csv(ruta_archivo)
    df_full['ts'] = pd.to_datetime(df_full['ts'])
    df_full = df_full.sort_values('ts').reset_index(drop=True)

    if df_full.empty:
        return resultados

    # Umbral dinámico calculado sobre el MES COMPLETO para consistencia entre ventanas
    q_max = df_full['caudal_ls'].max()
    threshold = max(0.5, q_max * 0.05)

    # Generar cortes de ventanas de 2 días desde el inicio del mes
    inicio = df_full['ts'].min().normalize()
    cortes = pd.date_range(start=inicio, periods=16, freq=f'{VENTANA_DIAS}D')

    for num_ventana, (t_inicio, t_fin) in enumerate(zip(cortes[:-1], cortes[1:]), start=1):
        df_ventana = df_full[(df_full['ts'] >= t_inicio) & (df_full['ts'] < t_fin)]

        if len(df_ventana) < 2:
            continue

        pozo = WellProfiler.from_dataframe(df_ventana.reset_index(drop=True), threshold, smooth_window=SMOOTH_WINDOW)
        pozo.extract_cycles()
        pozo.compute_global_metrics()
        pozo.classify_regime()

        features = pozo.get_features()
        features['archivo_origen'] = nombre_archivo
        features['ventana_num'] = num_ventana
        features['ventana_inicio'] = t_inicio.strftime('%Y-%m-%d')
        features['ventana_fin'] = (t_fin - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        resultados.append(features)

    return resultados


def procesar_lote_completo():
    archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
    total_archivos = len(archivos)

    if total_archivos == 0:
        print(f"❌ Error: No se encontraron archivos .csv en la ruta: {CARPETA_DATOS}")
        return

    print(f"📂 Se encontraron {total_archivos} pozos para procesar.")
    print(f"🚀 Iniciando procesamiento en ventanas de {VENTANA_DIAS} días...\n")

    resultados_lista = []
    errores_lista = []

    for i, ruta_archivo in enumerate(archivos):
        nombre_archivo = os.path.basename(ruta_archivo)
        print(f"[{i+1}/{total_archivos}] Procesando: {nombre_archivo}...", end=" ")

        try:
            ventanas = procesar_pozo_en_ventanas(ruta_archivo)

            if ventanas:
                resultados_lista.extend(ventanas)
                print(f"✅ OK ({len(ventanas)} ventanas)")
            else:
                print("⚠️ Saltado (datos insuficientes)")
                errores_lista.append(nombre_archivo)

        except Exception as e:
            print(f"❌ Error: {e}")
            errores_lista.append(f"{nombre_archivo} ({str(e)})")

    # Consolidación
    print("\n" + "="*50)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*50)

    if resultados_lista:
        df_final = pd.DataFrame(resultados_lista)

        cols_first = [
            'archivo_origen', 'ventana_num', 'ventana_inicio', 'ventana_fin',
            'regime_label', 'duty_cycle_pct', 'freq_cycles_day',
            'vol_total_m3', 'h_static_mean_m', 'h_dinamico_mean_m', 'k_mean_m2_s', 'A_mean_m2',
        ]
        cols = cols_first + [c for c in df_final.columns if c not in cols_first]
        df_final = df_final[cols]

        df_final.to_excel(ARCHIVO_SALIDA_EXCEL, index=False)
        df_final.to_csv(ARCHIVO_SALIDA_CSV, index=False)

        pozos_procesados = df_final['archivo_origen'].nunique()
        print(f"✨ Éxito: {pozos_procesados} pozos procesados → {len(df_final)} ventanas totales.")
        print(f"📊 Tabla Excel guardada en: {ARCHIVO_SALIDA_EXCEL}")
        print(f"🤖 Dataset ML guardado en: {ARCHIVO_SALIDA_CSV}")

        print("\nDistribución de Regímenes detectados:")
        print(df_final['regime_label'].value_counts())

    else:
        print("❌ No se generaron resultados válidos.")

    if errores_lista:
        print(f"\n⚠️ Hubo errores en {len(errores_lista)} archivos (verificar logs).")


if __name__ == "__main__":
    procesar_lote_completo()
