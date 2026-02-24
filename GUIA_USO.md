# Guía de uso del pipeline de procesamiento

Esta guía explica cómo ejecutar el script principal para procesar todos los CSV raw de una carpeta y generar un CSV consolidado con parámetros.

## 1) Requisitos de datos
Cada CSV de entrada debe contener estas columnas:

- `ts` (timestamp)
- `caudal_ls` (caudal en L/s)
- `nivel_m` (nivel en metros)

## 2) Ejecución básica
Desde la raíz del repo:

```bash
python main.py --input_dir /ruta/a/carpeta_raw --output_csv parametros_consolidados.csv
```

## 3) Ejecución con opciones
```bash
python main.py \
  --input_dir /ruta/a/carpeta_raw \
  --output_csv parametros_consolidados.csv \
  --errors_csv errores_procesamiento.csv \
  --smooth_window 5 \
  --min_threshold_ls 0.5 \
  --min_cycle_points 5
```

## 4) Qué genera

- `parametros_consolidados.csv`: una fila por archivo procesado con métricas operativas e hidrogeológicas.
- `errores_procesamiento.csv` (opcional): archivos que no pudieron procesarse y motivo.

## 5) Campos principales de salida

- Operacionales: `duty_cycle_pct`, `freq_cycles_day`, `avg_cycle_duration_min`, `typical_flow_ls`, `flow_stability_std`, `vol_total_m3`
- Hidrogeológicos: `h_static_mean_m`, `h_dinamico_mean_m`, `k_mean_m2_s`, `A_mean_m2`
- Clasificación: `regime_label`, `dynamic_threshold_used`
- Metadatos: `archivo_origen`, `fecha_inicio`, `fecha_fin`, `n_muestras`, `n_ciclos`

## 6) Script alternativo de periodos
Si necesitas agregación por periodos fijos:

```bash
python cycle_analysis.py --csv entrada.csv --out_periods_csv periodos.csv --out_summary_csv resumen.csv
```
