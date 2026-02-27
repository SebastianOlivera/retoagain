# Pipeline de análisis de pozos

## Estructura

- `pozos/core/profiler.py`: lógica principal de extracción de ciclos y cálculo de parámetros.
- `pozos/pipeline/batch_processor.py`: procesamiento por carpeta (`input_dir`) y consolidación de CSV.
- `pozos/cli.py`: punto de entrada CLI.
- `pozos/legacy/`: scripts antiguos preservados como referencia.
- `main.py`: wrapper para ejecutar la CLI.

## Unidades (SI)

Todo el pipeline opera en SI para caudal/volumen:

- `L -> m3`
- `L/s -> m3/s`
- `L/min -> m3/s`
- `L/h -> m3/s`

Columnas de entrada soportadas para caudal (se normalizan a `caudal_m3s`):

- `caudal_m3s`
- `caudal_ls`
- `caudal_lmin`
- `caudal_lh`

## Ejecución

```bash
python main.py --input_dir /ruta/a/raw_csv --output_csv parametros_consolidados.csv
```

Opcionales:

```bash
python main.py \
  --input_dir /ruta/a/raw_csv \
  --output_csv parametros_consolidados.csv \
  --errors_csv errores_procesamiento.csv \
  --smooth_window 5 \
  --min_threshold_m3s 0.0005 \
  --min_cycle_points 5
```

## CSV de salida

El consolidado incluye métricas operativas e hidrogeológicas por archivo:

- `duty_cycle_pct`, `freq_cycles_day`, `avg_cycle_duration_min`
- `typical_flow_m3s`, `flow_stability_m3s_std`, `vol_total_m3`
- `h_static_mean_m`, `h_dinamico_mean_m`, `k_mean_m2_s`
- `tiempo_entre_encendidos_mean`, `tiempo_entre_encendidos_std`, `tiempo_entre_encendidos_n`
- `k_m2_s_mean`, `k_m2_s_std`, `k_m2_s_n`
- `regime_label`, `dynamic_threshold_m3s_used`
- metadatos: `pozo_id`, `pozo_nombre`, `archivo_origen`, `fecha_inicio`, `fecha_fin`, `n_muestras`, `n_ciclos`
