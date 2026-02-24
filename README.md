# Pipeline de análisis de pozos

## Estructura

- `pozos/core/profiler.py`: lógica principal de extracción de ciclos y cálculo de parámetros.
- `pozos/pipeline/batch_processor.py`: procesamiento por carpeta (`input_dir`) y consolidación de CSV.
- `pozos/cli.py`: punto de entrada CLI.
- `pozos/legacy/`: scripts antiguos preservados como referencia.
- `main.py`: wrapper para ejecutar la CLI.

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
  --min_threshold_ls 0.5 \
  --min_cycle_points 5
```

## CSV de salida

El consolidado incluye métricas operativas e hidrogeológicas por archivo:

- `duty_cycle_pct`, `freq_cycles_day`, `avg_cycle_duration_min`
- `typical_flow_ls`, `flow_stability_std`, `vol_total_m3`
- `h_static_mean_m`, `h_dinamico_mean_m`, `k_mean_m2_s`, `A_mean_m2`
- `regime_label`, `dynamic_threshold_used`
- metadatos: `archivo_origen`, `fecha_inicio`, `fecha_fin`, `n_muestras`, `n_ciclos`
