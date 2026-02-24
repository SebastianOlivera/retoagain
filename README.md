# Pipeline de análisis de pozos

## Estructura

- `pozos/core/profiler.py`: núcleo de extracción de ciclos y parámetros hidrogeológicos.
- `pozos/pipeline/batch_processor.py`: procesamiento batch de una carpeta de CSV raw.
- `pozos/analysis/`: utilidades de análisis complementarias unificadas:
  - `basic_metrics.py`: intervalos ON/OFF, promedios y ratio encendido.
  - `fitting.py`: ajuste de parámetros `h_s`, `h_d`, `k`, `A`, `tau` por ciclo.
  - `periodic.py`: métricas agregadas por periodos temporales.
  - `visualization.py`: funciones de visualización de caudal.
- `pozos/cli.py`: punto de entrada del pipeline.
- Wrappers de compatibilidad en raíz (`main.py`, `batch_processor.py`, `profiler.py`, `cycle_analysis.py`, etc.).

## Ejecución batch

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

## Parámetros calculados en CSV final

- Operacionales: `duty_cycle_pct`, `freq_cycles_day`, `avg_cycle_duration_min`, `typical_flow_ls`, `flow_stability_std`, `vol_total_m3`
- Hidrogeológicos: `h_static_mean_m`, `h_dinamico_mean_m`, `k_mean_m2_s`, `A_mean_m2`
- Clasificación: `regime_label`, `dynamic_threshold_used`
- Metadatos: `archivo_origen`, `fecha_inicio`, `fecha_fin`, `n_muestras`, `n_ciclos`


## Guía de uso

Ver `GUIA_USO.md` para ejemplos de ejecución paso a paso.
