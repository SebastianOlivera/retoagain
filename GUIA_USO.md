# Guía de uso del pipeline de procesamiento

Este pipeline calcula por **pozo** y por **período o ciclo** las métricas clave con **tau global compartido**:

1. `h_static_nivel_m` (hs)
2. `h_dinamico_nivel_m` (hd)
3. `tau_s` (único global)
4. `C_const_ls`
5. `frecuencia_encendido_por_dia`
6. `tiempo_on_prom_s`

## 1) Requisitos de entrada
Cada CSV debe incluir:

- `ts` (timestamp parseable)
- `nivel_m` (float)
- `caudal_ls` (float)

Opcional:
- `estado_bomba` (0/1). Si no viene, se infiere por `caudal_ls > umbral_q`.

## 2) Ejecución por período (recomendado)

```bash
python main.py \
  --input_dir /ruta/a/raw_csv \
  --output_csv metricas_periodo.csv \
  --aggregate_mode period \
  --period_days 2 \
  --min_threshold_ls 0.05
```

## 3) Ejecución por ciclo

```bash
python main.py \
  --input_dir /ruta/a/raw_csv \
  --output_csv metricas_ciclo.csv \
  --aggregate_mode cycle \
  --min_threshold_ls 0.05
```

## 4) Ajuste global de tau
Se ajusta de forma conjunta:

`h_hat_j(t)=h_inf_j + (h0_j-h_inf_j)*exp(-t/tau_s)`

- Cada segmento `j` tiene su propio `h_inf_j`.
- Todos los segmentos comparten un único `tau_s`.
- Solo entran al fit segmentos con filtros de robustez:
  - `duracion_seg_s >= min_seg_dur_s`
  - `n_puntos_seg >= min_segment_points`
  - `delta_h >= min_delta_h`

Parámetros útiles del CLI:
- `--min_seg_dur_s`
- `--min_segment_points`
- `--min_delta_h`
- `--tau_min_s`
- `--tau_max_s`

## 5) Campos de salida por período
- `device_id`, `periodo`, `inicio`, `fin`, `n_on`
- `h_static_nivel_m`, `h_dinamico_nivel_m`, `tau_s`, `C_const_ls`
- `frecuencia_encendido_por_dia`, `tiempo_on_prom_s`
- calidad global de ajuste: `ok_fit_global`, `rmse_global`, `r2_global`

## 6) Precisión numérica
Todas las métricas numéricas de salida se limitan a **3 decimales**.


## 7) Estructura de estadísticas por métrica
Para cada métrica `X` en el CSV se exportan 4 columnas en este orden:
- `X` (mediana)
- `X_mean` (media)
- `X_std` (desviación estándar muestral, ddof=1; si n<2 => 0.0)
- `X_n` (cantidad de valores válidos)

Aplicado en:
- `h_static_nivel_m`
- `h_dinamico_nivel_m`
- `tau_s`
- `C_const_ls`
