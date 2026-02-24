# Guía de uso del pipeline de procesamiento

Este pipeline calcula por **pozo** y por **período o ciclo** las métricas clave con **tau global compartido**:

1. `h_static_nivel_median` (hs mediana)
2. `h_dinamico_nivel_median` (hd mediana)
3. `tau_s_median`
4. `C_const_ls_median`
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
python ejecutar_metricas_periodicas.py --csv entrada.csv --out_csv salida_periodos.csv --days_per_period 2
```

## 3) Ejecución por ciclo

```bash
(El flujo actual está enfocado en métricas por período).
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
- `h_static_nivel_*`, `h_dinamico_nivel_*`, `tau_s_*`, `C_const_ls_*`
- `frecuencia_encendido_por_dia`, `tiempo_on_prom_s`
- calidad global de ajuste: `ok_fit_global`, `rmse_global`, `r2_global`

## 6) Precisión numérica
Todas las métricas numéricas de salida se limitan a **3 decimales**.


## 7) Estructura de estadísticas por métrica
Para cada métrica agregada se exportan 4 columnas con sufijos explícitos:
- `<variable>_median`
- `<variable>_mean`
- `<variable>_std` (desviación estándar muestral, ddof=1; si n<2 => 0.0)
- `<variable>_n` (cantidad de valores válidos)

Aplicado en:
- `h_static_nivel_median`, `h_static_nivel_mean`, `h_static_nivel_std`, `h_static_nivel_n`
- `h_dinamico_nivel_median`, `h_dinamico_nivel_mean`, `h_dinamico_nivel_std`, `h_dinamico_nivel_n`
- `tau_s_median`, `tau_s_mean`, `tau_s_std`, `tau_s_n`
- `C_const_ls_median`, `C_const_ls_mean`, `C_const_ls_std`, `C_const_ls_n`



Nomenclatura estricta de agregados (sin columnas ambiguas):
- `h_static_nivel_median`, `h_static_nivel_mean`, `h_static_nivel_std`, `h_static_nivel_n`
- `h_dinamico_nivel_median`, `h_dinamico_nivel_mean`, `h_dinamico_nivel_std`, `h_dinamico_nivel_n`
- `tau_s_median`, `tau_s_mean`, `tau_s_std`, `tau_s_n`
- `C_const_ls_median`, `C_const_ls_mean`, `C_const_ls_std`, `C_const_ls_n`

No se exportan columnas ambiguas como `h_static_nivel_m`, `h_dinamico_nivel_m`, `tau_s` o `C_const_ls`.

## 8) Script de prueba rápida por período
```bash
python ejecutar_metricas_periodicas.py --csv entrada.csv --out_csv salida_periodos.csv --days_per_period 2
```


## 9) Nota sobre tau enorme
Si la curva de nivel del segmento ON es casi plana (baja identificabilidad), el ajuste puede devolver un `tau` muy grande. Esto es esperado físicamente y no se recorta de forma artificial.


### Procesar carpeta completa de CSV
```bash
python ejecutar_metricas_periodicas.py --input_dir /ruta/carpeta_csv --out_csv salida_periodos.csv --days_per_period 2
```
