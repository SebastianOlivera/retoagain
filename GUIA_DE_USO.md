# Guía de uso del pipeline de procesamiento

Este pipeline calcula por **pozo** y por **período o ciclo** las métricas clave con **tau global compartido**:

1. `h_static_nivel_median` (hs mediana)
2. `h_dinamico_nivel_median` (hd mediana)
3. `tau_s_median`
4. `C_const_m3s_median`
5. `frecuencia_encendido_por_dia`
6. `tiempo_on_prom_s`
7. `tiempo_entre_encendidos_*`
8. `k_*`

## 1) Requisitos de entrada
Cada CSV debe incluir:

- `ts` (timestamp parseable)
- `nivel_m` (float)
- `caudal_m3s` (float)

Opcional:
- `estado_bomba` (0/1). Si no viene, se infiere por `caudal_m3s > umbral_q`.
- `nombre_pozo` (si existe, se exporta justo después de `device_id`).
- Si faltan `device_id`/`nombre_pozo` (o vienen como placeholders), se infieren desde el nombre del archivo usando el formato `NOMBRE_POZO_DEVICE_ID.csv` (separando por el último `_`).

## 2) Ejecución por período (recomendado)

```bash
python ejecutar_metricas_periodicas.py --csv entrada.csv --out_csv salida_periodos.csv --out_cycles_csv salida_ciclos.csv --days_per_period 2
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
- `device_id`, `nombre_pozo`, `periodo`, `inicio`, `fin`, `n_on`
- `h_static_nivel_*`, `h_dinamico_nivel_*`, `tau_s_*`, `C_const_m3s_*`
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
- `C_const_m3s_median`, `C_const_m3s_mean`, `C_const_m3s_std`, `C_const_m3s_n`



Nomenclatura estricta de agregados (sin columnas ambiguas):
- `h_static_nivel_median`, `h_static_nivel_mean`, `h_static_nivel_std`, `h_static_nivel_n`
- `h_dinamico_nivel_median`, `h_dinamico_nivel_mean`, `h_dinamico_nivel_std`, `h_dinamico_nivel_n`
- `tau_s_median`, `tau_s_mean`, `tau_s_std`, `tau_s_n`
- `C_const_m3s_median`, `C_const_m3s_mean`, `C_const_m3s_std`, `C_const_m3s_n`

No se exportan columnas ambiguas como `h_static_nivel_m`, `h_dinamico_nivel_m`, `tau_s` o `C_const_m3s`.

## 8) Script de prueba rápida por período
```bash
python ejecutar_metricas_periodicas.py --csv entrada.csv --out_csv salida_periodos.csv --out_cycles_csv salida_ciclos.csv --days_per_period 2
```


## 9) Nota sobre tau enorme
Si la curva de nivel del segmento ON es casi plana (baja identificabilidad), el ajuste puede devolver un `tau` muy grande. Esto es esperado físicamente y no se recorta de forma artificial.


### Procesar carpeta completa de CSV
```bash
python ejecutar_metricas_periodicas.py --input_dir /ruta/carpeta_csv --out_csv salida_periodos.csv --out_cycles_csv salida_ciclos.csv --days_per_period 2
```


## 10) Nuevas métricas físicas
- `tiempo_entre_encendidos_s` se calcula por ciclo como: inicio_on(i+1) - fin_on(i).
- Si el valor resulta negativo, se marca como inválido (`NaN`) y se deja flag de error por ciclo.
- `k` por ciclo se calcula como `k = C / (hd_fit - h_static)` con validaciones:
  - `abs(hd_fit - h_static) >= 1e-6`
  - `C > 0`
  - `ok_fit=True`
- Si el denominador es casi cero o faltan datos, `k` queda `NaN`.
- En salida por período se agregan `k_median|mean|std|n` y `tiempo_entre_encendidos_median|mean|std|n`.


Salida por ciclo (`--out_cycles_csv`) incluye, por cada ON: `tau_fit`, `hd_fit`, `h_static`, `k`, `ok_k`, `tiempo_entre_encendidos_s` y flags de validación.
