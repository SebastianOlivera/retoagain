# Guía de uso del pipeline de procesamiento

Este pipeline calcula por **pozo** y por **período o ciclo** las métricas clave:

1. `h_static_nivel_m` (hs)
2. `h_dinamico_nivel_m` (hd)
3. `tau_off_s`
4. `tau_on_s`
5. `C_const_ls`
6. métricas operativas: `frecuencia_encendido_por_dia` y `tiempo_on_prom_s`

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
  --min_threshold_ls 0.05 \
  --smooth_window 1
```

## 3) Ejecución por ciclo

```bash
python main.py \
  --input_dir /ruta/a/raw_csv \
  --output_csv metricas_ciclo.csv \
  --aggregate_mode cycle \
  --min_threshold_ls 0.05
```

## 4) Cómo se calculan
- Segmentación ON/OFF por estados contiguos.
- Ajuste por segmento con modelo exponencial:
  `h(t)=h_inf + (h0-h_inf)*exp(-t/tau)`.
- OFF: `h_inf -> hs`, `tau -> tau_off_s`.
- ON: `h_inf -> hd`, `tau -> tau_on_s`, `C_const_ls -> mediana(caudal)` del segmento ON.
- Si el ajuste no es confiable (segmento corto/plano/no converge):
  - `h_inf` se estima robustamente con mediana del tramo final,
  - `tau` queda `NaN`,
  - se reporta `ok_fit=False` a nivel segmento (resumido en `ok_on`, `ok_off`).

## 5) Campos de salida por período
- `device_id`, `periodo`, `inicio`, `fin`, `n_on`
- `h_static_nivel_m`, `h_dinamico_nivel_m`, `tau_off_s`, `tau_on_s`, `C_const_ls`
- `frecuencia_encendido_por_dia`, `tiempo_on_prom_s`
- opcionales de dispersión/calidad: `hs_std`, `hd_std`, `tau_off_std`, `tau_on_std`, `C_std`, `rmse_*`, `r2_*`, `ok_*`

## 6) Archivos generados
- `output_csv`: métricas consolidadas.
- `errors_csv` (si aplica): archivos no procesados + motivo.


## 7) Precisión numérica
Todas las métricas numéricas de salida se limitan a **3 decimales**.
