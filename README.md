# Flujo activo del proyecto

Nombres de archivos alineados al propósito:

- `ejecutar_metricas_periodicas.py`: script principal para generar métricas por período (2 días por defecto).
- `pozos/analysis/ajuste_fisico.py`: ajuste físico por segmentos ON y agregación de calidad de ajuste.
- `pozos/analysis/metricas_por_periodo.py`: construcción de la tabla final por períodos.
- `pozos/analysis/metricas_basicas.py`: utilidades básicas de series temporales.
- `pozos/analysis/visualizacion.py`: funciones de visualización.
- `pozos/cli_metricas.py`: entrypoint CLI equivalente.

## Uso

```bash
python ejecutar_metricas_periodicas.py --csv entrada.csv --out_csv salida_periodos.csv --out_cycles_csv salida_ciclos.csv --days_per_period 2
```

También podés usar:

```bash
python main.py --csv entrada.csv --out_csv salida_periodos.csv --days_per_period 2
```


### Procesar carpeta completa de CSV
```bash
python ejecutar_metricas_periodicas.py --input_dir /ruta/carpeta_csv --out_csv salida_periodos.csv --days_per_period 2
```


Incluye agregados adicionales por período: `k_median|mean|std|n` y `tiempo_entre_encendidos_median|mean|std|n`.

También podés pedir detalle por ciclo con `--out_cycles_csv`.
