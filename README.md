# Flujo activo del proyecto

Este repositorio quedó **limpio** y centrado en un único flujo vigente:

- `run_periodic_pipeline.py`: script principal para generar métricas por períodos (2 días por defecto).
- `pozos/analysis/fitting.py`: fitting físico por segmento ON y agregación por período.
- `pozos/analysis/periodic.py`: construcción de la tabla final por período con columnas objetivo.

## Uso

```bash
python run_periodic_pipeline.py --csv entrada.csv --out_csv salida_periodos.csv --days_per_period 2
```

También podés usar:

```bash
python main.py --csv entrada.csv --out_csv salida_periodos.csv --days_per_period 2
```
