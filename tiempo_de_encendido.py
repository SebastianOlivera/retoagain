from pathlib import Path

from pozos.analysis.basic_metrics import load_timeseries, on_ratio


def promedio_bomba_encendida(path, columna_velocidad, columna_timestamp="timestamp"):
    df = load_timeseries(Path(path), ts_col=columna_timestamp)
    ratio = on_ratio(df, flow_col=columna_velocidad, threshold=0.0)
    return ratio
