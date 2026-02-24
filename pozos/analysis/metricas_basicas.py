from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"ts", "caudal_ls"}


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")


def load_timeseries(path: str | Path, ts_col: str = "ts") -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    _validate_columns(df, {ts_col})
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return df


def detect_on_intervals(df: pd.DataFrame, ts_col: str = "ts", flow_col: str = "caudal_ls", threshold: float = 0.0) -> pd.DataFrame:
    _validate_columns(df, {ts_col, flow_col})
    dfx = df.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce")
    dfx = dfx.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    active = dfx[flow_col] > threshold
    changes = active.astype(int).diff()

    starts = dfx.loc[changes == 1, ts_col]
    ends = dfx.loc[changes == -1, ts_col]

    if not dfx.empty and bool(active.iloc[0]):
        starts = pd.concat([pd.Series([dfx[ts_col].iloc[0]]), starts], ignore_index=True)
    if not dfx.empty and bool(active.iloc[-1]):
        ends = pd.concat([ends, pd.Series([dfx[ts_col].iloc[-1]])], ignore_index=True)

    return pd.DataFrame({"inicio": starts.values, "fin": ends.values})


def mean_flow_on_intervals(df: pd.DataFrame, intervals: pd.DataFrame, ts_col: str = "ts", flow_col: str = "caudal_ls") -> float:
    _validate_columns(df, {ts_col, flow_col})
    if intervals.empty:
        return float("nan")

    dfx = df.copy()
    dfx[ts_col] = pd.to_datetime(dfx[ts_col], errors="coerce").dt.tz_localize(None)
    dfx = dfx.dropna(subset=[ts_col])

    iv = intervals.copy()
    iv["inicio"] = pd.to_datetime(iv["inicio"], errors="coerce").dt.tz_localize(None)
    iv["fin"] = pd.to_datetime(iv["fin"], errors="coerce").dt.tz_localize(None)

    mask = pd.Series(False, index=dfx.index)
    for _, row in iv.iterrows():
        mask |= (dfx[ts_col] >= row["inicio"]) & (dfx[ts_col] <= row["fin"])

    return float(dfx.loc[mask, flow_col].mean())


def on_ratio(df: pd.DataFrame, flow_col: str = "caudal_ls", threshold: float = 0.0) -> float:
    _validate_columns(df, {flow_col})
    total = len(df)
    if total == 0:
        return float("nan")
    on = int((df[flow_col] > threshold).sum())
    return on / total
