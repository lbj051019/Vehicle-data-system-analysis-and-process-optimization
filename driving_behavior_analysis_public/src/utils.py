from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def to_datetime_sorted(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    out = out.sort_values(column).reset_index(drop=True)
    return out


def pretty_counts(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    return "\n".join(f"{idx}: {val}" for idx, val in counts.items())
