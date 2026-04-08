from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, load_csv, save_csv, to_datetime_sorted


def _merge_with_offset(can_df: pd.DataFrame, gps_df: pd.DataFrame, offset_seconds: float, tolerance_ms: int) -> pd.DataFrame:
    shifted_gps = gps_df.copy()
    shifted_gps["timestamp"] = shifted_gps["timestamp"] - pd.Timedelta(seconds=float(offset_seconds))
    merged = pd.merge_asof(
        can_df,
        shifted_gps,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=tolerance_ms),
    )
    return merged


def find_best_offset(
    can_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    search_range: Tuple[float, float] = (-60.0, 60.0),
    step: float = 0.5,
    tolerance_ms: int = 50,
) -> tuple[float, pd.DataFrame]:
    offsets = np.arange(search_range[0], search_range[1] + step, step)
    best_offset = 0.0
    best_merged = None
    best_matches = -1

    for offset in offsets:
        merged = _merge_with_offset(can_df, gps_df, offset, tolerance_ms)
        matches = int(merged.dropna().shape[0])
        if matches > best_matches:
            best_matches = matches
            best_offset = float(offset)
            best_merged = merged

    if best_merged is None:
        raise RuntimeError("Unable to merge CAN and GPS data.")
    return best_offset, best_merged


def merge_datasets(
    can_path: str | Path,
    gps_path: str | Path,
    output_path: str | Path = "outputs/merged_data.csv",
    search_offset: bool = True,
    tolerance_ms: int = 50,
) -> pd.DataFrame:
    can_df = to_datetime_sorted(load_csv(can_path))
    gps_df = to_datetime_sorted(load_csv(gps_path))

    if search_offset:
        _, merged = find_best_offset(can_df, gps_df, tolerance_ms=tolerance_ms)
    else:
        merged = _merge_with_offset(can_df, gps_df, 0.0, tolerance_ms)

    merged = merged.dropna().reset_index(drop=True)
    save_csv(merged, output_path)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CAN and GPS data by timestamp.")
    parser.add_argument("--can", required=True, help="Path to CAN CSV")
    parser.add_argument("--gps", required=True, help="Path to GPS CSV")
    parser.add_argument("--output", default="outputs/merged_data.csv", help="Output CSV path")
    parser.add_argument("--no-offset-search", action="store_true", help="Disable offset search")
    args = parser.parse_args()

    ensure_dir(Path(args.output).parent)
    result = merge_datasets(args.can, args.gps, args.output, search_offset=not args.no_offset_search)
    print(f"Merged rows: {len(result)}")
    print(f"Saved to: {args.output}")
