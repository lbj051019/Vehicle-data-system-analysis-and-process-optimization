from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, load_csv, save_csv


def kalman_1d_fusion(vehicle_speed: np.ndarray, gps_speed: np.ndarray, process_var: float = 1.0, r_vehicle: float = 4.0, r_gps: float = 25.0) -> np.ndarray:
    """Simple 1D Kalman-style fusion without external filter libraries."""
    x = float(vehicle_speed[0]) if len(vehicle_speed) else 0.0
    p = 1.0
    fused = []

    for v, g in zip(vehicle_speed, gps_speed):
        # predict
        p = p + process_var

        # update with vehicle sensor
        k = p / (p + r_vehicle)
        x = x + k * (float(v) - x)
        p = (1 - k) * p

        # update with GPS sensor
        k = p / (p + r_gps)
        x = x + k * (float(g) - x)
        p = (1 - k) * p

        fused.append(x)

    return np.asarray(fused, dtype=float)


def run(input_csv: str | Path, output_csv: str | Path = "outputs/merged_data_with_kalman.csv") -> pd.DataFrame:
    df = load_csv(input_csv)
    df = df.dropna(subset=["vehicle_speed_kmh", "gps_speed_kmh"]).copy()
    fused = kalman_1d_fusion(df["vehicle_speed_kmh"].to_numpy(), df["gps_speed_kmh"].to_numpy())
    df["combined_speed_kmh"] = fused
    save_csv(df, output_csv)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse vehicle speed and GPS speed using a simple Kalman-style filter.")
    parser.add_argument("--input", default="outputs/merged_data.csv", help="Input merged CSV")
    parser.add_argument("--output", default="outputs/merged_data_with_kalman.csv", help="Output CSV")
    args = parser.parse_args()

    ensure_dir(Path(args.output).parent)
    df = run(args.input, args.output)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
