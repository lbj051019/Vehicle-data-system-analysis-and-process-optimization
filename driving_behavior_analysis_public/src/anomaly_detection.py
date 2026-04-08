from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, load_csv, save_csv


def run(input_csv: str | Path, output_csv: str | Path = "outputs/merged_data_with_anomalies.csv") -> pd.DataFrame:
    df = load_csv(input_csv).copy()

    print("Missing values:")
    print(df.isnull().sum())

    df = df.dropna().reset_index(drop=True)

    df["speed_m/s"] = df["combined_speed_kmh"] / 3.6
    df["acceleration_m/s2"] = df["speed_m/s"].diff()
    df["acceleration_abnormal"] = (df["acceleration_m/s2"].abs() > 10).astype(int)

    df["combined_speed_abnormal"] = ((df["combined_speed_kmh"] > 250) | (df["combined_speed_kmh"] < 0)).astype(int)
    df["combined_speed_kmh"] = df["combined_speed_kmh"].where(df["combined_speed_kmh"] >= 0, 0)

    df["rpm_abnormal"] = ((df["engine_rpm"] > 7000) | (df["engine_rpm"] < 0)).astype(int)
    df["speed_diff_abnormal"] = ((df["vehicle_speed_kmh"] - df["gps_speed_kmh"]).abs() > 15).astype(int)

    save_csv(df, output_csv)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect anomalies in vehicle and GPS signals.")
    parser.add_argument("--input", default="outputs/merged_data_with_kalman.csv", help="Input CSV")
    parser.add_argument("--output", default="outputs/merged_data_with_anomalies.csv", help="Output CSV")
    args = parser.parse_args()

    ensure_dir(Path(args.output).parent)
    df = run(args.input, args.output)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
