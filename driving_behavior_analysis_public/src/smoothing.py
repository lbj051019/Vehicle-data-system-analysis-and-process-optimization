from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir, load_csv


def moving_average(values, window_size: int = 5):
    window = np.ones(window_size) / window_size
    return np.convolve(values, window, mode="same")


def run(input_csv: str | Path, column: str, output_path: str | Path | None = None, window_size: int = 5) -> None:
    df = load_csv(input_csv)
    ts = df["timestamp"]
    raw = df[column]
    smooth = moving_average(raw.to_numpy(), window_size=window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(ts, raw, "o", label="raw")
    plt.plot(ts, smooth, label=f"smoothed (window={window_size})", linewidth=2)
    plt.xlabel("timestamp")
    plt.ylabel(column)
    plt.legend()
    plt.xticks([])
    plt.grid(True)
    plt.title(f"Smoothed {column}")
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=160)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply moving average smoothing to speed or RPM.")
    parser.add_argument("--input", default="outputs/merged_data.csv", help="Input CSV")
    parser.add_argument("--column", default="vehicle_speed_kmh", help="Column to smooth")
    parser.add_argument("--output", default="", help="Optional output image")
    parser.add_argument("--window", type=int, default=5, help="Window size")
    args = parser.parse_args()

    run(args.input, args.column, args.output or None, args.window)
