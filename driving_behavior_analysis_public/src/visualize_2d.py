from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir, load_csv


def run(input_csv: str | Path, output_path: str | Path = "outputs/example/trajectory_2d.png") -> None:
    df = load_csv(input_csv)
    plt.figure(figsize=(8, 6))
    plt.plot(df["longitude"], df["latitude"], "o-", linewidth=1, markersize=3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("2D Trajectory")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 2D trajectory from GPS data.")
    parser.add_argument("--input", default="outputs/merged_data.csv", help="Input CSV")
    parser.add_argument("--output", default="outputs/example/trajectory_2d.png", help="Output image")
    args = parser.parse_args()
    run(args.input, args.output)
