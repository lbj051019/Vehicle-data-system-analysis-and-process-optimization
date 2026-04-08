from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .utils import ensure_dir, load_csv


def run(input_csv: str | Path, output_path: str | Path = "outputs/example/trajectory_3d.png") -> None:
    df = load_csv(input_csv)
    latitude = df["latitude"]
    longitude = df["longitude"]
    altitude = df["altitude"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(longitude, latitude, altitude, color="blue", linewidth=1, alpha=0.7)
    scatter = ax.scatter(longitude, latitude, altitude, c=altitude, cmap="viridis", s=20)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("3D Trajectory")
    fig.colorbar(scatter, label="Altitude (m)")
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 3D trajectory from GPS data.")
    parser.add_argument("--input", default="outputs/merged_data.csv", help="Input CSV")
    parser.add_argument("--output", default="outputs/example/trajectory_3d.png", help="Output image")
    args = parser.parse_args()
    run(args.input, args.output)
