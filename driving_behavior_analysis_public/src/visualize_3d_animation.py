from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .utils import load_csv


def run(input_csv: str | Path) -> None:
    df = load_csv(input_csv)
    latitude = df["latitude"]
    longitude = df["longitude"]
    altitude = df["altitude"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], "b-", alpha=0.5)
    scatter = ax.scatter([], [], [], c=[], cmap="viridis", s=20)

    def update(num):
        line.set_data(longitude[:num], latitude[:num])
        line.set_3d_properties(altitude[:num])
        scatter._offsets3d = (longitude[:num], latitude[:num], altitude[:num])
        return line, scatter

    FuncAnimation(fig, update, frames=len(longitude), interval=200, blit=False)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("3D Trajectory Animation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate the 3D driving trajectory.")
    parser.add_argument("--input", default="outputs/merged_data.csv", help="Input CSV")
    args = parser.parse_args()
    run(args.input)
