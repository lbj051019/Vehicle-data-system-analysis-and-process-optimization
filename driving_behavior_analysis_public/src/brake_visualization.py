from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from .utils import ensure_dir, load_csv


def run(input_csv: str | Path, output_path: str | Path = "outputs/example/brake_speed_rpm.png") -> None:
    df = load_csv(input_csv)

    timestamp = df["timestamp"]
    combined_speed = df["combined_speed_kmh"]
    rpm = df["engine_rpm"]
    brake = df["brake_pedal_status"]

    colors = ["red" if b else "blue" for b in brake]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(timestamp, combined_speed, c=colors, marker="o", s=14)
    plt.xlabel("Timestamp")
    plt.ylabel("Combined speed (km/h)")
    plt.xticks([])
    plt.grid(True)
    plt.title("Vehicle Speed with Braking Indicators")

    plt.subplot(1, 2, 2)
    plt.scatter(timestamp, rpm, c=colors, marker="o", s=14)
    plt.xlabel("Timestamp")
    plt.ylabel("RPM")
    plt.xticks([])
    plt.grid(True)
    plt.title("Engine RPM with Braking Indicators")

    blue_patch = mpatches.Patch(color="blue", label="No Brake (0)")
    red_patch = mpatches.Patch(color="red", label="Braking (1)")
    plt.legend(handles=[blue_patch, red_patch], bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize braking indicators on speed and RPM.")
    parser.add_argument("--input", default="outputs/merged_data_with_anomalies.csv", help="Input CSV")
    parser.add_argument("--output", default="outputs/example/brake_speed_rpm.png", help="Output image")
    args = parser.parse_args()

    run(args.input, args.output)
