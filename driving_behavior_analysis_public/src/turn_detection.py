from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir, load_csv, save_csv


def calculate_angle(p1, p2, p3) -> float:
    x1, y1 = p1[1], p1[0]
    x2, y2 = p2[1], p2[0]
    x3, y3 = p3[1], p3[0]

    vec1 = (x1 - x2, y1 - y2)
    vec2 = (x3 - x2, y3 - y2)
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mod1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    mod2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    if mod1 * mod2 == 0:
        return 0.0
    cos_theta = dot_product / (mod1 * mod2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def run(input_csv: str | Path, output_csv: str | Path = "outputs/marked_turns.csv", plot_path: str | Path | None = None) -> pd.DataFrame:
    df = load_csv(input_csv).copy()
    df["is_turn"] = False
    df["turn_angle"] = 0.0

    latitude = df["latitude"].to_numpy()
    longitude = df["longitude"].to_numpy()

    for i in range(1, len(df) - 1):
        p1 = (latitude[i - 1], longitude[i - 1])
        p2 = (latitude[i], longitude[i])
        p3 = (latitude[i + 1], longitude[i + 1])
        angle = calculate_angle(p1, p2, p3)
        df.at[i, "turn_angle"] = angle
        if 45 < angle < 135:
            df.at[i, "is_turn"] = True

    save_csv(df, output_csv)

    if plot_path:
        plt.figure(figsize=(10, 6))
        plt.scatter(df["longitude"], df["latitude"], c="blue", s=10, label="Normal Points")
        turn_points = df[df["is_turn"]]
        plt.scatter(turn_points["longitude"], turn_points["latitude"], c="red", s=30, label="Turn Points")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Vehicle Path with Turn Points Marked")
        plt.legend()
        plt.tight_layout()
        ensure_dir(Path(plot_path).parent)
        plt.savefig(plot_path, dpi=160)
        plt.close()

    print(f"Turn points found: {int(df['is_turn'].sum())}")
    print(f"Average turn angle: {df['turn_angle'].mean():.2f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect turning points from GPS trajectory.")
    parser.add_argument("--input", default="outputs/merged_data_with_anomalies.csv", help="Input CSV")
    parser.add_argument("--output", default="outputs/marked_turns.csv", help="Output CSV")
    parser.add_argument("--plot", default="outputs/example/turn_points.png", help="Plot output path")
    args = parser.parse_args()

    ensure_dir(Path(args.output).parent)
    run(args.input, args.output, args.plot)
