from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, load_csv, save_csv

STOP_SPEED = 1.0
FAST_ACCEL = 1.5
FAST_DECEL = -1.5


def run(input_csv: str | Path, output_csv: str | Path = "outputs/driving_conditions_marked.csv", out_dir: str | Path = "outputs/example") -> pd.DataFrame:
    df = load_csv(input_csv).copy()
    df["condition"] = "normal"
    df.loc[df["combined_speed_kmh"] <= STOP_SPEED, "condition"] = "stop"
    df.loc[df["acceleration_m/s2"] > FAST_ACCEL, "condition"] = "fast_accel"
    df.loc[df["acceleration_m/s2"] < FAST_DECEL, "condition"] = "fast_decel"

    save_csv(df, output_csv)

    out_dir = ensure_dir(out_dir)
    counts = df["condition"].value_counts()
    colors = {"normal": "gray", "stop": "blue", "fast_accel": "green", "fast_decel": "red"}
    color_list = [colors[c] for c in counts.index]

    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color=color_list)
    plt.title("Distribution of Driving Conditions")
    plt.xlabel("Condition Type")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "driving_condition_barplot_en.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=color_list, wedgeprops={"edgecolor": "black"})
    plt.title("Proportion of Driving Conditions")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "driving_condition_piechart_en.png", dpi=160)
    plt.close()

    print(df["condition"].value_counts())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify driving conditions and generate summary charts.")
    parser.add_argument("--input", default="outputs/merged_data_with_anomalies.csv", help="Input CSV")
    parser.add_argument("--output", default="outputs/driving_conditions_marked.csv", help="Output CSV")
    parser.add_argument("--out-dir", default="outputs/example", help="Directory for chart outputs")
    args = parser.parse_args()

    ensure_dir(Path(args.output).parent)
    run(args.input, args.output, args.out_dir)
