from __future__ import annotations

import argparse
from pathlib import Path

from src.anomaly_detection import run as anomaly_run
from src.brake_visualization import run as brake_run
from src.driving_conditions import run as conditions_run
from src.kalman_fusion import run as kalman_run
from src.merge import merge_datasets
from src.turn_detection import run as turn_run
from src.visualize_2d import run as plot2d_run
from src.visualize_3d import run as plot3d_run
from src.smoothing import run as smoothing_run
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full vehicle driving behavior analysis pipeline.")
    parser.add_argument("--can", default="examples/Task2_CAN_data.csv", help="CAN data CSV")
    parser.add_argument("--gps", default="examples/Task2_GPS_data.csv", help="GPS data CSV")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    example = ensure_dir(Path(out) / "example")

    print("Step 1/6: merging CAN and GPS data...")
    merged = merge_datasets(args.can, args.gps, Path(out) / "merged_data.csv")

    print("Step 2/6: Kalman-style speed fusion...")
    kalman = kalman_run(Path(out) / "merged_data.csv", Path(out) / "merged_data_with_kalman.csv")

    print("Step 3/6: anomaly detection...")
    anomaly = anomaly_run(Path(out) / "merged_data_with_kalman.csv", Path(out) / "merged_data_with_anomalies.csv")

    print("Step 4/6: turning point detection...")
    turn_run(Path(out) / "merged_data_with_anomalies.csv", Path(out) / "marked_turns.csv", Path(example) / "turn_points.png")

    print("Step 5/6: driving condition classification...")
    conditions_run(Path(out) / "merged_data_with_anomalies.csv", Path(out) / "driving_conditions_marked.csv", example)

    if not args.skip_plots:
        print("Step 6/6: generating visualizations...")
        brake_run(Path(out) / "merged_data_with_anomalies.csv", Path(example) / "brake_speed_rpm.png")
        plot2d_run(Path(out) / "merged_data.csv", Path(example) / "trajectory_2d.png")
        plot3d_run(Path(out) / "merged_data.csv", Path(example) / "trajectory_3d.png")
        smoothing_run(Path(out) / "merged_data.csv", "vehicle_speed_kmh", Path(example) / "smoothed_speed.png")
        smoothing_run(Path(out) / "merged_data.csv", "engine_rpm", Path(example) / "smoothed_rpm.png")

    print("Done.")
    print(f"Merged rows: {len(merged)}")
    print(f"Anomaly rows: {len(anomaly)}")


if __name__ == "__main__":
    main()
