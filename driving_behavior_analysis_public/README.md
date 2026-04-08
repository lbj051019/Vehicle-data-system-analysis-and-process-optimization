# Vehicle Driving Behavior Analysis

An end-to-end data processing and visualization project for vehicle driving behavior analysis using CAN bus and GPS data.

## What it does

- Aligns CAN and GPS timestamps
- Corrects time offset between sources
- Fuses speed signals with a Kalman-style filter
- Detects abnormal acceleration, speed, and RPM values
- Detects braking and turning behavior
- Classifies driving conditions such as normal, stop, fast acceleration, and fast deceleration
- Produces 2D and 3D trajectory visualizations

## Project structure

```text
.
├── run_pipeline.py
├── requirements.txt
├── examples/
│   ├── Task2_CAN_data.csv
│   └── Task2_GPS_data.csv
├── outputs/
│   └── example/
│       ├── speed_comparison.png
│       ├── driving_condition_barplot_en.png
│       └── driving_condition_piechart_en.png
└── src/
    ├── merge.py
    ├── kalman_fusion.py
    ├── anomaly_detection.py
    ├── turn_detection.py
    ├── driving_conditions.py
    ├── brake_visualization.py
    ├── smoothing.py
    ├── visualize_2d.py
    ├── visualize_3d.py
    └── visualize_3d_animation.py
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py --can examples/Task2_CAN_data.csv --gps examples/Task2_GPS_data.csv
```

The pipeline will create an `outputs/` folder with merged data, anomaly flags, behavior labels, and figures.

## Example data

The repository includes a sample CAN dataset and GPS dataset under `examples/`.
They are based on the provided assignment data and can be used immediately to run the full pipeline.

## Main pipeline steps

1. `merge.py` — synchronize CAN and GPS by timestamp and optional offset search
2. `kalman_fusion.py` — fuse vehicle and GPS speed into one smoother speed signal
3. `anomaly_detection.py` — detect abnormal speed, acceleration, and RPM values
4. `turn_detection.py` — detect turning points from GPS trajectories
5. `driving_conditions.py` — classify driving conditions and generate bar/pie charts
6. `brake_visualization.py` — visualize brake events on speed/RPM signals
7. `visualize_2d.py` / `visualize_3d.py` — trajectory plots

## Default input columns

### CAN data
- `timestamp`
- `vehicle_speed_kmh`
- `engine_rpm`
- `brake_pedal_status`

### GPS data
- `timestamp`
- `latitude`
- `longitude`
- `altitude`
- `gps_speed_kmh`

## Outputs

- `merged_data.csv`
- `merged_data_with_kalman.csv`
- `merged_data_with_anomalies.csv`
- `marked_turns.csv`
- `driving_conditions_marked.csv`
- `driving_condition_barplot_en.png`
- `driving_condition_piechart_en.png`

## Notes

- This project is a portfolio/demo project.
- The thresholds are heuristic and meant for analysis and visualization.
- The sample data is included so the project can be run without any extra downloads.
