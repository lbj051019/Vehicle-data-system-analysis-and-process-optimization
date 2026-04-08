# Battery Signal Analysis & Fault Warning

A portfolio-ready project for battery signal analysis, feature engineering, frequency-domain exploration, and fault warning modeling.

## What it does

- Compares voltage and current signals between paired samples
- Extracts statistical, FFT, STFT, and wavelet-based features
- Builds a baseline health-check feature table
- Trains a recall-focused Random Forest fault warning model
- Visualizes healthy vs. fault patterns

## Example data

This repository ships with a **small demo subset** of the provided dataset so the scripts can be run without the full original file.

- `Task3_Raw_Battery_Signal_Data.csv`
- `labels_sample.csv`

The demo subset keeps both healthy and fault samples.

## Project structure

```text
battery_health_fault_analysis_public/
├── Task3_Raw_Battery_Signal_Data.csv   # demo sample from the provided dataset
├── labels_sample.csv
├── src/
│   ├── baseline_health_check.py
│   ├── fault_warning_model.py
│   ├── frequency_10khz_sliding_window_features.py
│   ├── wavelet_transform_stft_10khz.py
│   ├── feature_visualized.py
│   ├── feature_statistics.py
│   ├── current_statistics.py
│   ├── fft_compare.py
│   └── voltage_current_time_series.py
├── docs/
│   └── project_report.pdf
├── outputs/
│   └── example/
└── run_demo.sh
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full demo on Windows PowerShell:

```powershell
python src/baseline_health_check.py
python src/fault_warning_model.py --input Task3_Raw_Battery_Signal_Data.csv --labels labels_sample.csv --outdir rf_outputs
```

Run the feature comparison scripts and enter a sample ID such as `0` when prompted:

```powershell
python src/voltage_current_time_series.py
python src/fft_compare.py
python src/frequency_10khz_sliding_window_features.py
python src/wavelet_transform_stft_10khz.py
```

## Notes

- The demo dataset is a reduced sample for GitHub sharing and quick testing.
- The model script uses sample-level labels and a recall-oriented Random Forest.
- Generated outputs are written to `rf_outputs/`, `vis_features/`, or the current working directory depending on the script.

## Author

Your Name
