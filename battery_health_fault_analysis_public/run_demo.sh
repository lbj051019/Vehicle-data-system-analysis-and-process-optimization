#!/usr/bin/env bash
set -e
printf '0\n' | python src/voltage_current_time_series.py
printf '0\n' | python src/fft_compare.py
printf '0\n' | python src/frequency_10khz_sliding_window_features.py
printf '0\n' | python src/wavelet_transform_stft_10khz.py
python src/baseline_health_check.py
python src/fault_warning_model.py --input Task3_Raw_Battery_Signal_Data.csv --labels labels_sample.csv --outdir rf_outputs
