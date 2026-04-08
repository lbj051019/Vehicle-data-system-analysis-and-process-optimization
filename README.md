# 🚀 AI & Data Engineering Portfolio

A collection of end-to-end projects combining AI, signal processing, and data engineering.

---

## 📌 Overview

This repository showcases three major projects:

1. Driving Behavior Analysis System  
2. Battery Health & Signal Analysis  
3. Battery Fault Detection Pipeline  

These projects demonstrate:

- Multi-source data processing  
- Signal processing (FFT, STFT, Kalman Filter)  
- Machine learning (Random Forest)  
- Feature engineering (time + frequency domain)  
- End-to-end pipeline design  

---

# 🚗 Project 1: Driving Behavior Analysis

📂 driving_behavior_analysis_public

## Description

Analyzes vehicle data (CAN + GPS) to detect driving patterns and anomalies.

## Key Features

- Data alignment (CAN + GPS)  
- Kalman filter for speed fusion  
- Anomaly detection (speed, RPM, acceleration)  
- Driving behavior classification:
  - Braking  
  - Turning  
  - Acceleration  
- Visualization:
  - 2D & 3D trajectories  
  - Driving condition distribution  

---

# 🔋 Project 2: Battery Health & Signal Analysis

📂 battery_health_fault_analysis_public

## Description

Extracts signal-level features from high-frequency battery data (10kHz).

## Key Features

- Time-domain feature extraction  
- Frequency-domain analysis (FFT)  
- Time-frequency analysis (STFT, Wavelet)  
- Sliding window feature extraction  
- Healthy vs Fault signal comparison  

---

# ⚡ Project 3: Battery Fault Detection System

📂 battery_project_github_ready

## Description

End-to-end machine learning pipeline for battery fault detection.

## Key Features

- Feature extraction from voltage & current signals  
- Baseline health detection (unsupervised)  
- Random Forest classification  
- Recall-priority threshold tuning  
- Model outputs:
  - Prediction results  
  - Feature importance  
  - Model evaluation report  

---

# 🧠 Skills Demonstrated

- Data Engineering (multi-source data fusion)  
- Signal Processing (FFT, STFT, Kalman)  
- Machine Learning (classification, evaluation)  
- Feature Engineering  
- Visualization  
- End-to-end system design  

---

# 📦 How to Run

Each project contains its own instructions.

Example:

pip install -r requirements.txt  
python src/model_training.py  

---

# 👤 Author

Your Name
