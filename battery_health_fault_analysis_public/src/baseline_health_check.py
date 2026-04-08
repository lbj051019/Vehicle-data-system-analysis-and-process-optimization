# -*- coding: utf-8 -*-
"""
基线健康检测（无监督阈值法，自动从 0–99 学健康阈值）
输入: Task3_Raw_Battery_Signal_Data.csv (columns: sample_id, time_step, voltage, current)
输出: health_features.csv (含特征与判定)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft

# ===== 采样与STFT设置 =====
dt = 1e-4
fs = 10_000.0
NYQ = fs / 2.0
STFT_NPERSEG = 512
STFT_OVERLAP = 0.90

try:
    from scipy.integrate import trapezoid as TRAPEZOID
except Exception:
    TRAPEZOID = getattr(np, "trapezoid", None)
    if TRAPEZOID is None:
        TRAPEZOID = np.trapz

def compute_stft_power(x, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1))
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(x)
    nperseg = int(max(32, min(int(nperseg), N)))
    noverlap = int(np.clip(nperseg * overlap_ratio, 0, nperseg - 1))
    f, t, Zxx = stft(x, fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap,
                     return_onesided=True, boundary='zeros', padded=True)
    Sxx = np.abs(Zxx) ** 2
    return f, t, Sxx

def feature_highfreq_fraction(f, t, Sxx, fmin=1000.0):
    """高频能量占比（≥fmin vs 全频，频率和时间双积分）"""
    if f.size == 0 or t.size == 0:
        return 0.0
    be_all_t = TRAPEZOID(Sxx, f, axis=0)
    e_all = TRAPEZOID(be_all_t, t)
    mask = f >= fmin
    if not np.any(mask):
        return 0.0
    be_hf_t = TRAPEZOID(Sxx[mask, :], f[mask], axis=0)
    e_hf = TRAPEZOID(be_hf_t, t)
    return float(e_hf / e_all) if e_all > 0 else 0.0

def feature_spectral_kurtosis(x):
    """功率谱峭度（对 rfft 幅度平方做峭度）"""
    x = np.asarray(x, float)
    x = x - np.mean(x)
    N = len(x)
    if N < 8:
        return 0.0
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    P = (np.abs(X) ** 2)
    P = P[1:]  # 去掉DC
    if P.size < 4:
        return 0.0
    m = np.mean(P)
    if m <= 0:
        return 0.0
    num = np.mean((P - m) ** 2)  # 实际这里是二阶中心矩；常见谱峭度实现用四阶/二阶^2
    den = (np.mean(P)) ** 2
    # 简化稳健版：用 (二阶中心矩 / 均值^2) 作为“尖锐度”近似，数值稳定且区分度强
    return float(num / (den + 1e-30))

def feature_crest_factor(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-30
    return float(np.max(np.abs(x)) / rms)

def feature_transient_rate(f, t, Sxx, fmin=2000.0, k=5.0):
    """
    ≥fmin 频带内能量超过自适应门限(中位数 + k*MAD)的时间占比
    """
    if f.size == 0 or t.size == 0:
        return 0.0
    mask = f >= fmin
    if not np.any(mask):
        return 0.0
    be_t = TRAPEZOID(Sxx[mask, :], f[mask], axis=0)  # [T,]
    med = np.median(be_t)
    mad = np.median(np.abs(be_t - med)) + 1e-30
    thr = med + k * mad
    rate = np.mean(be_t > thr)
    return float(rate)

def extract_features_for_signal(x):
    f, t, Sxx = compute_stft_power(x, fs)
    feats = {
        "hf_frac_1k": feature_highfreq_fraction(f, t, Sxx, fmin=1000.0),
        "spec_kurt":  feature_spectral_kurtosis(x),
        "crest":      feature_crest_factor(x),
        "transient_rate_2k": feature_transient_rate(f, t, Sxx, fmin=2000.0, k=5.0),
    }
    return feats

def main(input_csv="Task3_Raw_Battery_Signal_Data.csv", output_csv="health_features.csv"):
    df = pd.read_csv(input_csv)
    need = {"sample_id", "time_step", "voltage", "current"}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少必要列: {sorted(need - set(df.columns))}")

    rows = []
    for sid in sorted(df["sample_id"].unique()):
        d = df[df["sample_id"] == sid].sort_values("time_step")
        v = d["voltage"].to_numpy()
        i = d["current"].to_numpy()
        fv = extract_features_for_signal(v)
        fi = extract_features_for_signal(i)
        rows.append({"sample_id": sid, "signal": "voltage", **fv})
        rows.append({"sample_id": sid, "signal": "current", **fi})

    feats = pd.DataFrame(rows).sort_values(["sample_id", "signal"]).reset_index(drop=True)

    # ===== 从 0–99 学健康阈值（越大越异常的方向）=====
    healthy_mask = feats["sample_id"] <= 99
    feat_cols = ["hf_frac_1k", "spec_kurt", "crest", "transient_rate_2k"]
    stats = feats.loc[healthy_mask, ["signal"] + feat_cols].groupby("signal").agg(["mean", "std"])

    # 构建阈值：mean + 3*std
    thr = {}
    for sig in ["voltage", "current"]:
        thr[sig] = {}
        for c in feat_cols:
            mu = stats.loc[sig, (c, "mean")]
            sd = stats.loc[sig, (c, "std")]
            thr[sig][c] = float(mu + 3.0 * (0.0 if np.isnan(sd) else sd))

    # 判定：每个样本/信号，≥2项超阈 → Fault，否则 Healthy
    decisions = []
    for idx, row in feats.iterrows():
        sig = row["signal"]
        violations = sum(float(row[c]) > thr[sig][c] for c in feat_cols)
        label = "Fault" if violations >= 2 else "Healthy"
        decisions.append(label)
    feats["pred"] = decisions

    # 若假定 0–99 Healthy、100–199 Fault，打印简单混淆矩阵
    true = np.where(feats["sample_id"] <= 99, "Healthy", "Fault")
    pred = feats["pred"].values
    cm = pd.crosstab(pd.Series(true, name="True"), pd.Series(pred, name="Pred"))
    print("\n阈值(Mean+3σ)：")
    for sig in thr:
        print(f"  [{sig}] " + "  ".join([f"{k}={thr[sig][k]:.4g}" for k in feat_cols]))
    print("\n简易混淆矩阵（按 0–99 健康 / 100–199 故障假定）：")
    print(cm)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(output_csv, index=False)
    print(f"\n[OK] 已保存特征与预测: {Path(output_csv).resolve()}")

if __name__ == "__main__":
    main()
