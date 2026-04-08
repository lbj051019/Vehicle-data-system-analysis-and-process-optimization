# rf_detect_healthy.py
# 用训练好的 RF 模型检测 healthy.csv（单样本或带 sample_id 的多样本）

import numpy as np, pandas as pd, joblib
from pathlib import Path
from scipy.signal import stft

# ==== 和训练保持一致的常量 ====
FS = 10_000.0           # 训练时的 fs
STFT_NPERSEG = 512
STFT_OVERLAP = 0.90

# ==== 特征计算（和训练一致） ====
def compute_stft_power(x, fs=FS, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.array([0.0]), np.array([0.0]), np.zeros((1,1))
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(x)
    nperseg = int(max(32, min(int(nperseg), N)))
    noverlap = int(np.clip(nperseg * overlap_ratio, 0, nperseg - 1))
    f, t, Zxx = stft(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                     return_onesided=True, boundary='zeros', padded=True)
    Sxx = np.abs(Zxx)**2
    return f, t, Sxx

def trapz_xy(y, x, axis=-1):
    # 兼容不同 SciPy 版本
    try:
        from scipy.integrate import trapezoid as TRAPEZOID
    except Exception:
        TRAPEZOID = None
    if TRAPEZOID is None:
        return np.trapz(y, x, axis=axis)
    return TRAPEZOID(y, x, axis=axis)

def feat_highfreq_fraction(f, t, Sxx, fmin=1000.0):
    if f.size == 0 or t.size == 0: return 0.0
    be_all_t = trapz_xy(Sxx, f, axis=0)
    e_all = trapz_xy(be_all_t, t)
    mask = f >= fmin
    if not np.any(mask) or e_all <= 0: return 0.0
    be_hf_t = trapz_xy(Sxx[mask,:], f[mask], axis=0)
    e_hf = trapz_xy(be_hf_t, t)
    return float(e_hf/e_all) if e_all>0 else 0.0

def feat_spectral_kurtosis_stable(x):
    x = np.asarray(x, float); x = x - np.mean(x)
    if x.size < 8: return 0.0
    w = np.hanning(x.size); X = np.fft.rfft(x*w); P = (np.abs(X)**2)[1:]
    if P.size < 4: return 0.0
    m = np.mean(P); 
    if m <= 0: return 0.0
    var = np.mean((P - m)**2)
    return float(var/(m**2 + 1e-30))

def feat_crest_factor(x):
    x = np.asarray(x, float); 
    if x.size == 0: return 0.0
    x = x - np.mean(x); rms = np.sqrt(np.mean(x**2)) + 1e-30
    return float(np.max(np.abs(x))/rms)

def feat_transient_rate(f, t, Sxx, fmin=2000.0, k=5.0):
    if f.size == 0 or t.size == 0: return 0.0
    mask = f >= fmin
    if not np.any(mask): return 0.0
    be_t = trapz_xy(Sxx[mask,:], f[mask], axis=0)
    med = np.median(be_t); mad = np.median(np.abs(be_t - med)) + 1e-30
    thr = med + k*mad
    return float(np.mean(be_t > thr))

def extract_feats_for_signal(x):
    f, t, Sxx = compute_stft_power(x, fs=FS)
    return {
        "hf_frac_1k": feat_highfreq_fraction(f, t, Sxx, fmin=1000.0),
        "spec_kurt":  feat_spectral_kurtosis_stable(x),
        "crest":      feat_crest_factor(x),
        "transient_rate_2k": feat_transient_rate(f, t, Sxx, fmin=2000.0, k=5.0),
    }

def prefix(d, p): return {f"{p}{k}": v for k,v in d.items()}

# ==== 推理 ====
def main(csv_path="healthy.csv", model_path="rf_outputs/trained_random_forest.joblib",
         out_path="rf_outputs/healthy_detection.csv"):
    outdir = Path(out_path).parent; outdir.mkdir(parents=True, exist_ok=True)

    # 1) 加载模型与阈值
    bundle = joblib.load(model_path)
    rf = bundle["model"]; thr = bundle["threshold"]; feat_names = bundle["features"]

    # 2) 读CSV（支持：仅单段波形；或已带 sample_id 的多段）
    df = pd.read_csv(csv_path)
    # 排序：优先 time_step，其次 time
    if "time_step" in df.columns: 
        df = df.sort_values("time_step")
    elif "time" in df.columns:
        df = df.sort_values("time")
    # 若无 sample_id，默认视作一个样本 id=0
    if "sample_id" not in df.columns:
        df["sample_id"] = 0

    rows = []
    for sid, d in df.groupby("sample_id"):
        v = d["voltage"].to_numpy()
        i = d["current"].to_numpy()
        fv = extract_feats_for_signal(v)
        fi = extract_feats_for_signal(i)
        row = {"sample_id": int(sid)}; row.update(prefix(fv,"v_")); row.update(prefix(fi,"i_"))
        rows.append(row)
    feats = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    # 3) 对齐特征列（训练时的列顺序）
    X = feats.reindex(columns=feat_names, fill_value=0.0)

    # 4) 打分 & 判别
    score = rf.predict_proba(X)[:,1]
    pred  = (score >= thr).astype(int)

    # 5) 保存 & 打印
    out = feats[["sample_id"]].copy()
    out["score"] = score
    out["threshold"] = thr
    out["predicted"] = pred
    out.to_csv(out_path, index=False)

    print("\n=== 检测结果 ===")
    print(out)
    print(f"\n已保存到：{Path(out_path).resolve()}")

if __name__ == "__main__":
    main()
