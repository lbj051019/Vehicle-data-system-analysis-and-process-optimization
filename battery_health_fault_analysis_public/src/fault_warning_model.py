# -*- coding: utf-8 -*-
"""
步骤三：基于随机森林的故障预警（高召回优先，单模型RF、无标准化）
- 输入：
    Task3_Raw_Battery_Signal_Data.csv   # 必须，含: sample_id,time_step,voltage,current
    [可选] --labels labels.csv          # 两列: sample_id,label (Healthy/Fault 或 0/1)
- 输出（默认目录 rf_outputs/）：
    features_agg.csv
    rf_feature_importance.csv
    trained_random_forest.joblib
    model_report.txt
    prediction_results.csv
    prediction_results_all.csv
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy.signal import stft
try:
    from scipy.integrate import trapezoid as TRAPEZOID
except Exception:
    TRAPEZOID = getattr(np, "trapezoid", None)
    if TRAPEZOID is None:
        TRAPEZOID = np.trapz

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import joblib

# ===================== 固定参数 =====================
dt = 1e-4
fs = 10_000.0
STFT_NPERSEG = 512
STFT_OVERLAP = 0.90
RANDOM_STATE = 42

# ===================== 信号→时频→特征 =====================
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

def feat_highfreq_fraction(f, t, Sxx, fmin=1000.0):
    if f.size == 0 or t.size == 0:
        return 0.0
    be_all_t = TRAPEZOID(Sxx, f, axis=0)
    e_all = TRAPEZOID(be_all_t, t)
    mask = f >= fmin
    if not np.any(mask) or e_all <= 0:
        return 0.0
    be_hf_t = TRAPEZOID(Sxx[mask, :], f[mask], axis=0)
    e_hf = TRAPEZOID(be_hf_t, t)
    return float(e_hf / e_all) if e_all > 0 else 0.0

def feat_spectral_kurtosis_stable(x):
    """稳健谱'尖锐度'：Var(P)/Mean(P)^2（P为rfft功率，去DC）"""
    x = np.asarray(x, float)
    x = x - np.mean(x)
    if x.size < 8:
        return 0.0
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    P = (np.abs(X)**2)[1:]
    if P.size < 4:
        return 0.0
    m = np.mean(P)
    if m <= 0:
        return 0.0
    var = np.mean((P - m)**2)
    return float(var / (m**2 + 1e-30))

def feat_crest_factor(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x**2)) + 1e-30
    return float(np.max(np.abs(x)) / rms)

def feat_transient_rate(f, t, Sxx, fmin=2000.0, k=5.0):
    """≥fmin 频段能量超过 [median + k * MAD] 的时间占比"""
    if f.size == 0 or t.size == 0:
        return 0.0
    mask = f >= fmin
    if not np.any(mask):
        return 0.0
    be_t = TRAPEZOID(Sxx[mask, :], f[mask], axis=0)
    med = np.median(be_t)
    mad = np.median(np.abs(be_t - med)) + 1e-30
    thr = med + k * mad
    return float(np.mean(be_t > thr))

def extract_feats_for_signal(x):
    f, t, Sxx = compute_stft_power(x, fs)
    return {
        "hf_frac_1k": feat_highfreq_fraction(f, t, Sxx, fmin=1000.0),
        "spec_kurt":  feat_spectral_kurtosis_stable(x),
        "crest":      feat_crest_factor(x),
        "transient_rate_2k": feat_transient_rate(f, t, Sxx, fmin=2000.0, k=5.0),
    }

def prefix(d, p): return {f"{p}{k}": v for k, v in d.items()}

# ===================== 标签与评估工具 =====================
def load_labels_default(sample_ids):
    """默认：0–99 Healthy(0)，>=100 Fault(1)"""
    return np.array([0 if int(sid) <= 99 else 1 for sid in sample_ids], dtype=int)

def load_labels_from_file(label_path, sample_ids):
    lab = pd.read_csv(label_path)
    assert "sample_id" in lab.columns and "label" in lab.columns, "label文件需含 sample_id,label"
    m = {int(r.sample_id): r.label for _, r in lab.iterrows()}
    y = []
    for sid in sample_ids:
        v = m.get(int(sid), None)
        if v is None:
            raise ValueError(f"标签缺失 sample_id={sid}")
        if isinstance(v, str):
            v = v.strip().lower()
            y.append(1 if v in ["fault", "abnormal", "1"] else 0)
        else:
            y.append(int(v))
    return np.array(y, dtype=int)

def choose_threshold_by_fbeta(y_true, y_score, beta=2.0):
    """训练集扫描阈值 → 使Fβ最大（β>1更强调召回）"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision = precision[1:]; recall = recall[1:]
    if thresholds.size == 0 or precision.size == 0:
        return 0.5, 0.0, 0.0, 0.0
    beta2 = beta**2
    fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + 1e-12)
    i = int(np.argmax(fbeta))
    return float(thresholds[i]), float(precision[i]), float(recall[i]), float(fbeta[i])

def evaluate(y_test, y_score, thr, beta=2.0):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    P = precision_score(y_test, y_pred, zero_division=0)
    R = recall_score(y_test, y_pred, zero_division=0)
    F1 = f1_score(y_test, y_pred, zero_division=0)
    beta2 = beta**2
    Fbeta = (1 + beta2) * (P * R) / (beta2 * P + R + 1e-12)
    ap = average_precision_score(y_test, y_score)
    report = classification_report(y_test, y_pred, target_names=["Healthy","Fault"], digits=4)
    return dict(threshold=thr, precision=P, recall=R, F1=F1, Fbeta=Fbeta, AP=ap,
                confusion_matrix=cm, report=report, y_pred=y_pred)

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="随机森林故障预警（高召回优先，仅RF，无标准化）")
    parser.add_argument("--input", "-i", default="Task3_Raw_Battery_Signal_Data.csv",
                        help="原始CSV（含 sample_id,time_step,voltage,current）")
    parser.add_argument("--labels", "-l", default="",
                        help="可选：标签CSV（sample_id,label），label=Healthy/Fault或0/1")
    parser.add_argument("--outdir", "-o", default="rf_outputs",
                        help="输出目录")
    parser.add_argument("--test_size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="阈值选择的Fβ（β>1强调召回；默认2.0）")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读取原始数据 & 提取样本级特征（电压+电流合并）
    raw = pd.read_csv(args.input)
    need = {"sample_id","time_step","voltage","current"}
    if not need.issubset(raw.columns):
        raise ValueError(f"输入缺少列：{sorted(need - set(raw.columns))}")

    rows = []
    for sid in sorted(raw["sample_id"].unique()):
        d = raw[raw["sample_id"]==sid].sort_values("time_step")
        v = d["voltage"].to_numpy()
        i = d["current"].to_numpy()
        fv = extract_feats_for_signal(v)
        fi = extract_feats_for_signal(i)
        row = {"sample_id": int(sid)}
        row.update(prefix(fv,"v_"))
        row.update(prefix(fi,"i_"))
        rows.append(row)
    feats = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    feats.to_csv(outdir/"features_agg.csv", index=False)
    print("\n=== 前几行特征（features_agg.csv） ===")
    print(feats.head())

    # 2) 获取标签
    sample_ids = feats["sample_id"].to_numpy()
    if args.labels and Path(args.labels).exists():
        y = load_labels_from_file(args.labels, sample_ids)
    else:
        y = load_labels_default(sample_ids)

    X = feats.drop(columns=["sample_id"])

    # 3) —— 这里就是 train_test_split —— 划分训练 / 测试（分层） 👇
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, sample_ids, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[Split] 训练集: {len(y_tr)} 条, 测试集: {len(y_te)} 条 (test_size={args.test_size})")

    # 4) 类别不平衡权重
    classes = np.unique(y_tr)
    cls_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    cw_map = {c:w for c,w in zip(classes, cls_w)}
    sw_tr = compute_sample_weight(class_weight=cw_map, y=y_tr)

    # 5) 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tr, y_tr, sample_weight=sw_tr)

    # 6) 训练集上用 Fβ 选阈值（召回优先）
    score_tr = rf.predict_proba(X_tr)[:,1]
    thr, p_tr, r_tr, fbeta_tr = choose_threshold_by_fbeta(y_tr, score_tr, beta=args.beta)

    # 7) 测试集评估并打印
    score_te = rf.predict_proba(X_te)[:,1]
    ev = evaluate(y_te, score_te, thr, beta=args.beta)

    print("\n=== 训练集阈值（Fβ最大，召回优先） ===")
    print(f"beta={args.beta:.2f}  threshold={thr:.6f}  Train P={p_tr:.4f}  R={r_tr:.4f}  Fβ={fbeta_tr:.4f}")

    print("\n=== 测试集指标（关注 Recall） ===")
    print(f"Recall={ev['recall']:.4f}  Precision={ev['precision']:.4f}  F1={ev['F1']:.4f}  "
          f"Fβ={ev['Fbeta']:.4f}  AP={ev['AP']:.4f}")

    print("\n=== 混淆矩阵 (rows=True cols=Pred [Healthy, Fault]) ===")
    print(ev["confusion_matrix"])

    print("\n=== 分类报告 ===")
    print(ev["report"])

    # 8) 特征重要性（保存并打印Top10）
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.to_csv(outdir/"rf_feature_importance.csv", header=["importance"])
    print("\n=== 随机森林特征重要性 Top10 ===")
    for i, (name, val) in enumerate(fi.head(10).items(), 1):
        print(f"{i:>2}. {name}: {val:.6f}")

    # 9) 保存每个样本的预测结果 —— 测试集
    pred_te = (score_te >= thr).astype(int)
    pred_df = pd.DataFrame({
        "sample_id": id_te,
        "true_label": y_te,
        "score": score_te,
        "threshold": thr,
        "predicted": pred_te,
        "split": "test"
    }).sort_values("sample_id").reset_index(drop=True)
    pred_df.to_csv(outdir/"prediction_results.csv", index=False)
    print("\n=== 测试集样本预测（前10行） ===")
    print(pred_df.head(10))

    # 10) 保存每个样本的预测结果 —— 全部样本（训练+测试）
    score_all = rf.predict_proba(X)[:,1]
    pred_all = (score_all >= thr).astype(int)
    pred_all_df = pd.DataFrame({
        "sample_id": feats["sample_id"],
        "true_label": y,
        "score": score_all,
        "threshold": thr,
        "predicted": pred_all
    }).sort_values("sample_id").reset_index(drop=True)
    pred_all_df.to_csv(outdir/"prediction_results_all.csv", index=False)
    print("\n=== 全部样本预测（前10行） ===")
    print(pred_all_df.head(10))

    print(f"\n[OK] 已保存：\n- { (outdir/'features_agg.csv').resolve() }\n"
          f"- { (outdir/'rf_feature_importance.csv').resolve() }\n"
          f"- { (outdir/'prediction_results.csv').resolve() }\n"
          f"- { (outdir/'prediction_results_all.csv').resolve() }")

    # 11) 保存模型
    joblib.dump({"model": rf, "threshold": thr, "features": X.columns.tolist()},
                outdir/"trained_random_forest.joblib")

    # 12) 写入总报告
    lines = []
    lines.append("=== TRAIN Threshold by Fβ (Recall-priority) ===")
    lines.append(f"beta={args.beta:.2f}, threshold={thr:.6f}, Train P={p_tr:.4f}, R={r_tr:.4f}, Fβ={fbeta_tr:.4f}")
    lines.append("")
    lines.append("=== TEST Metrics ===")
    lines.append(f"Recall={ev['recall']:.4f}  Precision={ev['precision']:.4f}  "
                 f"F1={ev['F1']:.4f}  Fβ={ev['Fbeta']:.4f}  AP={ev['AP']:.4f}")
    lines.append("Confusion Matrix (rows=True cols=Pred [Healthy, Fault]):")
    lines.append(str(ev["confusion_matrix"]))
    lines.append("")
    lines.append("Classification Report:")
    lines.append(ev["report"])
    lines.append("")
    lines.append("=== RandomForest Feature Importance (Top 10) ===")
    for i, (name, val) in enumerate(fi.head(10).items(), 1):
        lines.append(f"{i:>2}. {name}: {val:.6f}")
    with open(outdir/"model_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] 已保存模型与报告：\n- { (outdir/'trained_random_forest.joblib').resolve() }\n"
          f"- { (outdir/'model_report.txt').resolve() }")

if __name__ == "__main__":
    main()
