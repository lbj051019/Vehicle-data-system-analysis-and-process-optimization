# -*- coding: utf-8 -*-
"""
可视化电气特征：Healthy vs Fault 对比
- 默认读取: rf_outputs/features_agg.csv
- 可选标签: --labels labels.csv  (两列: sample_id,label; label=Healthy/Fault 或 0/1)
- 可选模型: --model rf_outputs/trained_random_forest.joblib  (画特征重要性/二维散点)
- 输出图: vis_features/ 下的 PNG 图片
- 输出表: vis_features/feature_group_stats.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 读取标签 ----------
def load_labels_default(sample_ids):
    return np.array([0 if int(s) <= 99 else 1 for s in sample_ids], dtype=int)

def load_labels_from_file(label_path, sample_ids):
    lab = pd.read_csv(label_path)
    assert "sample_id" in lab.columns and "label" in lab.columns, "labels.csv 需含 sample_id,label"
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

# ---------- 效应量 ----------
def effect_size_cohen_d(x_h, x_f):
    x_h = np.asarray(x_h, float); x_f = np.asarray(x_f, float)
    mu_h, mu_f = np.mean(x_h), np.mean(x_f)
    s_h = np.std(x_h, ddof=1) + 1e-30
    s_f = np.std(x_f, ddof=1) + 1e-30
    n_h, n_f = len(x_h), len(x_f)
    s_p = np.sqrt(((n_h-1)*s_h**2 + (n_f-1)*s_f**2) / max(n_h+n_f-2, 1)) + 1e-30
    return (mu_f - mu_h) / s_p

def group_stats(x_h, x_f):
    def stats(arr):
        arr = np.asarray(arr, float)
        return dict(
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1) if len(arr)>1 else 0.0),
            median=float(np.median(arr)),
            iqr=float(np.percentile(arr,75)-np.percentile(arr,25))
        )
    sh, sf = stats(x_h), stats(x_f)
    d = effect_size_cohen_d(x_h, x_f)
    out = {}
    for k in sh: out[f"Healthy_{k}"] = sh[k]
    for k in sf: out[f"Fault_{k}"] = sf[k]
    out["effect_size_d"] = float(d)
    return out

# ---------- 画图工具 ----------
def plot_box_two_groups(values_h, values_f, title, ylabel, out_png):
    plt.figure(figsize=(6,4))
    # Matplotlib 3.9+: 用 tick_labels 取代 labels
    plt.boxplot([values_h, values_f], tick_labels=["Healthy","Fault"], showfliers=False)
    plt.ylabel(ylabel); plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_hist_overlay(values_h, values_f, title, xlabel, out_png, bins=30):
    plt.figure(figsize=(6,4))
    vmin = np.nanmin([np.nanmin(values_h), np.nanmin(values_f)])
    vmax = np.nanmax([np.nanmax(values_h), np.nanmax(values_f)])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax:
        vmin, vmax = 0.0, 1.0
    bins_edges = np.linspace(vmin, vmax, bins+1)
    hist_h, edges = np.histogram(values_h, bins=bins_edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    plt.plot(centers, hist_h, label="Healthy")
    hist_f, edges = np.histogram(values_f, bins=bins_edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    plt.plot(centers, hist_f, label="Fault")
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_feature_importance_bar(names, importances, out_png, topk=12):
    idx = np.argsort(importances)[::-1][:topk]
    plt.figure(figsize=(7, max(3, int(0.35*topk)+2)))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), [names[i] for i in idx][::-1])
    plt.xlabel("Importance"); plt.title(f"RandomForest Feature Importance (Top {topk})")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_scatter_2d(df, feat_x, feat_y, labels, out_png):
    plt.figure(figsize=(5.2,5))
    m_h = labels==0; m_f = labels==1
    plt.scatter(df.loc[m_h, feat_x], df.loc[m_h, feat_y], label="Healthy", alpha=0.7, s=18)
    plt.scatter(df.loc[m_f, feat_x], df.loc[m_f, feat_y], label="Fault", alpha=0.7, s=18, marker="x")
    plt.xlabel(feat_x); plt.ylabel(feat_y); plt.title(f"{feat_x} vs {feat_y}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser(description="可视化电气特征：Healthy vs Fault 对比")
    ap.add_argument("--features", "-f", default="rf_outputs/features_agg.csv",
                    help="特征CSV（包含 sample_id 与若干特征列）")
    ap.add_argument("--labels", "-l", default="",
                    help="可选：标签CSV（sample_id,label；label=Healthy/Fault 或 0/1）")
    ap.add_argument("--model", "-m", default="",
                    help="可选：已训练模型 joblib（用于画特征重要性与二维散点）")
    ap.add_argument("--outdir", "-o", default="vis_features",
                    help="输出目录")
    ap.add_argument("--topk", type=int, default=8, help="展示特征重要性的前K个")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    assert "sample_id" in df.columns, "features_agg.csv 必须包含 sample_id 列"
    feature_cols = [c for c in df.columns if c != "sample_id"]
    assert len(feature_cols)>0, "未找到特征列"

    # 标签
    sids = df["sample_id"].to_numpy()
    if args.labels and Path(args.labels).exists():
        y = load_labels_from_file(args.labels, sids)
    else:
        y = load_labels_default(sids)

    # 分组统计 + 可视化
    stats_rows = []
    for feat in feature_cols:
        vals = df[feat].to_numpy()
        v_h = vals[y==0]; v_f = vals[y==1]
        row = {"feature": feat}; row.update(group_stats(v_h, v_f))
        stats_rows.append(row)
        plot_box_two_groups(v_h, v_f,
                            title=f"{feat} (Healthy vs Fault)",
                            ylabel=feat,
                            out_png=outdir / f"box_{feat}.png")
        plot_hist_overlay(v_h, v_f,
                          title=f"{feat} Distribution",
                          xlabel=feat,
                          out_png=outdir / f"hist_{feat}.png")

    stats_df = pd.DataFrame(stats_rows).sort_values("effect_size_d", ascending=False)
    stats_df.to_csv(outdir / "feature_group_stats.csv", index=False)
    print("[OK] 已保存分组统计: ", (outdir / "feature_group_stats.csv").resolve())
    print(stats_df.head(12))

    # 若提供模型，则画特征重要性与二维散点
    if args.model and Path(args.model).exists():
        import joblib
        obj = joblib.load(args.model)
        rf = obj["model"]
        names = feature_cols
        importances = getattr(rf, "feature_importances_", None)
        if importances is not None:
            plot_feature_importance_bar(np.array(names), np.array(importances),
                                        out_png=outdir / "rf_feature_importance_topk.png",
                                        topk=args.topk)
            idx = np.argsort(importances)[::-1]
            if len(idx) >= 2:
                f1, f2 = names[idx[0]], names[idx[1]]
                plot_scatter_2d(df, f1, f2, y, out_png=outdir / f"scatter_{f1}_vs_{f2}.png")
                print(f"[OK] 已保存特征重要性与二维散点: {f1}, {f2}")
        else:
            print("[提示] 模型不含 feature_importances_，跳过特征重要性可视化。")

    print(f"[OK] 所有图像已输出至: {outdir.resolve()}")

if __name__ == "__main__":
    main()
