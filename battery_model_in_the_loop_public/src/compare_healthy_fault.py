# -*- coding: utf-8 -*-
"""
healthy.csv vs fault.csv 对比检测（10 kHz）
输出目录：compare_outputs/
- diff_events_voltage.csv / diff_events_current.csv   # 异常事件列表
- 若需要，可自行保存图像（当前默认 plt.show() 交互显示）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import stft

# ======================= 基本设置 =======================
dt = 1e-4            # 采样间隔（秒）
fs = 10_000.0        # 采样率（Hz）= 1/dt
nyquist = fs/2.0

# STFT 参数（与前文一致）
STFT_NPERSEG = 512
STFT_OVERLAP = 0.90   # hop ≈ 5.12 ms

# 频带（可按需调整）
freq_bands = {
    "0-100Hz":     (0,     100),
    "100-300Hz":   (100,   300),
    "300-1kHz":    (300,  1_000),
    "1k-2kHz":     (1_000, 2_000),
    "2k-3.5kHz":   (2_000, 3_500),
    "3.5k-5kHz":   (3_500, 5_000),
}

# 数值积分兜底（兼容不同 SciPy 版本）
try:
    from scipy.integrate import trapezoid as TRAPEZOID
except Exception:
    TRAPEZOID = getattr(np, "trapezoid", None)
    if TRAPEZOID is None:
        TRAPEZOID = np.trapz

# ======================= 工具函数 =======================
def pick_col(df, names):
    for n in names:
        if n in df.columns: return n
    raise ValueError(f"找不到列名（尝试过）：{names}")

def load_csv_to_uniform(path, fs=fs, dt=dt):
    """读取 CSV → 返回统一等间隔时间网格上的 (t, v, i)"""
    df = pd.read_csv(path)
    # 兼容列名
    vcol = pick_col(df, ["voltage", "V_V", "v", "Voltage"])
    icol = pick_col(df, ["current", "I_A", "i", "Current"])
    # 时间列优先级：time_step*dt > time > 构造索引
    if "time_step" in df.columns:
        df = df.sort_values("time_step")
        t = df["time_step"].to_numpy(dtype=float) * dt
    elif "time" in df.columns:
        df = df.sort_values("time")
        t = df["time"].to_numpy(dtype=float)
    else:
        # 无时间列则按等间隔构造
        t = np.arange(len(df), dtype=float) * dt

    v = df[vcol].to_numpy(dtype=float)
    i = df[icol].to_numpy(dtype=float)

    # 统一到等间隔网格
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        raise ValueError(f"{path}: 时间长度不合法")
    t_uniform = np.arange(t0, t1, 1.0/fs, dtype=float)
    v_uni = np.interp(t_uniform, t, v)
    i_uni = np.interp(t_uniform, t, i)
    return t_uniform, v_uni, i_uni

def align_two_series(t1, v1, i1, t2, v2, i2, fs=fs):
    """对齐到公共时间窗口并返回同一网格"""
    t_start = max(t1[0], t2[0])
    t_end   = min(t1[-1], t2[-1])
    if t_end <= t_start:
        raise ValueError("两段数据时间范围无交集，无法对齐。")
    t = np.arange(t_start, t_end, 1.0/fs, dtype=float)
    v1i = np.interp(t, t1, v1); i1i = np.interp(t, t1, i1)
    v2i = np.interp(t, t2, v2); i2i = np.interp(t, t2, i2)
    return t, v1i, i1i, v2i, i2i

def compute_power_spectrum(x, dt=dt):
    x = np.asarray(x, float); N = len(x)
    x = x - np.mean(x)
    w = np.hanning(N); xw = x*w; U = np.mean(w**2)
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=dt)
    P = (np.abs(X)**2) * 2.0 / ((N**2)*U)
    P[0] *= 0.5
    if N % 2 == 0: P[-1] *= 0.5
    return freqs, P

def compute_stft_power(x, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP):
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(x)
    nperseg = int(max(32, min(int(nperseg), N)))
    noverlap = int(np.clip(nperseg*overlap_ratio, 0, nperseg-1))
    f, t, Zxx = stft(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                     return_onesided=True, boundary='zeros', padded=True)
    Sxx = np.abs(Zxx)**2
    Sxx = np.nan_to_num(Sxx, nan=0.0, posinf=0.0, neginf=0.0)
    return f, t, Sxx

def band_energy_over_time(f, t, Sxx, bands=freq_bands, nyquist=nyquist):
    out = {}
    for name, (lo, hi) in bands.items():
        lo_c, hi_c = max(0.0, lo), min(nyquist, hi)
        if f.size==0 or t.size==0 or hi_c<=lo_c:
            out[name] = np.zeros_like(t); continue
        mask = (f>=lo_c) & (f<hi_c)
        out[name] = TRAPEZOID(Sxx[mask,:], f[mask], axis=0) if np.any(mask) else np.zeros_like(t)
    return out

def detect_diff_anomalies(t, f, S_fault, S_healthy,
                          f_min=1_000.0, db_thresh=1.0,
                          min_duration_sec=0.02, merge_gap_sec=0.01):
    """在差分时频图 (Fault−Healthy, dB) 上找异常时段"""
    eps = 1e-30
    Z_fault = 10*np.log10(S_fault + eps)
    Z_health = 10*np.log10(S_healthy + eps)
    Z_diff = Z_fault - Z_health
    if Z_diff.size==0 or t.size==0 or f.size==0:
        return []
    fmask = (f >= f_min)
    if not np.any(fmask):
        return []
    M = Z_diff[fmask, :] > db_thresh
    if not np.any(M):
        return []
    time_mask = M.any(axis=0)
    hop = (t[1]-t[0]) if t.size>1 else 1.0
    def frames(sec): return max(1, int(round(sec/hop)))
    max_gap = frames(merge_gap_sec)
    min_len = frames(min_duration_sec)
    idx = np.where(time_mask)[0]
    spans = []
    if idx.size:
        start = idx[0]; prev = idx[0]
        for k in idx[1:]:
            if (k - prev - 1) <= max_gap:
                prev = k
            else:
                if (prev - start + 1) >= min_len:
                    spans.append((start, prev))
                start = k; prev = k
        if (prev - start + 1) >= min_len:
            spans.append((start, prev))
    events = []
    Fsel = f[fmask]; Zsel = Z_diff[fmask, :]
    for i0, i1 in spans:
        block = Zsel[:, i0:i1+1]
        pf, pt = np.unravel_index(np.argmax(block), block.shape)
        events.append({
            "start_t": float(t[i0]),
            "end_t":   float(t[i1]),
            "duration": float(t[i1]-t[i0]),
            "peak_t":  float(t[i0+pt]),
            "peak_freq": float(Fsel[pf]),
            "peak_db": float(block[pf,pt])
        })
    return events

def print_events(events, tag):
    print(f"\n=== 异常事件（{tag}） f>=1kHz Δ>{1.0} dB ===")
    if not events:
        print("无。"); return
    for i,e in enumerate(events,1):
        print(f"[{i}] {e['start_t']:.6f}s ~ {e['end_t']:.6f}s "
              f"(dur={e['duration']*1e3:.2f} ms)  "
              f"peak@{e['peak_t']:.6f}s  {e['peak_freq']/1000:.3f} kHz  "
              f"Δ={e['peak_db']:.2f} dB")

# ======================= 主流程 =======================
def main(healthy_path="healthy.csv", fault_path="fault.csv", outdir="compare_outputs"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读取并统一网格
    t_h, v_h, i_h = load_csv_to_uniform(healthy_path, fs=fs, dt=dt)
    t_f, v_f, i_f = load_csv_to_uniform(fault_path,   fs=fs, dt=dt)

    # 2) 对齐到公共时间窗口
    t, v_h, i_h, v_f, i_f = align_two_series(t_h, v_h, i_h, t_f, v_f, i_f, fs=fs)
    print(f"[Info] 对齐后时长：{t[-1]-t[0]:.3f}s，样本数：{len(t)}，fs={fs:.0f} Hz")

    # 3) 全局功率谱
    freqs_vh, P_vh = compute_power_spectrum(v_h, dt=1/fs)
    freqs_vf, P_vf = compute_power_spectrum(v_f, dt=1/fs)
    freqs_ih, P_ih = compute_power_spectrum(i_h, dt=1/fs)
    freqs_if, P_if = compute_power_spectrum(i_f, dt=1/fs)

    # 4) STFT
    f_vh, t_vh, S_vh = compute_stft_power(v_h, fs)
    f_vf, t_vf, S_vf = compute_stft_power(v_f, fs)
    f_ih, t_ih, S_ih = compute_stft_power(i_h, fs)
    f_if, t_if, S_if = compute_stft_power(i_f, fs)

    # 5) 分频带能量
    be_vh = band_energy_over_time(f_vh, t_vh, S_vh)
    be_vf = band_energy_over_time(f_vf, t_vf, S_vf)
    be_ih = band_energy_over_time(f_ih, t_ih, S_ih)
    be_if = band_energy_over_time(f_if, t_if, S_if)

    # 6) 差分异常检测（Fault−Healthy）
    events_v = detect_diff_anomalies(t_vf, f_vf, S_vf, S_vh,
                                     f_min=1_000.0, db_thresh=1.0,
                                     min_duration_sec=0.02, merge_gap_sec=0.01)
    events_i = detect_diff_anomalies(t_if, f_if, S_if, S_ih,
                                     f_min=1_000.0, db_thresh=1.0,
                                     min_duration_sec=0.02, merge_gap_sec=0.01)
    print_events(events_v, "Voltage STFT Difference")
    print_events(events_i, "Current STFT Difference")

    pd.DataFrame(events_v).to_csv(outdir/"diff_events_voltage.csv", index=False)
    pd.DataFrame(events_i).to_csv(outdir/"diff_events_current.csv", index=False)
    print(f"\n[OK] 已保存事件：\n- { (outdir/'diff_events_voltage.csv').resolve() }\n- { (outdir/'diff_events_current.csv').resolve() }")

    # =================== 可视化 ===================
    eps = 1e-30

    # 图1：时域对齐对比
    fig1, ax1 = plt.subplots(2,1, figsize=(14,6), constrained_layout=True)
    ax1[0].plot(t, v_h, label='Healthy'); ax1[0].plot(t, v_f, '--', label='Fault')
    ax1[0].set_title('Voltage (Time)'); ax1[0].set_xlabel('s'); ax1[0].set_ylabel('V'); ax1[0].grid(True); ax1[0].legend()
    ax1[1].plot(t, i_h, label='Healthy'); ax1[1].plot(t, i_f, '--', label='Fault')
    ax1[1].set_title('Current (Time)'); ax1[1].set_xlabel('s'); ax1[1].set_ylabel('A'); ax1[1].grid(True); ax1[1].legend()
    plt.show()

    # 图2：全局功率谱
    fig2, ax2 = plt.subplots(2,1, figsize=(14,6), constrained_layout=True)
    ax2[0].plot(freqs_vh, 10*np.log10(P_vh+eps), label='Healthy'); ax2[0].plot(freqs_vf, 10*np.log10(P_vf+eps), '--', label='Fault')
    ax2[0].set_xlim(0, nyquist); ax2[0].set_title('Voltage Spectrum (dB)'); ax2[0].set_xlabel('Hz'); ax2[0].set_ylabel('dB'); ax2[0].grid(True); ax2[0].legend()
    ax2[1].plot(freqs_ih, 10*np.log10(P_ih+eps), label='Healthy'); ax2[1].plot(freqs_if, 10*np.log10(P_if+eps), '--', label='Fault')
    ax2[1].set_xlim(0, nyquist); ax2[1].set_title('Current Spectrum (dB)'); ax2[1].set_xlabel('Hz'); ax2[1].set_ylabel('dB'); ax2[1].grid(True); ax2[1].legend()
    plt.show()

    # 图3：STFT（2x2：电压H/F、 电流H/F），统一色阶
    stft_db_arrays = [
        10*np.log10(S_vh+eps), 10*np.log10(S_vf+eps),
        10*np.log10(S_ih+eps), 10*np.log10(S_if+eps)
    ]
    vals = np.concatenate([a.ravel() for a in stft_db_arrays])
    vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)

    fig3, axes3 = plt.subplots(2,2, figsize=(16,9), constrained_layout=True)
    def _pm(ax, t_, f_, Z, title):
        im = ax.pcolormesh(t_, f_, Z, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_ylim(0, nyquist); ax.set_title(title); ax.set_xlabel('Time (s)'); ax.set_ylabel('Hz')
        return im
    _pm(axes3[0,0], t_vh, f_vh, stft_db_arrays[0], 'Voltage STFT - Healthy')
    _pm(axes3[0,1], t_vf, f_vf, stft_db_arrays[1], 'Voltage STFT - Fault')
    _pm(axes3[1,0], t_ih, f_ih, stft_db_arrays[2], 'Current STFT - Healthy')
    im = _pm(axes3[1,1], t_if, f_if, stft_db_arrays[3], 'Current STFT - Fault')
    cbar = fig3.colorbar(im, ax=axes3.ravel().tolist(), shrink=0.9, pad=0.02); cbar.set_label('Power (dB)')
    plt.show()

    # 图4：差分时频图（Fault−Healthy, dB）
    def plot_diff(t_h, f_h, S_h, t_f, f_f, S_f, title):
        # 以 F 端轴为准做差（两者步长相同；若不同，也能显示）
        eps = 1e-30
        Zf = 10*np.log10(S_f+eps); Zh = 10*np.log10(S_h+eps)
        # 如果时间轴不同，这里简单取 Fault 轴；视觉比较用
        Z_diff = Zf - Zh
        vmin = np.percentile(Z_diff, 5); vmax = np.percentile(Z_diff, 95)
        plt.figure(figsize=(10,4))
        plt.pcolormesh(t_f, f_f, Z_diff, shading='auto', vmin=vmin, vmax=vmax)
        plt.ylim(0, nyquist); plt.title(title+' (Fault - Healthy, dB)')
        plt.xlabel('Time (s)'); plt.ylabel('Hz'); cb = plt.colorbar(); cb.set_label('Δ Power (dB)')
        plt.tight_layout(); plt.show()

    plot_diff(t_vh, f_vh, S_vh, t_vf, f_vf, S_vf, 'Voltage STFT Difference')
    plot_diff(t_ih, f_ih, S_ih, t_if, f_if, S_if, 'Current STFT Difference')

if __name__ == "__main__":
    main()  # 如需改路径：main("healthy.csv","fault.csv","compare_outputs")
