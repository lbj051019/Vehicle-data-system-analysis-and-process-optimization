# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import stft
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ======================= 基本设置（已按 10 kHz 采样调整） =======================
dt = 1e-4                 # 采样间隔：100 微秒
fs = 10_000.0             # 采样率：10 kHz
nyquist = fs / 2.0        # 奈奎斯特：5 kHz

# 仅分析 1–4.8 kHz（避免贴边 5 kHz）
freq_bands = {
    "1-2kHz":  (1_000, 2_000),
    "2-3kHz":  (2_000, 3_000),
    "3-4kHz":  (3_000, 4_000),
    "4-4.8kHz":(4_000, 4_800),
}
MICRO_BAND = (1_000, 4_800)

# STFT 参数（为 10 kHz 数据重设：时间/频率兼顾）
# nperseg=512 -> Δf≈19.53 Hz；overlap=0.875 -> hop≈6.4 ms
STFT_NPERSEG = 512
STFT_OVERLAP = 0.875

# ======================= 数值积分（trapz 兜底） =======================
try:
    from scipy.integrate import trapezoid as TRAPEZOID
except Exception:
    TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz

# ======================= 数据读取与样本选择 =======================
df = pd.read_csv('Task3_Raw_Battery_Signal_Data.csv')

available_samples = df['sample_id'].unique()
while True:
    try:
        user_input = input("请输入要查看的 sample_id (0-99 或 100-199): ")
        target_sample = int(user_input)
        if not (0 <= target_sample <= 199):
            print("错误：请输入 0-199 之间的整数！")
            continue
        if 0 <= target_sample <= 99:
            sample_id_pair = [target_sample, target_sample + 100]
        else:
            sample_id_pair = [target_sample - 100, target_sample]
        valid_samples = [sid for sid in sample_id_pair if sid in available_samples]
        if len(valid_samples) < 2:
            print(f"错误：需要两个 sample_id 进行对比，但 {sample_id_pair} 中只有 {valid_samples} 存在！")
            continue
        break
    except ValueError:
        print("错误：请输入一个有效的整数 sample_id！")

print(f"对比的 sample_id 为: {sample_id_pair}")

sample1 = df[df['sample_id'] == sample_id_pair[0]].sort_values('time_step')
sample2 = df[df['sample_id'] == sample_id_pair[1]].sort_values('time_step')

# 仅保留电流
current1 = sample1['current'].values
current2 = sample2['current'].values

# ======================= 时域统计（仅电流；mean_raw 单独输出） =======================
def calculate_statistical_features(signal):
    x = np.asarray(signal, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    feats = {}
    feats['mean_raw'] = float(np.mean(x))
    x_dc = x - feats['mean_raw']
    feats['std'] = float(np.std(x_dc))
    feats['rms'] = float(np.sqrt(np.mean(x_dc**2)))
    feats['peak_to_peak'] = float(np.max(x) - np.min(x))
    feats['skewness'] = float(skew(x_dc))
    feats['kurtosis'] = float(kurtosis(x_dc))
    rms = feats['rms'] if feats['rms'] != 0 else 1e-12
    mean_abs = np.mean(np.abs(x_dc)) or 1e-12
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(x_dc))) or 1e-12
    feats['crest_factor']  = float(np.max(np.abs(x_dc)) / rms)
    feats['impulse_factor'] = float(np.max(np.abs(x_dc)) / mean_abs)
    feats['margin_factor']  = float(np.max(np.abs(x_dc)) / (mean_sqrt_abs**2))
    return feats

def print_features(title, feats):
    print(f"\n{title}")
    for k, v in feats.items():
        print(f"{k}: {v:.6f}")

current_features_sample1 = calculate_statistical_features(current1)
current_features_sample2 = calculate_statistical_features(current2)
print_features(f"Sample 1 (ID: {sample_id_pair[0]}) Current Features:", current_features_sample1)
print_features(f"Sample 2 (ID: {sample_id_pair[1]}) Current Features:", current_features_sample2)

# ======================= FFT 功率谱（辅助） =======================
def compute_power_spectrum(signal, dt=dt):
    x = np.asarray(signal, dtype=float)
    N = len(x)
    if N == 0:
        raise ValueError("输入信号长度为 0")
    x = x - np.mean(x)
    w = np.hanning(N)
    xw = x * w
    U = np.mean(w**2)
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=dt)
    power = (np.abs(X)**2) * 2.0 / ((N**2) * U)
    power[0] *= 0.5
    if N % 2 == 0:
        power[-1] *= 0.5
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
    total_energy = float(np.sum(power) * (df if df > 0 else 1.0))
    return freqs, power, df, total_energy

freqs_i1, P_i1, _, _ = compute_power_spectrum(current1, dt)
freqs_i2, P_i2, _, _ = compute_power_spectrum(current2, dt)

# ======================= STFT（仅电流） =======================
def compute_stft_power(signal, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP):
    x = np.asarray(signal, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(x)
    if N < 2:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1))
    nperseg = int(max(32, nperseg))
    nperseg = min(nperseg, N)
    noverlap = int(np.clip(nperseg * overlap_ratio, 0, nperseg - 1))
    f, t, Zxx = stft(x, fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap,
                     return_onesided=True, boundary='zeros', padded=True)
    Sxx = np.abs(Zxx) ** 2
    Sxx = np.nan_to_num(Sxx, nan=0.0, posinf=0.0, neginf=0.0)
    return f, t, Sxx

def band_energy_over_time(f, t, Sxx, bands, nyquist=nyquist):
    out = {}
    for name, (lo, hi) in bands.items():
        lo_c, hi_c = max(0.0, lo), min(nyquist, hi)
        if f.size == 0 or t.size == 0 or hi_c <= lo_c:
            out[name] = np.zeros_like(t)
            continue
        mask = (f >= lo_c) & (f < hi_c)
        out[name] = TRAPEZOID(Sxx[mask, :], f[mask], axis=0) if np.any(mask) else np.zeros_like(t)
    return out

f_i1, t_i1, S_i1 = compute_stft_power(current1, fs, STFT_NPERSEG, STFT_OVERLAP)
f_i2, t_i2, S_i2 = compute_stft_power(current2, fs, STFT_NPERSEG, STFT_OVERLAP)

def mask_band(f, band=MICRO_BAND):
    lo, hi = band
    return (f >= lo) & (f < hi)

mask1 = mask_band(f_i1, MICRO_BAND)
mask2 = mask_band(f_i2, MICRO_BAND)

be_time_i1 = band_energy_over_time(f_i1, t_i1, S_i1, freq_bands, nyquist)
be_time_i2 = band_energy_over_time(f_i2, t_i2, S_i2, freq_bands, nyquist)

# ======================= 滑动窗口统计（与 STFT 同步；至少一窗 + 单点可视） =======================
def windowed_stats_over_stft_grid(x, dt, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(x)
    nperseg = int(max(32, nperseg))
    if N <= 0:
        return np.array([]), {}
    segs, centers = [], []
    if N < nperseg:
        pad = nperseg - N
        x_pad = np.pad(x, (0, pad), mode='constant', constant_values=0.0)
        segs = [x_pad[:nperseg]]
        centers = [((0 + nperseg/2.0) * dt)]
    else:
        hop = int(max(1, nperseg * (1.0 - overlap_ratio)))
        nwin = 1 + (N - nperseg) // hop
        for k in range(nwin):
            i0 = k * hop
            i1 = i0 + nperseg
            seg = x[i0:i1]
            if seg.size < nperseg:
                break
            segs.append(seg)
            centers.append((i0 + nperseg/2.0) * dt)
    feats = {"rms": [], "kurtosis": [], "skewness": [], "crest_factor": []}
    for seg in segs:
        rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        kr  = float(kurtosis(seg)) if seg.size else 0.0
        sk  = float(skew(seg)) if seg.size else 0.0
        cf  = float((np.max(np.abs(seg)) / (rms if rms != 0 else 1e-12))) if seg.size else 0.0
        feats["rms"].append(rms)
        feats["kurtosis"].append(kr)
        feats["skewness"].append(sk)
        feats["crest_factor"].append(cf)
    t_centers = np.asarray(centers, dtype=float).reshape(-1)
    for k in feats:
        feats[k] = np.asarray(feats[k], dtype=float).reshape(-1)
    return t_centers, feats

t_win1, feats1 = windowed_stats_over_stft_grid(current1, dt, fs, STFT_NPERSEG, STFT_OVERLAP)
t_win2, feats2 = windowed_stats_over_stft_grid(current2, dt, fs, STFT_NPERSEG, STFT_OVERLAP)

print(f"[INFO] sliding windows: sample {sample_id_pair[0]} -> {len(t_win1)}, sample {sample_id_pair[1]} -> {len(t_win2)}")

# ======================= 绘图（图1~图4） =======================
eps = 1e-30
time1 = sample1['time_step'].values * dt
time2 = sample2['time_step'].values * dt

# 图1：时域 + FFT（频轴到 0–5 kHz；阴影 1–4.8 kHz）
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
ax_t = axes1[0]
ax_t.plot(time1, current1, label=f'Sample {sample_id_pair[0]} Current')
ax_t.plot(time2, current2, linestyle='--', label=f'Sample {sample_id_pair[1]} Current')
ax_t.set_title('Current Time Domain'); ax_t.set_xlabel('Time (s)'); ax_t.set_ylabel('Current (A)')
ax_t.grid(True); ax_t.legend()

ax_p = axes1[1]
ax_p.plot(freqs_i1, 10*np.log10(P_i1 + eps), label=f'Sample {sample_id_pair[0]} Current')
ax_p.plot(freqs_i2, 10*np.log10(P_i2 + eps), linestyle='--', label=f'Sample {sample_id_pair[1]} Current')
ax_p.set_title('Current Power Spectrum (dB)'); ax_p.set_xlabel('Frequency (Hz)'); ax_p.set_ylabel('Power (dB)')
ax_p.set_xlim(0, nyquist); ax_p.grid(True); ax_p.legend()
ax_p.axvspan(MICRO_BAND[0], MICRO_BAND[1], color='grey', alpha=0.15, label='1–4.8 kHz')

fig1.suptitle('Figure 1: Current (Time + Spectrum Focused on 1–4.8 kHz, fs=10 kHz)', fontsize=14)
plt.show()

# 图2：STFT（0–5 kHz；统一色阶靠 1–4.8 kHz）
stft_db_i1 = 10*np.log10(S_i1 + eps)
stft_db_i2 = 10*np.log10(S_i2 + eps)
band_vals = np.concatenate([stft_db_i1[mask1, :].ravel(), stft_db_i2[mask2, :].ravel()]) \
            if (np.any(mask1) and np.any(mask2)) else np.concatenate([stft_db_i1.ravel(), stft_db_i2.ravel()])
stft_vmin = np.percentile(band_vals, 5)
stft_vmax = np.percentile(band_vals, 95)

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
def _pcolor(ax, t, f, Z_db, title, vmin, vmax):
    if t.size==0 or f.size==0 or Z_db.size==0:
        ax.set_title(title + ' (empty)'); ax.set_axis_off(); return None
    im = ax.pcolormesh(t, f, Z_db, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
    ax.set_ylim(0, nyquist)
    return im
_ = _pcolor(axes2[0], t_i1, f_i1, stft_db_i1, f'Sample {sample_id_pair[0]} Current STFT (nperseg={STFT_NPERSEG})', stft_vmin, stft_vmax)
_ = _pcolor(axes2[1], t_i2, f_i2, stft_db_i2, f'Sample {sample_id_pair[1]} Current STFT (nperseg={STFT_NPERSEG})', stft_vmin, stft_vmax)
norm = Normalize(vmin=stft_vmin, vmax=stft_vmax)
sm = ScalarMappable(norm=norm, cmap=plt.get_cmap())
cbar = fig2.colorbar(sm, ax=axes2.ravel().tolist(), shrink=0.9, pad=0.02)
cbar.set_label('Power (dB)')
fig2.suptitle('Figure 2: Current STFT Spectrograms (0–5 kHz; focus on 1–4.8 kHz)', fontsize=14)
plt.show()

# 图3：分频带能量-时间
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
def _plot_be(ax, t, be_dict, title):
    if t.size == 0:
        ax.set_title(title + ' (empty)'); ax.set_axis_off(); return
    for name, y in be_dict.items():
        ax.plot(t, y, label=name)
    ax.set_title(title); ax.set_xlabel('Time (s)'); ax.set_ylabel('Band Energy (Relative)')
    ax.grid(True); ax.legend(fontsize=9, ncol=2)
_plot_be(axes3[0], t_i1, be_time_i1, f'Sample {sample_id_pair[0]} Current Band Energy (1–4.8 kHz)')
_plot_be(axes3[1], t_i2, be_time_i2, f'Sample {sample_id_pair[1]} Current Band Energy (1–4.8 kHz)')
fig3.suptitle('Figure 3: Band Energy vs Time (Current, 1–4.8 kHz)', fontsize=14)
plt.show()

# 图4：滑动窗口统计（单点可视）
def _plot_feat(ax, t, y, name, hop_time_guess=None):
    if t.size == 0 or y.size == 0:
        ax.set_title(name + ' (empty)'); ax.set_axis_off(); return
    ax.plot(t, y, marker='o', linewidth=1)
    ax.set_ylabel(name); ax.grid(True)
    if t.size == 1:
        pad = float(hop_time_guess or 0.01)
        ax.set_xlim(t[0] - pad, t[0] + pad)

hop_time = int(STFT_NPERSEG * (1.0 - STFT_OVERLAP)) / fs
fig4, axes4 = plt.subplots(4, 2, figsize=(18, 12), sharex='col', constrained_layout=True)
axes4[0,0].set_title(f"Sample {sample_id_pair[0]}"); axes4[0,1].set_title(f"Sample {sample_id_pair[1]}")
_plot_feat(axes4[0,0], t_win1, feats1.get("rms", np.array([])), "RMS", hop_time)
_plot_feat(axes4[0,1], t_win2, feats2.get("rms", np.array([])), "RMS", hop_time)
_plot_feat(axes4[1,0], t_win1, feats1.get("kurtosis", np.array([])), "Kurtosis", hop_time)
_plot_feat(axes4[1,1], t_win2, feats2.get("kurtosis", np.array([])), "Kurtosis", hop_time)
_plot_feat(axes4[2,0], t_win1, feats1.get("skewness", np.array([])), "Skewness", hop_time)
_plot_feat(axes4[2,1], t_win2, feats2.get("skewness", np.array([])), "Skewness", hop_time)
_plot_feat(axes4[3,0], t_win1, feats1.get("crest_factor", np.array([])), "Crest Factor", hop_time)
_plot_feat(axes4[3,1], t_win2, feats2.get("crest_factor", np.array([])), "Crest Factor", hop_time)
axes4[3,0].set_xlabel("Time (s)"); axes4[3,1].set_xlabel("Time (s)")
fig4.suptitle('Figure 4: Sliding-Window Statistics of Current (fs=10 kHz)', fontsize=14)
plt.show()

# 图5：电流 STFT 三维图（1–4.8 kHz）
def plot_stft_3d_current(T, F, Sxx, sample_id, band=MICRO_BAND, title_prefix="Current STFT Energy 3D"):
    lo, hi = band
    mask = (F >= lo) & (F < hi)
    if not np.any(mask) or T.size == 0:
        print(f"[警告] 样本 {sample_id} 的 STFT 数据在 {lo}-{hi} Hz 范围为空，跳过 3D 图。")
        return None
    F_band = F[mask]; S_band = Sxx[mask, :]
    f_step = max(1, int(len(F_band) / 128))
    t_step = max(1, int(len(T) / 256))
    F_ds = F_band[::f_step]; T_ds = T[::t_step]
    Z_ds = 10.0 * np.log10(S_band[::f_step, ::t_step] + 1e-30)
    TT, FF = np.meshgrid(T_ds, F_ds)
    fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(TT, FF, Z_ds, linewidth=0, antialiased=True, cmap=plt.get_cmap())
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_zlabel("Power (dB)")
    ax.set_title(f"{title_prefix} (Sample {sample_id}, 1–4.8 kHz, nperseg={STFT_NPERSEG})")
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Power (dB)")
    plt.tight_layout(); plt.show(); return fig

_ = plot_stft_3d_current(t_i1, f_i1, S_i1, sample_id_pair[0], MICRO_BAND, "Figure 5A: Current STFT Energy 3D")
_ = plot_stft_3d_current(t_i2, f_i2, S_i2, sample_id_pair[1], MICRO_BAND, "Figure 5B: Current STFT Energy 3D")

# 图6：1–4.8 kHz 电流差分异常检测（故障-健康）
def detect_diff_anomalies_band(t, f, S_fault, S_healthy, 
                               band=(1_000.0, 4_800.0), db_thresh=0.5, 
                               min_duration_sec=None, merge_gap_sec=None):
    """
    dB 差分 (Fault-Healthy) 只在给定频带内检测。
    min_duration_sec / merge_gap_sec 若为 None，则按 STFT hop 自动给出合理值：
      min_duration ≈ max(2*hop, 0.02s), merge_gap ≈ hop
    """
    eps = 1e-30
    Z_fault = 10*np.log10(S_fault + eps)
    Z_healthy = 10*np.log10(S_healthy + eps)
    Z_diff = Z_fault - Z_healthy
    if Z_diff.size == 0 or t.size == 0 or f.size == 0:
        print("[提示] 差分矩阵为空。"); return []
    lo, hi = band
    fmask = (f >= lo) & (f < hi)
    if not np.any(fmask):
        print(f"[提示] 无频点满足 {lo}–{hi} Hz。"); return []
    M = Z_diff[fmask, :] > db_thresh
    if not np.any(M):
        print(f"[提示] 无时刻满足 ΔPower > {db_thresh} dB（在 {int(lo)}–{int(hi)} Hz 内）。"); return []
    time_mask = M.any(axis=0)

    hop = (t[1] - t[0]) if t.size > 1 else 1.0
    if merge_gap_sec is None: merge_gap_sec = hop
    if min_duration_sec is None: min_duration_sec = max(2*hop, 0.02)  # 给 10kHz 一个更合理的默认门限

    frames = lambda sec: max(1, int(round(sec / hop)))
    max_gap = frames(merge_gap_sec); min_len = frames(min_duration_sec)

    idx = np.where(time_mask)[0]; events_idx = []
    if idx.size > 0:
        start = prev = idx[0]
        for k in idx[1:]:
            if (k - prev - 1) <= max_gap: prev = k
            else:
                if (prev - start + 1) >= min_len: events_idx.append((start, prev))
                start = prev = k
        if (prev - start + 1) >= min_len: events_idx.append((start, prev))

    events = []
    Fsel = f[fmask]; Zsel = Z_diff[fmask, :]
    for i0, i1 in events_idx:
        block = Zsel[:, i0:i1+1]
        pf, pt = np.unravel_index(np.argmax(block), block.shape)
        events.append({
            "start_t": float(t[i0]),
            "end_t": float(t[i1]),
            "duration": float(t[i1] - t[i0]),
            "peak_t": float(t[i0 + pt]),
            "peak_freq": float(Fsel[pf]),
            "peak_db": float(block[pf, pt]),
        })
    return events

def print_diff_events(events, tag, band=MICRO_BAND):
    print(f"\n=== 异常事件（{tag}） {int(band[0])}-{int(band[1])} Hz 且 Δ>0.5 dB ===")
    if not events:
        print("无。"); return
    for i, e in enumerate(events, 1):
        print(f"[{i}] {e['start_t']:.6f}s ~ {e['end_t']:.6f}s (dur={e['duration']*1e3:.2f} ms)  "
              f"peak@{e['peak_t']:.6f}s  {e['peak_freq']/1000:.3f} kHz  Δ={e['peak_db']:.2f} dB")

def plot_diff_spectrogram_band(t, f, S_fault, S_healthy, band=MICRO_BAND, title='Figure 6: Current Difference Spectrogram'):
    eps = 1e-30
    Z_fault = 10*np.log10(S_fault + eps)
    Z_healthy = 10*np.log10(S_healthy + eps)
    Z_diff = Z_fault - Z_healthy
    lo, hi = band
    fmask = (f >= lo) & (f < hi)
    if not np.any(fmask):
        print(f"[提示] 无频点满足 {lo}–{hi} Hz，无法绘制差分图。"); return
    Zb = Z_diff[fmask, :]
    vmin = np.percentile(Zb, 5); vmax = np.percentile(Zb, 95)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f[fmask], Zb, shading='auto', vmin=vmin, vmax=vmax)
    plt.title(title + f' (dB, {int(lo)}–{int(hi)} Hz, fs=10 kHz)')
    plt.ylabel('Frequency (Hz)'); plt.xlabel('Time (s)')
    plt.ylim(lo, hi); cb = plt.colorbar(); cb.set_label('Δ Power (dB)')
    plt.tight_layout(); plt.show()

# 假设较大的 sample_id 是“故障”
current_events = detect_diff_anomalies_band(t_i2, f_i2, S_i2, S_i1, MICRO_BAND, 0.5, None, None)
print_diff_events(current_events, "Current STFT Difference", MICRO_BAND)
plot_diff_spectrogram_band(t_i2, f_i2, S_i2, S_i1, MICRO_BAND)
