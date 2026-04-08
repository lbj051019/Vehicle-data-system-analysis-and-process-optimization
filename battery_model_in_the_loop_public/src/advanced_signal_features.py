# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import stft

# ======================= 基本设置（更新后） =======================
dt = 1e-4               # 采样间隔：100 微秒
fs = 10_000.0           # 采样率：10 kHz（与 1/dt 一致）
nyquist = fs / 2.0      # 奈奎斯特频率：5 kHz

# ======================= 时频频带（更新后，单位 Hz） =======================
freq_bands = {
    "0-100Hz":     (0,     100),
    "100-300Hz":   (100,   300),
    "300-1kHz":    (300,  1_000),
    "1k-2kHz":     (1_000, 2_000),
    "2k-3.5kHz":   (2_000, 3_500),
    "3.5k-5kHz":   (3_500, 5_000),
}

# ======================= STFT 参数（更新后） =======================
STFT_NPERSEG = 512      # ≈51.2 ms；频率分辨率 ≈ 19.5 Hz
STFT_OVERLAP = 0.90     # hop ≈ 5.12 ms

# ======================= CWT 参数（更新后） =======================
CWT_FMIN = 20.0
CWT_FMAX = 4_500.0
CWT_SCALES = 96

# ======================= 数值积分（修复 trapz 弃用） =======================
try:
    from scipy.integrate import trapezoid as TRAPEZOID
except Exception:
    TRAPEZOID = getattr(np, "trapezoid", None)
    if TRAPEZOID is None:
        TRAPEZOID = np.trapz  # 旧版兜底（可能有 warning）

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

        # 成对 sample_id（0-99 与 100-199 配对）
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

# ======================= 时域统计特征 =======================
def calculate_statistical_features(signal):
    signal = np.asarray(signal)
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))  # 均方根
    features['peak_to_peak'] = np.max(signal) - np.min(signal)  # 峰峰值
    features['skewness'] = skew(signal)  # 偏度
    features['kurtosis'] = kurtosis(signal)  # 峭度
    # 波形指标（防0处理）
    rms = features['rms'] if features['rms'] != 0 else 1e-12
    mean_abs = np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 1e-12
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal))) if np.mean(np.sqrt(np.abs(signal))) != 0 else 1e-12
    features['crest_factor'] = np.max(np.abs(signal)) / rms
    features['impulse_factor'] = np.max(np.abs(signal)) / mean_abs
    features['margin_factor'] = np.max(np.abs(signal)) / (mean_sqrt_abs**2)
    return features

voltage1 = sample1['voltage'].values
current1 = sample1['current'].values
voltage2 = sample2['voltage'].values
current2 = sample2['current'].values

voltage_features_sample1 = calculate_statistical_features(voltage1)
current_features_sample1 = calculate_statistical_features(current1)
voltage_features_sample2 = calculate_statistical_features(voltage2)
current_features_sample2 = calculate_statistical_features(current2)

def print_features(title, feats):
    print(f"\n{title}")
    for k, v in feats.items():
        print(f"{k}: {v:.4f}")

print_features(f"Sample 1 (ID: {sample_id_pair[0]}) Voltage Features:", voltage_features_sample1)
print_features(f"Sample 1 (ID: {sample_id_pair[0]}) Current Features:", current_features_sample1)
print_features(f"Sample 2 (ID: {sample_id_pair[1]}) Voltage Features:", voltage_features_sample2)
print_features(f"Sample 2 (ID: {sample_id_pair[1]}) Current Features:", current_features_sample2)

# ======================= 全局频谱（FFT 功率谱） =======================
def compute_power_spectrum(signal, dt=dt):
    """
    返回:
      freqs: 单边频率轴 (rfft)
      power: 单边功率谱（含2x，DC/奈奎斯特修正，窗功率校正）
      df   : 频率分辨率
      total_energy: 总能量（∑power·df）
    """
    x = np.asarray(signal, dtype=float)
    N = len(x)
    if N == 0:
        raise ValueError("输入信号长度为 0")

    # 去均值 + Hann窗
    x = x - np.mean(x)
    w = np.hanning(N)
    xw = x * w
    U = np.mean(w**2)   # 窗功率平均值

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=dt)

    power = (np.abs(X)**2) * 2.0 / ((N**2) * U)
    power[0] *= 0.5
    if N % 2 == 0:
        power[-1] *= 0.5

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
    total_energy = float(np.sum(power) * (df if df > 0 else 1.0))
    return freqs, power, df, total_energy

freqs_v1, P_v1, df1, _ = compute_power_spectrum(voltage1, dt)
freqs_i1, P_i1, _,   _ = compute_power_spectrum(current1,  dt)
freqs_v2, P_v2, df2, _ = compute_power_spectrum(voltage2, dt)
freqs_i2, P_i2, _,   _ = compute_power_spectrum(current2,  dt)

# ======================= STFT：短时傅里叶变换 =======================
def compute_stft_power(signal, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP):
    """
    返回:
      f: 频率轴 (Hz) [F,]
      t: 时间轴 (s)   [T,]
      Sxx: 功率谱 |Zxx|^2 [F, T]
    具备鲁棒性：清洗 NaN/Inf、限制 nperseg、零填充以避免空白图。
    """
    x = np.asarray(signal, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    N = len(x)
    if N < 2:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1))

    nperseg = int(nperseg)
    nperseg = max(32, nperseg)
    nperseg = min(nperseg, N)
    noverlap = int(np.clip(nperseg * overlap_ratio, 0, nperseg - 1))

    # 允许零填充，防止空窗
    boundary = 'zeros'
    padded = True

    f, t, Zxx = stft(x, fs=fs, window='hann',
                     nperseg=nperseg, noverlap=noverlap,
                     return_onesided=True, boundary=boundary, padded=padded)

    Sxx = np.abs(Zxx) ** 2
    Sxx = np.nan_to_num(Sxx, nan=0.0, posinf=0.0, neginf=0.0)
    return f, t, Sxx

def band_energy_over_time(f, t, Sxx, bands=freq_bands, nyquist=nyquist):
    """
    对每个时间窗，将频带内功率对频率积分（trapezoid），得到能量随时间的曲线。
    返回 dict{name: [T,]}
    """
    out = {}
    for name, (lo, hi) in bands.items():
        lo_c, hi_c = max(0.0, lo), min(nyquist, hi)
        if f.size == 0 or t.size == 0 or hi_c <= lo_c:
            out[name] = np.zeros_like(t)
            continue
        mask = (f >= lo_c) & (f < hi_c)
        if np.any(mask):
            out[name] = TRAPEZOID(Sxx[mask, :], f[mask], axis=0)
        else:
            out[name] = np.zeros_like(t)
    return out

f_v1, t_v1, S_v1 = compute_stft_power(voltage1, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP)
f_i1, t_i1, S_i1 = compute_stft_power(current1, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP)
f_v2, t_v2, S_v2 = compute_stft_power(voltage2, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP)
f_i2, t_i2, S_i2 = compute_stft_power(current2, fs, nperseg=STFT_NPERSEG, overlap_ratio=STFT_OVERLAP)

be_time_v1 = band_energy_over_time(f_v1, t_v1, S_v1, freq_bands, nyquist)
be_time_i1 = band_energy_over_time(f_i1, t_i1, S_i1, freq_bands, nyquist)
be_time_v2 = band_energy_over_time(f_v2, t_v2, S_v2, freq_bands, nyquist)
be_time_i2 = band_energy_over_time(f_i2, t_i2, S_i2, freq_bands, nyquist)

# ======================= CWT：连续小波变换（可选） =======================
HAS_PYWT = True
try:
    import pywt
except Exception:
    HAS_PYWT = False
    print("\n[提示] 未检测到 PyWavelets (pywt)，将跳过 CWT。可通过 `pip install pywavelets` 安装。")

def compute_cwt_morl(signal, dt=dt, fmin=CWT_FMIN, fmax=CWT_FMAX, num_scales=CWT_SCALES):
    """
    使用 PyWavelets 的 'morl' 实 Morlet 小波做 CWT。
    返回：
      t_vec: 时间轴（秒）
      freqs: 频率轴（Hz）
      power: |coeff|^2 的功率（shape: [num_scales, N]）
    """
    x = np.asarray(signal, dtype=float) - np.mean(signal)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(x)
    t_vec = np.arange(N) * dt

    wavelet = 'morl'
    freqs_target = np.geomspace(fmax, fmin, num_scales)  # 高->低
    fc = pywt.central_frequency(wavelet)
    scales = (fc / (freqs_target * dt))
    coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    freqs = pywt.scale2frequency(wavelet, scales) / dt
    return t_vec, freqs, power

if HAS_PYWT:
    t_vec_v1, freqs_cwt_v1, P_cwt_v1 = compute_cwt_morl(voltage1, dt=dt)
    t_vec_v2, freqs_cwt_v2, P_cwt_v2 = compute_cwt_morl(voltage2, dt=dt)
    t_vec_i1, freqs_cwt_i1, P_cwt_i1 = compute_cwt_morl(current1, dt=dt)
    t_vec_i2, freqs_cwt_i2, P_cwt_i2 = compute_cwt_morl(current2, dt=dt)

# ======================= 绘图 =======================
eps = 1e-30
time1 = sample1['time_step'].values * dt
time2 = sample2['time_step'].values * dt

# ---------- 图1：时域 + 全局功率谱 ----------
fig1, axes1 = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

# 时域
ax_t1, ax_t2 = axes1[0, 0], axes1[0, 1]
ax_t1.plot(time1, voltage1, label=f'Sample {sample_id_pair[0]} Voltage')
ax_t1.plot(time2, voltage2, linestyle='--', label=f'Sample {sample_id_pair[1]} Voltage')
ax_t1.set_title('Voltage Time Domain'); ax_t1.set_xlabel('Time (s)'); ax_t1.set_ylabel('Voltage (V)')
ax_t1.grid(True); ax_t1.legend()

ax_t2.plot(time1, current1, label=f'Sample {sample_id_pair[0]} Current')
ax_t2.plot(time2, current2, linestyle='--', label=f'Sample {sample_id_pair[1]} Current')
ax_t2.set_title('Current Time Domain'); ax_t2.set_xlabel('Time (s)'); ax_t2.set_ylabel('Current (A)')
ax_t2.grid(True); ax_t2.legend()

# FFT（功率谱，dB）
ax_p1, ax_p2 = axes1[1, 0], axes1[1, 1]
ax_p1.plot(freqs_v1, 10*np.log10(P_v1 + eps), label=f'Sample {sample_id_pair[0]} Voltage')
ax_p1.plot(freqs_v2, 10*np.log10(P_v2 + eps), linestyle='--', label=f'Sample {sample_id_pair[1]} Voltage')
ax_p1.set_title('Voltage Power Spectrum (Global, dB)'); ax_p1.set_xlabel('Frequency (Hz)'); ax_p1.set_ylabel('Power (dB)')
ax_p1.set_xlim(0, nyquist); ax_p1.grid(True); ax_p1.legend()

ax_p2.plot(freqs_i1, 10*np.log10(P_i1 + eps), label=f'Sample {sample_id_pair[0]} Current')
ax_p2.plot(freqs_i2, 10*np.log10(P_i2 + eps), linestyle='--', label=f'Sample {sample_id_pair[1]} Current')
ax_p2.set_title('Current Power Spectrum (Global, dB)'); ax_p2.set_xlabel('Frequency (Hz)'); ax_p2.set_ylabel('Power (dB)')
ax_p2.set_xlim(0, nyquist); ax_p2.grid(True); ax_p2.legend()

fig1.suptitle('Figure 1: Time Domain + Global Power Spectrum', fontsize=14)
plt.show()

# ---------- 图2：STFT 时频图（统一颜色条） ----------
stft_db_arrays = [
    10*np.log10(S_v1 + eps), 10*np.log10(S_v2 + eps),
    10*np.log10(S_i1 + eps), 10*np.log10(S_i2 + eps)
]
stft_vmin = np.percentile(np.concatenate([a.ravel() for a in stft_db_arrays]), 5)
stft_vmax = np.percentile(np.concatenate([a.ravel() for a in stft_db_arrays]), 95)

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

def _pcolor(ax, t, f, Z_db, title, vmin, vmax):
    if t.size==0 or f.size==0 or Z_db.size==0:
        ax.set_title(title + ' (empty)'); ax.set_axis_off(); return None
    im = ax.pcolormesh(t, f, Z_db, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
    ax.set_ylim(0, nyquist)
    return im

_pcolor(axes2[0, 0], t_v1, f_v1, stft_db_arrays[0], f'Sample {sample_id_pair[0]} Voltage STFT', stft_vmin, stft_vmax)
_pcolor(axes2[0, 1], t_v2, f_v2, stft_db_arrays[1], f'Sample {sample_id_pair[1]} Voltage STFT', stft_vmin, stft_vmax)
_pcolor(axes2[1, 0], t_i1, f_i1, stft_db_arrays[2], f'Sample {sample_id_pair[0]} Current STFT', stft_vmin, stft_vmax)
_pcolor(axes2[1, 1], t_i2, f_i2, stft_db_arrays[3], f'Sample {sample_id_pair[1]} Current STFT', stft_vmin, stft_vmax)

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
norm = Normalize(vmin=stft_vmin, vmax=stft_vmax)
sm = ScalarMappable(norm=norm, cmap=plt.get_cmap())
cbar = fig2.colorbar(sm, ax=axes2.ravel().tolist(), shrink=0.9, pad=0.02)
cbar.set_label('Power (dB)')

fig2.suptitle('Figure 2: STFT Spectrograms', fontsize=14)
plt.show()

# ---------- 图3：分频带能量-时间 ----------
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

def _plot_be(ax, t, be_dict, title):
    if t.size == 0:
        ax.set_title(title + ' (empty)'); ax.set_axis_off(); return
    for name, y in be_dict.items():
        ax.plot(t, y, label=name)
    ax.set_title(title); ax.set_xlabel('Time (s)'); ax.set_ylabel('Band Energy')
    ax.grid(True); ax.legend(fontsize=8, ncol=2)

_plot_be(axes3[0, 0], t_v1, be_time_v1, f'Sample {sample_id_pair[0]} Voltage Band Energy')
_plot_be(axes3[0, 1], t_v2, be_time_v2, f'Sample {sample_id_pair[1]} Voltage Band Energy')
_plot_be(axes3[1, 0], t_i1, be_time_i1, f'Sample {sample_id_pair[0]} Current Band Energy')
_plot_be(axes3[1, 1], t_i2, be_time_i2, f'Sample {sample_id_pair[1]} Current Band Energy')

fig3.suptitle('Figure 3: Band Energy vs Time (from STFT)', fontsize=14)
plt.show()

# ---------- 图4：CWT（若未安装 pywt 则跳过） ----------
if HAS_PYWT:
    cwt_db_arrays = [
        10*np.log10(P_cwt_v1 + eps), 10*np.log10(P_cwt_v2 + eps),
        10*np.log10(P_cwt_i1 + eps), 10*np.log10(P_cwt_i2 + eps)
    ]
    cwt_vmin = np.percentile(np.concatenate([a.ravel() for a in cwt_db_arrays]), 5)
    cwt_vmax = np.percentile(np.concatenate([a.ravel() for a in cwt_db_arrays]), 95)

    fig4, axes4 = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    def _pcolor_cwt(ax, t, f, P_db, title, vmin, vmax):
        if P_db.size==0:
            ax.set_title(title + ' (empty)'); ax.set_axis_off(); return None
        im = ax.pcolormesh(t, f, P_db, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
        ax.set_ylim(0, nyquist)
        return im

    _pcolor_cwt(axes4[0, 0], t_vec_v1, freqs_cwt_v1, cwt_db_arrays[0], f'Sample {sample_id_pair[0]} Voltage CWT', cwt_vmin, cwt_vmax)
    _pcolor_cwt(axes4[0, 1], t_vec_v2, freqs_cwt_v2, cwt_db_arrays[1], f'Sample {sample_id_pair[1]} Voltage CWT', cwt_vmin, cwt_vmax)
    _pcolor_cwt(axes4[1, 0], t_vec_i1, freqs_cwt_i1, cwt_db_arrays[2], f'Sample {sample_id_pair[0]} Current CWT', cwt_vmin, cwt_vmax)
    _pcolor_cwt(axes4[1, 1], t_vec_i2, freqs_cwt_i2, cwt_db_arrays[3], f'Sample {sample_id_pair[1]} Current CWT', cwt_vmin, cwt_vmax)

    norm2 = Normalize(vmin=cwt_vmin, vmax=cwt_vmax)
    sm2 = ScalarMappable(norm=norm2, cmap=plt.get_cmap())
    cbar2 = fig4.colorbar(sm2, ax=axes4.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar2.set_label('Power (dB)')

    fig4.suptitle('Figure 4: CWT (Morlet) Scalograms', fontsize=14)
    plt.show()
else:
    print("[提示] 未安装 pywt，已跳过图4（CWT）。如需 CWT，请先安装：pip install pywavelets")

# === 差分时频图与自动异常检测（适配 10 kHz 采样） ===
def detect_diff_anomalies(t, f, S_fault, S_healthy, 
                          f_min=1_000.0,
                          db_thresh=1.0,
                          min_duration_sec=0.02,
                          merge_gap_sec=0.01):
    """
    输入:
      t, f: STFT 的时间/频率轴
      S_fault, S_healthy: |Zxx|^2（功率谱），形状 [F, T]
    返回:
      events: [{start_t, end_t, peak_t, peak_freq, peak_db, duration}]
    """
    eps_local = 1e-30
    Z_fault = 10*np.log10(S_fault + eps_local)
    Z_healthy = 10*np.log10(S_healthy + eps_local)
    Z_diff = Z_fault - Z_healthy  # [F, T]

    if Z_diff.size == 0 or t.size == 0 or f.size == 0:
        print("[提示] 差分矩阵为空。"); 
        return []

    fmask = (f >= f_min)                          # [F,]
    if not np.any(fmask):
        print(f"[提示] 无频点满足 f >= {f_min} Hz。")
        return []

    M = Z_diff[fmask, :] > db_thresh              # [F_sel, T]
    if not np.any(M):
        print(f"[提示] 无时刻满足 ΔPower > {db_thresh} dB（在 f>={f_min}Hz 条件下）。")
        return []

    time_mask = M.any(axis=0)                     # [T,]

    hop = (t[1] - t[0]) if t.size > 1 else 1.0
    def frames(sec): 
        return max(1, int(round(sec / hop)))
    max_gap = frames(merge_gap_sec)
    min_len = frames(min_duration_sec)

    idx = np.where(time_mask)[0]
    events_idx = []
    if idx.size > 0:
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if (k - prev - 1) <= max_gap:
                prev = k
            else:
                if (prev - start + 1) >= min_len:
                    events_idx.append((start, prev))
                start = k
                prev = k
        if (prev - start + 1) >= min_len:
            events_idx.append((start, prev))

    events = []
    Fsel = f[fmask]
    Zsel = Z_diff[fmask, :]  # [F_sel, T]
    for i0, i1 in events_idx:
        block = Zsel[:, i0:i1+1]                 # [F_sel, T_seg]
        flat_idx = np.argmax(block)
        pf, pt = np.unravel_index(flat_idx, block.shape)
        peak_db = float(block[pf, pt])
        peak_t = float(t[i0 + pt])
        peak_freq = float(Fsel[pf])
        events.append({
            "start_t": float(t[i0]),
            "end_t": float(t[i1]),
            "duration": float(t[i1] - t[i0]),
            "peak_t": peak_t,
            "peak_freq": peak_freq,
            "peak_db": peak_db,
        })
    return events

def print_diff_events(events, tag):
    print(f"\n=== 异常事件（{tag}） f>=1kHz 且 Δ>{1.0} dB ===")
    if not events:
        print("无。"); 
        return
    for i, e in enumerate(events, 1):
        print(f"[{i}] {e['start_t']:.6f}s ~ {e['end_t']:.6f}s "
              f"(dur={e['duration']*1e3:.2f} ms)  "
              f"peak@{e['peak_t']:.6f}s  {e['peak_freq']/1000:.3f} kHz  "
              f"Δ={e['peak_db']:.2f} dB")

# 差分时频图（Fault - Healthy）
def plot_diff_spectrogram(t, f, S_fault, S_healthy, title):
    Z_fault = 10*np.log10(S_fault + eps)
    Z_healthy = 10*np.log10(S_healthy + eps)
    Z_diff = Z_fault - Z_healthy           # dB 差分
    vmin = np.percentile(Z_diff, 5)
    vmax = np.percentile(Z_diff, 95)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, Z_diff, shading='auto', vmin=vmin, vmax=vmax)
    plt.title(title + ' (Fault - Healthy, dB)')
    plt.ylabel('Frequency (Hz)'); plt.xlabel('Time (s)')
    plt.ylim(0, nyquist); cb = plt.colorbar(); cb.set_label('Δ Power (dB)')
    plt.tight_layout(); plt.show()

# 假设较大的 sample_id 是“故障”、较小的是“健康”（若相反请对调）
plot_diff_spectrogram(t_v2, f_v2, S_v2, S_v1, 'Voltage STFT Difference')
plot_diff_spectrogram(t_i2, f_i2, S_i2, S_i1, 'Current STFT Difference')

voltage_events = detect_diff_anomalies(
    t_v2, f_v2, S_v2, S_v1,
    f_min=1_000.0, db_thresh=1.0,
    min_duration_sec=0.02, merge_gap_sec=0.01
)
current_events = detect_diff_anomalies(
    t_i2, f_i2, S_i2, S_i1,
    f_min=1_000.0, db_thresh=1.0,
    min_duration_sec=0.02, merge_gap_sec=0.01
)

print_diff_events(voltage_events, "Voltage STFT Difference")
print_diff_events(current_events, "Current STFT Difference")

# ================= 新增功能 1：电流 T–F–E 三维图 =================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 触发3D支持

def plot_tfe_3d(t, f, Sxx, title):
    """
    t: [T,] 秒；f: [F,] Hz；Sxx: [F, T]（|Zxx|^2）
    """
    if t.size == 0 or f.size == 0 or Sxx.size == 0:
        print(f"[提示] {title} 数据为空，跳过3D图。")
        return
    T, F = np.meshgrid(t, f)              # X=时间, Y=频率
    Z = 10*np.log10(Sxx + eps)            # dB 更易观察动态范围

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    # 适度稀疏以提速
    skip_t = max(1, int(len(t) / 400))
    skip_f = max(1, int(len(f) / 300))
    ax.plot_surface(
        T[::skip_f, ::skip_t], F[::skip_f, ::skip_t], Z[::skip_f, ::skip_t],
        linewidth=0, antialiased=False, shade=True
    )
    ax.set_title(title)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)'); ax.set_zlabel('Power (dB)')
    ax.set_zlim(np.percentile(Z, 5), np.percentile(Z, 95))
    plt.tight_layout(); plt.show()

# 两组电流 3D 图
plot_tfe_3d(t_i1, f_i1, S_i1, f'Current T–F–E 3D (Sample {sample_id_pair[0]})')
plot_tfe_3d(t_i2, f_i2, S_i2, f'Current T–F–E 3D (Sample {sample_id_pair[1]})')

# ============ 新增功能 2：分频段能量“占比”对比图 ============
def band_energy_total(f, t, Sxx, bands):
    """
    对每个频段：先对频率积分 -> 各时刻能量；再对时间积分 -> 频段总能量
    返回 dict{name: scalar_energy}
    """
    be_t = band_energy_over_time(f, t, Sxx, bands=bands, nyquist=nyquist)
    totals = {}
    for k, y in be_t.items():
        totals[k] = float(TRAPEZOID(y, t)) if t.size > 1 else float(np.sum(y))
    return totals

def plot_band_share_compare(f1, t1, S1, f2, t2, S2, bands, title, legend_pair):
    """
    计算两组样本在各频段的能量占比并以并列柱状图展示
    legend_pair: ('Sample A', 'Sample B')
    """
    totals1 = band_energy_total(f1, t1, S1, bands)
    totals2 = band_energy_total(f2, t2, S2, bands)

    band_names = list(bands.keys())
    v1 = np.array([totals1.get(b, 0.0) for b in band_names])
    v2 = np.array([totals2.get(b, 0.0) for b in band_names])

    p1 = v1 / (v1.sum() + eps)
    p2 = v2 / (v2.sum() + eps)

    idx = np.arange(len(band_names))
    w = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(idx - w/2, p1, width=w, label=f'{legend_pair[0]}')
    plt.bar(idx + w/2, p2, width=w, label=f'{legend_pair[1]}')
    plt.xticks(idx, band_names, rotation=20)
    plt.ylabel('Energy Share'); plt.title(title)
    plt.ylim(0, 1.0); plt.grid(axis='y', alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

# 电压占比对比
plot_band_share_compare(
    f_v1, t_v1, S_v1, f_v2, t_v2, S_v2,
    bands=freq_bands,
    title='Voltage: Band Energy Share Comparison',
    legend_pair=(f'Sample {sample_id_pair[0]}', f'Sample {sample_id_pair[1]}')
)
# 电流占比对比
plot_band_share_compare(
    f_i1, t_i1, S_i1, f_i2, t_i2, S_i2,
    bands=freq_bands,
    title='Current: Band Energy Share Comparison',
    legend_pair=(f'Sample {sample_id_pair[0]}', f'Sample {sample_id_pair[1]}')
)

# ======== 新增功能 3：滑动窗口统计特征图（电流，两样本） ========
def sliding_features(signal, fs, window_sec=0.10, hop_sec=0.05):
    """
    计算滑动窗口统计：mean, std, rms, skewness, kurtosis, crest_factor
    返回：t_mid, feats_dict
    """
    x = np.asarray(signal, float)
    N = len(x)
    win = max(2, int(round(window_sec * fs)))
    hop = max(1, int(round(hop_sec * fs)))
    if N < win:
        return np.array([]), {}

    t_mid = []
    means, stds, rmss, skews, kurts, crests = [], [], [], [], [], []
    for start in range(0, N - win + 1, hop):
        seg = x[start:start+win]
        m = seg.mean()
        s = seg.std()
        r = np.sqrt((seg**2).mean())
        skv = skew(seg)
        ku = kurtosis(seg)
        cf = (np.max(np.abs(seg)) / (r if r != 0 else 1e-12))
        t_mid.append((start + win/2) / fs)
        means.append(m); stds.append(s); rmss.append(r); skews.append(skv); kurts.append(ku); crests.append(cf)

    feats = {
        'mean': np.array(means),
        'std': np.array(stds),
        'rms': np.array(rmss),
        'skewness': np.array(skews),
        'kurtosis': np.array(kurts),
        'crest_factor': np.array(crests)
    }
    return np.array(t_mid), feats

# 计算两组电流的滑窗特征（默认窗长 0.10 s、步长 0.05 s）
t_feat1, feats1 = sliding_features(current1, fs, window_sec=0.10, hop_sec=0.05)
t_feat2, feats2 = sliding_features(current2, fs, window_sec=0.10, hop_sec=0.05)

def plot_sliding_features(t1, f1, t2, f2, title_prefix):
    """
    将两组样本的 6 个统计特征按 3x2 子图绘制
    """
    if t1.size == 0 or t2.size == 0:
        print("[提示] 滑动窗口长度大于数据长度，未生成特征图。可减小 window_sec 或增大数据长度。")
        return
    keys = ['mean', 'std', 'rms', 'skewness', 'kurtosis', 'crest_factor']
    nrow, ncol = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    for i, k in enumerate(keys):
        axes[i].plot(t1, f1[k], label=f'Sample {sample_id_pair[0]}')
        axes[i].plot(t2, f2[k], linestyle='--', label=f'Sample {sample_id_pair[1]}')
        axes[i].set_title(k); axes[i].set_xlabel('Time (s)'); axes[i].grid(True)
        if i == 0:
            axes[i].legend()
    fig.suptitle(f'{title_prefix}: Sliding-Window Stats (win=0.10s, hop=0.05s)', fontsize=14)
    plt.show()

plot_sliding_features(t_feat1, feats1, t_feat2, feats2, title_prefix='Current')

# =========（可选）参数建议=========
# - 瞬态优先：STFT_NPERSEG=256（≈25.6ms），OVERLAP=0.9；sliding win=0.05s, hop=0.02s
# - 细频优先：STFT_NPERSEG=1024（≈102.4ms），OVERLAP=0.9；sliding win=0.20s, hop=0.05s
