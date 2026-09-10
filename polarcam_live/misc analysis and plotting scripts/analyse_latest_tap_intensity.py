import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

rec = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\rod_x1872_y1556_20260722-122139_1784719299395138000")
out = rec / "tap_intensity_analysis"
out.mkdir(exist_ok=True)

def strip_marker(arr):
    if arr.ndim == 3 and arr.shape[0] > 1:
        last = arr[-1]
        finite = np.isfinite(last)
        if finite.sum() == 1 and np.nanmax(last) == 1.0:
            return arr[:-1]
    return arr

def channel_images(raw, roi):
    x = int(roi.get('x', 0)); y = int(roi.get('y', 0))
    phase_x = x % 2; phase_y = y % 2
    g = raw[:, :14, :14].astype(np.float64)
    return {
        'I0': g[:, phase_y::2, phase_x::2],
        'I45': g[:, phase_y::2, (1-phase_x)::2],
        'I135': g[:, (1-phase_y)::2, phase_x::2],
        'I90': g[:, (1-phase_y)::2, (1-phase_x)::2],
    }

def odd_window(n, limit):
    n = max(1, int(round(n)))
    if n % 2 == 0:
        n += 1
    return max(1, min(n, limit if limit % 2 == 1 else limit - 1))

def centered_crop_mean(stack, cx, cy, w):
    h, ww = stack.shape[1:]
    half = w // 2
    ix = max(half, min(ww - half - 1, int(round(cx))))
    iy = max(half, min(h - half - 1, int(round(cy))))
    patch = stack[:, iy-half:iy+half+1, ix-half:ix+half+1]
    return patch.mean(axis=(1,2)), patch.shape[1] * patch.shape[2]

def get_traces(rec):
    arr = strip_marker(np.load(rec/'capture_maxfps_15x15.npy', mmap_mode='r'))
    with open(rec/'capture_maxfps_15x15_meta.json', 'r') as f:
        meta = json.load(f)
    actual = meta.get('actual', {})
    roi = actual.get('roi', {})
    fps = float(actual.get('fps') or meta.get('fps') or 1640.0709219858156)
    raw11 = arr[:, :14, :14].astype(np.float64)[:,1:12,1:12].mean(axis=(1,2))
    chans = channel_images(arr, roi)
    req = meta.get('requested', {})
    center = req.get('center', {})
    cx = (int(center.get('x', int(roi.get('x',0))+7)) - int(roi.get('x',0))) / 2.0
    cy = (int(center.get('y', int(roi.get('y',0))+7)) - int(roi.get('y',0))) / 2.0
    cw = odd_window(float(req.get('window_raw', 11)) / 2.0, min(next(iter(chans.values())).shape[1:]))
    parts = []
    pix_total = 0
    for name in ['I0','I45','I135','I90']:
        tr, npix = centered_crop_mean(chans[name], cx, cy, cw)
        parts.append(tr * npix)
        pix_total += npix
    channel_mean = np.sum(parts, axis=0) / pix_total
    return fps, arr.shape[0], raw11, channel_mean, cw, pix_total

def stat(name, tr, fps):
    qs = np.percentile(tr, [1,5,50,95,99])
    mean = float(np.mean(tr))
    return {
        'trace': name,
        'mean': mean,
        'std': float(np.std(tr, ddof=1)),
        'min': float(np.min(tr)),
        'max': float(np.max(tr)),
        'range': float(np.max(tr)-np.min(tr)),
        'p1': float(qs[0]),
        'p5': float(qs[1]),
        'median': float(qs[2]),
        'p95': float(qs[3]),
        'p99': float(qs[4]),
        'p95_minus_p5': float(qs[3]-qs[1]),
        'p95_minus_p5_percent_mean': float((qs[3]-qs[1])/mean*100),
        'p99_minus_p1': float(qs[4]-qs[0]),
        'p99_minus_p1_percent_mean': float((qs[4]-qs[0])/mean*100),
        'n_frames': int(len(tr)),
        'duration_s': float(len(tr)/fps),
    }

fps, n, raw11, channel_mean, cw, pix_total = get_traces(rec)
time = np.arange(n)/fps
rows = [stat('raw11_mean', raw11, fps), stat('channel_mean', channel_mean, fps)]
fields = list(rows[0].keys())
with open(out/'tap_intensity_stats.csv','w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(out/'tap_intensity_stats.json','w') as f:
    json.dump({'recording':str(rec), 'fps':fps, 'n_frames':n, 'window_channel_width':cw, 'channel_total_px':pix_total, 'stats':rows}, f, indent=2)

fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
ax.plot(time, channel_mean, lw=0.8, label='angle-window mean')
ax.plot(time, raw11, lw=0.7, alpha=0.6, label='raw central 11x11 mean')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity')
ax.set_title('Tap recording intensity trace')
ax.grid(alpha=0.25); ax.legend()
fig.savefig(out/'tap_intensity_trace.png', dpi=200); plt.close(fig)

# Spectrum for context.
y = channel_mean - channel_mean.mean()
f, pxx = welch(y, fs=fps, nperseg=min(len(y), 2048), noverlap=min(1024, min(len(y),2048)//2), scaling='density')
fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
ax.semilogy(f, pxx, lw=0.9)
ax.set_xlim(1, min(820, fps/2)); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD')
ax.set_title('Tap recording spectrum, angle-window mean intensity')
ax.grid(alpha=0.25)
fig.savefig(out/'tap_intensity_spectrum.png', dpi=200); plt.close(fig)

peak_idx = np.argmax(pxx[(f>=1)&(f<=fps/2)])
valid = np.flatnonzero((f>=1)&(f<=fps/2))
peak = valid[peak_idx]
print('OUTPUT_DIR', out)
print('fps', fps, 'frames', n, 'duration', n/fps)
for r in rows:
    print(r['trace'], 'mean', round(r['mean'],3), 'p95-p5', round(r['p95_minus_p5'],3), 'pct', round(r['p95_minus_p5_percent_mean'],3), 'p99-p1 pct', round(r['p99_minus_p1_percent_mean'],3), 'range pct', round(r['range']/r['mean']*100,3), 'std', round(r['std'],3))
print('strongest spectrum peak Hz', round(float(f[peak]),3), 'PSD', float(pxx[peak]))
