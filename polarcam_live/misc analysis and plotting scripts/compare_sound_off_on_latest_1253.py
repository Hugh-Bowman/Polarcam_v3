import json, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

base = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending")
recs = [
    ("sound_off", base / "rod_x1663_y1398_20260722-125303_1784721183269833400"),
    ("sound_on", base / "rod_x1663_y1398_20260722-125314_1784721194140322900"),
]
out = base / "sound_off_vs_on_20260722-125303_125314_analysis"
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
    px = x % 2; py = y % 2
    g = raw[:, :14, :14].astype(np.float64)
    return {
        'I0': g[:, py::2, px::2],
        'I45': g[:, py::2, (1-px)::2],
        'I135': g[:, (1-py)::2, px::2],
        'I90': g[:, (1-py)::2, (1-px)::2],
    }

def odd_window(n, limit):
    n = max(1, int(round(n)))
    if n % 2 == 0: n += 1
    return max(1, min(n, limit if limit % 2 == 1 else limit - 1))

def centered_crop_mean(stack, cx, cy, w):
    h, ww = stack.shape[1:]
    half = w // 2
    ix = max(half, min(ww - half - 1, int(round(cx))))
    iy = max(half, min(h - half - 1, int(round(cy))))
    patch = stack[:, iy-half:iy+half+1, ix-half:ix+half+1]
    return patch.mean(axis=(1,2)), patch.shape[1] * patch.shape[2]

def traces(rec):
    with open(rec/'capture_maxfps_15x15_meta.json', 'r') as f:
        meta = json.load(f)
    arr = strip_marker(np.load(rec/'capture_maxfps_15x15.npy', mmap_mode='r'))
    actual = meta.get('actual', {})
    fps = float(actual.get('fps') or meta.get('fps') or 1640.0709219858156)
    roi = actual.get('roi', {})
    raw11 = arr[:, :14, :14].astype(np.float64)[:, 1:12, 1:12].mean(axis=(1,2))
    chans = channel_images(arr, roi)
    req = meta.get('requested', {})
    center = req.get('center', {})
    cx = (int(center.get('x', int(roi.get('x',0))+7)) - int(roi.get('x',0))) / 2.0
    cy = (int(center.get('y', int(roi.get('y',0))+7)) - int(roi.get('y',0))) / 2.0
    cw = odd_window(float(req.get('window_raw', 11)) / 2.0, min(next(iter(chans.values())).shape[1:]))
    parts=[]; pix_total=0
    for name in ['I0','I45','I135','I90']:
        tr,npix=centered_crop_mean(chans[name],cx,cy,cw)
        parts.append(tr*npix); pix_total += npix
    channel_mean = np.sum(parts, axis=0)/pix_total
    return {'fps':fps,'n_frames':int(arr.shape[0]),'duration_s':float(arr.shape[0]/fps),'raw11_mean':raw11,'channel_mean':channel_mean,'channel_total_px':int(pix_total)}

def stat_row(label, trace, tr, fps):
    p1,p5,p50,p95,p99=np.percentile(tr,[1,5,50,95,99])
    mean=float(np.mean(tr))
    return {'label':label,'trace':trace,'mean':mean,'std':float(np.std(tr,ddof=1)),'min':float(np.min(tr)),'max':float(np.max(tr)),'range':float(np.max(tr)-np.min(tr)),'p1':float(p1),'p5':float(p5),'median':float(p50),'p95':float(p95),'p99':float(p99),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'n_frames':int(len(tr)),'duration_s':float(len(tr)/fps)}

def spectrum(tr,fps):
    y=tr-np.mean(tr); nperseg=min(len(y),2048)
    return welch(y,fs=fps,nperseg=nperseg,noverlap=min(nperseg//2,1024),scaling='density')

def band_peak(f,pxx,lo,hi):
    m=(f>=lo)&(f<=hi); idxs=np.flatnonzero(m)
    if not len(idxs): return math.nan, math.nan
    idx=idxs[np.argmax(pxx[m])]
    return float(f[idx]), float(pxx[idx])

def nearest(f,pxx,freq):
    idx=int(np.argmin(np.abs(f-freq)))
    return float(f[idx]), float(pxx[idx])

data=[(label,rec,traces(rec)) for label,rec in recs]
rows=[]
for label,rec,t in data:
    for name in ['raw11_mean','channel_mean']:
        rows.append(stat_row(label,name,t[name],t['fps']))
with open(out/'intensity_range_comparison.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(out/'intensity_range_comparison.json','w') as f: json.dump(rows,f,indent=2)

spec_rows=[]; specs=[]
for label,rec,t in data:
    f,pxx=spectrum(t['channel_mean'],t['fps']); specs.append((label,f,pxx,t))
    for lo,hi in [(1,100),(100,300),(300,600),(600,680),(600,820),(1,820)]:
        pf,pp=band_peak(f,pxx,lo,hi); spec_rows.append({'label':label,'kind':f'band_{lo}_{hi}_Hz_peak','frequency_hz':pf,'power_density':pp})
    for freq in [640.0,640.65,1000.0,24.0,29.0,162.0,450.0,750.0]:
        nf,npow=nearest(f,pxx,freq); spec_rows.append({'label':label,'kind':f'nearest_{freq}_Hz','frequency_hz':nf,'power_density':npow})
with open(out/'spectrum_peak_comparison.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['label','kind','frequency_hz','power_density']); w.writeheader(); w.writerows(spec_rows)

fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
for label,rec,t in data:
    tt=np.arange(len(t['channel_mean']))/t['fps']; ax.plot(tt,t['channel_mean'],lw=.8,label=label)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity over angle window'); ax.set_title('Intensity trace: sound off vs sound on'); ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'intensity_time_off_vs_on.png',dpi=200); plt.close(fig)
for fname,xlim,title in [('spectrum_off_vs_on_1_820Hz.png',(1,820),'Spectrum: 1 to Nyquist'),('spectrum_off_vs_on_640Hz_region.png',(580,700),'Spectrum around 640 Hz'),('spectrum_off_vs_on_lowfreq.png',(1,250),'Low-frequency spectrum')]:
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for label,f,pxx,t in specs: ax.semilogy(f,pxx,lw=1,label=label)
    ax.axvline(640,color='k',ls='--',lw=.8,alpha=.5,label='640 Hz')
    ax.set_xlim(*xlim); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD'); ax.set_title(title); ax.grid(alpha=.25); ax.legend()
    fig.savefig(out/fname,dpi=200); plt.close(fig)
summary={'assumption':'Two newest folders used; older labelled sound_off and newest labelled sound_on.', 'recordings':[{'label':l,'path':str(r),'fps':t['fps'],'n_frames':t['n_frames'],'duration_s':t['duration_s']} for l,r,t in data], 'intensity_stats':rows,'spectrum_checks':spec_rows,'output_dir':str(out)}
with open(out/'comparison_summary.json','w') as f: json.dump(summary,f,indent=2)
print('OUTPUT_DIR',out)
print('\nIntensity stats:')
for r in rows:
    print(r['label'],r['trace'],'mean',round(r['mean'],3),'p95-p5',round(r['p95_minus_p5'],3),'pct',round(r['p95_minus_p5_percent_mean'],3),'p99-p1 pct',round(r['p99_minus_p1_percent_mean'],3),'range pct',round(r['range']/r['mean']*100,3),'std',round(r['std'],3))
print('\nSpectrum checks:')
for row in spec_rows:
    if row['kind'] in ['band_600_680_Hz_peak','nearest_640.0_Hz','nearest_640.65_Hz','band_1_820_Hz_peak']:
        print(row['label'],row['kind'],round(row['frequency_hz'],3),f"{row['power_density']:.6g}")
