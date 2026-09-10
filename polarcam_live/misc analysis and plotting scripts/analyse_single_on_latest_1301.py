import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

rec = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\rod_x1466_y1493_20260722-130150_1784721710721298300")
out = rec / "single_on_intensity_spectrum_analysis"
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

def load_traces(rec):
    with open(rec/'capture_maxfps_15x15_meta.json', 'r') as f:
        meta=json.load(f)
    arr=strip_marker(np.load(rec/'capture_maxfps_15x15.npy', mmap_mode='r'))
    actual=meta.get('actual',{})
    fps=float(actual.get('fps') or meta.get('fps') or 1640.0709219858156)
    roi=actual.get('roi',{})
    raw11=arr[:,:14,:14].astype(np.float64)[:,1:12,1:12].mean(axis=(1,2))
    chans=channel_images(arr, roi)
    req=meta.get('requested',{}); center=req.get('center',{})
    cx=(int(center.get('x', int(roi.get('x',0))+7))-int(roi.get('x',0)))/2.0
    cy=(int(center.get('y', int(roi.get('y',0))+7))-int(roi.get('y',0)))/2.0
    cw=odd_window(float(req.get('window_raw',11))/2.0, min(next(iter(chans.values())).shape[1:]))
    parts=[]; pix_total=0
    for name in ['I0','I45','I135','I90']:
        tr,npix=centered_crop_mean(chans[name],cx,cy,cw)
        parts.append(tr*npix); pix_total+=npix
    channel_mean=np.sum(parts,axis=0)/pix_total
    return fps, arr.shape[0], raw11, channel_mean, cw, pix_total

def stats(name,tr,fps):
    p1,p5,p50,p95,p99=np.percentile(tr,[1,5,50,95,99])
    mean=float(np.mean(tr))
    return {'trace':name,'mean':mean,'std':float(np.std(tr,ddof=1)),'min':float(np.min(tr)),'max':float(np.max(tr)),'range':float(np.max(tr)-np.min(tr)),'p1':float(p1),'p5':float(p5),'median':float(p50),'p95':float(p95),'p99':float(p99),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'range_percent_mean':float((np.max(tr)-np.min(tr))/mean*100),'n_frames':int(len(tr)),'duration_s':float(len(tr)/fps)}

fps,n,raw11,channel_mean,cw,pix_total=load_traces(rec)
time=np.arange(n)/fps
rows=[stats('raw11_mean',raw11,fps), stats('channel_mean',channel_mean,fps)]
with open(out/'intensity_stats.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# Welch PSD and peak ranking.
y=channel_mean-channel_mean.mean()
nperseg=min(len(y),2048)
f,pxx=welch(y,fs=fps,nperseg=nperseg,noverlap=min(nperseg//2,1024),scaling='density')
mask=(f>=1)&(f<=fps/2)
peaks,_=find_peaks(pxx[mask], prominence=np.nanmedian(pxx[mask])*5)
valid_idx=np.flatnonzero(mask)
peak_idx=valid_idx[peaks]
# Always include largest bins if prominence misses broad features.
top_bins=valid_idx[np.argsort(pxx[mask])[-40:]]
all_idx=np.unique(np.concatenate([peak_idx, top_bins]))
all_idx=all_idx[np.argsort(pxx[all_idx])[::-1]]
peak_rows=[]
for idx in all_idx[:30]:
    freq=float(f[idx]); power=float(pxx[idx])
    # possible true frequency if alias from above Nyquist using fs-f_alias
    alias_partner=float(fps - freq) if freq <= fps/2 else None
    peak_rows.append({'rank':len(peak_rows)+1,'frequency_hz':freq,'power_density':power,'possible_alias_of_hz':alias_partner})
with open(out/'spectrum_top_peaks.csv','w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=['rank','frequency_hz','power_density','possible_alias_of_hz']); w.writeheader(); w.writerows(peak_rows)

def nearest(freq):
    idx=int(np.argmin(np.abs(f-freq)))
    return {'requested_hz':freq,'nearest_hz':float(f[idx]),'power_density':float(pxx[idx])}
checks=[nearest(v) for v in [23,29,70,160,162,320,450,500,600,640,640.65,700,750,800,1000]]
with open(out/'specific_frequency_checks.csv','w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=['requested_hz','nearest_hz','power_density']); w.writeheader(); w.writerows(checks)

fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.plot(time,channel_mean,lw=.8,label='angle-window mean')
ax.plot(time,raw11,lw=.65,alpha=.6,label='raw central 11x11 mean')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity'); ax.set_title('Sound-on recording intensity trace'); ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'intensity_trace.png',dpi=200); plt.close(fig)

for fname,xlim,title in [('spectrum_1_to_nyquist.png',(1,fps/2),'Spectrum: 1 Hz to Nyquist'),('spectrum_0_to_250Hz.png',(1,250),'Spectrum: 1-250 Hz'),('spectrum_580_to_700Hz.png',(580,700),'Spectrum around 640 Hz'),('spectrum_300_to_820Hz.png',(300,fps/2),'Spectrum: 300 Hz to Nyquist')]:
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    ax.semilogy(f,pxx,lw=.9)
    for v in [640, 162, 23]:
        if xlim[0] <= v <= xlim[1]: ax.axvline(v,color='k',ls='--',lw=.8,alpha=.4)
    ax.set_xlim(*xlim); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD'); ax.set_title(title); ax.grid(alpha=.25)
    fig.savefig(out/fname,dpi=200); plt.close(fig)

summary={'recording':str(rec),'fps':fps,'n_frames':n,'duration_s':n/fps,'window_channel_width':cw,'channel_total_px':pix_total,'intensity_stats':rows,'top_peaks':peak_rows,'specific_checks':checks,'output_dir':str(out)}
with open(out/'analysis_summary.json','w') as fjson: json.dump(summary,fjson,indent=2)
print('OUTPUT_DIR',out)
print('fps',fps,'frames',n,'duration',n/fps)
print('\nIntensity stats:')
for r in rows:
    print(r['trace'],'mean',round(r['mean'],3),'p95-p5',round(r['p95_minus_p5'],3),'pct',round(r['p95_minus_p5_percent_mean'],3),'p99-p1 pct',round(r['p99_minus_p1_percent_mean'],3),'range pct',round(r['range_percent_mean'],3),'std',round(r['std'],3))
print('\nTop spectrum peaks:')
for r in peak_rows[:12]:
    print(r['rank'],round(r['frequency_hz'],3),f"{r['power_density']:.6g}",'alias_of',round(r['possible_alias_of_hz'],3))
print('\nSpecific checks:')
for r in checks:
    if r['requested_hz'] in [640,640.65,1000,162,23,700,800]:
        print(r['requested_hz'],round(r['nearest_hz'],3),f"{r['power_density']:.6g}")
