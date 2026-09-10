import json, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, find_peaks, savgol_filter

base = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending")
off_rec = base / "rod_x1466_y1493_20260722-130559_1784721959572876100"
on_rec = base / "rod_x1466_y1493_20260722-130620_1784721980746502900"
out = base / "frequency_sweep_20Hz_20kHz_20260722-130559_130620_analysis"
out.mkdir(exist_ok=True)
F_START = 20.0
F_END = 20000.0

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
    return patch.mean(axis=(1,2)), patch.shape[1]*patch.shape[2]

def load_trace(rec):
    with open(rec/'capture_maxfps_15x15_meta.json','r') as f:
        meta=json.load(f)
    arr=strip_marker(np.load(rec/'capture_maxfps_15x15.npy', mmap_mode='r'))
    actual=meta.get('actual',{})
    fps=float(actual.get('fps') or meta.get('fps') or 1640.0709219858156)
    roi=actual.get('roi',{})
    raw11=arr[:,:14,:14].astype(np.float64)[:,1:12,1:12].mean(axis=(1,2))
    chans=channel_images(arr,roi)
    req=meta.get('requested',{}); center=req.get('center',{})
    cx=(int(center.get('x', int(roi.get('x',0))+7))-int(roi.get('x',0)))/2.0
    cy=(int(center.get('y', int(roi.get('y',0))+7))-int(roi.get('y',0)))/2.0
    cw=odd_window(float(req.get('window_raw',11))/2.0, min(next(iter(chans.values())).shape[1:]))
    parts=[]; pix_total=0
    for name in ['I0','I45','I135','I90']:
        tr,npix=centered_crop_mean(chans[name],cx,cy,cw)
        parts.append(tr*npix); pix_total += npix
    channel_mean=np.sum(parts,axis=0)/pix_total
    return {'fps':fps,'raw11':raw11,'channel':channel_mean,'n':len(channel_mean),'duration':len(channel_mean)/fps,'path':str(rec),'cw':cw,'pix_total':pix_total}

def alias_freq(freq, fs):
    # Fold true frequency into [0, fs/2].
    return np.abs(((freq + fs/2) % fs) - fs/2)

def pct_stats(label, trace, fps):
    p1,p5,p50,p95,p99=np.percentile(trace,[1,5,50,95,99]); mean=float(np.mean(trace))
    return {'label':label,'mean':mean,'std':float(np.std(trace,ddof=1)),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'minmax_percent_mean':float((np.max(trace)-np.min(trace))/mean*100),'duration_s':len(trace)/fps}

off=load_trace(off_rec); on=load_trace(on_rec); fs=on['fps']
# Match lengths if needed for plotting only.
rows=[]
for label,d in [('no_sound',off),('sweep_on',on)]:
    r=pct_stats(label+'_angle_window', d['channel'], d['fps']); rows.append(r)
    r2=pct_stats(label+'_raw11', d['raw11'], d['fps']); rows.append(r2)
with open(out/'intensity_range_stats.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# PSD comparison.
def welch_psd(y, fs):
    yy=y-np.mean(y); nperseg=min(len(yy),2048)
    return welch(yy,fs=fs,nperseg=nperseg,noverlap=min(nperseg//2,1024),scaling='density')
f_off,p_off=welch_psd(off['channel'], off['fps'])
f_on,p_on=welch_psd(on['channel'], on['fps'])

# Spectrogram of sweep-on.
y=on['channel']-np.mean(on['channel'])
nperseg=256; noverlap=224
f,t,S=spectrogram(y,fs=fs,nperseg=nperseg,noverlap=noverlap,mode='psd',scaling='density')
true_f = F_START + (F_END-F_START)*(t/on['duration'])
alias = alias_freq(true_f, fs)
# Extract response near expected alias track.
bin_width = f[1]-f[0]
half_bw_hz = max(10.0, 2*bin_width)
resp=[]; bg=[]; alias_bin=[]
for ti, af in enumerate(alias):
    m=(f>=max(1,af-half_bw_hz)) & (f<=min(fs/2,af+half_bw_hz))
    if not np.any(m):
        resp.append(np.nan); alias_bin.append(np.nan)
    else:
        resp.append(float(np.nanmax(S[m,ti]))); alias_bin.append(float(f[np.flatnonzero(m)[np.argmax(S[m,ti])]]))
    bm=(f>=1)&(f<=fs/2)
    bg.append(float(np.nanmedian(S[bm,ti])))
resp=np.array(resp); bg=np.array(bg); alias_bin=np.array(alias_bin)
score=resp/(np.array(bg)+1e-12)
# Smooth in time/frequency to find broad peaks.
score_s=score.copy()
finite=np.isfinite(score_s)
if finite.sum() >= 9:
    # interpolate nans and smooth
    x=np.arange(len(score_s))
    score_s[~finite]=np.interp(x[~finite], x[finite], score_s[finite])
    win=min(31, len(score_s)//2*2-1)
    if win >= 7:
        score_s=savgol_filter(score_s, win, 3)
peaks, props = find_peaks(score_s, prominence=max(1.0, np.nanstd(score_s)*0.7), distance=4)
# Also keep top scores to avoid missing narrow peaks.
candidate=np.unique(np.concatenate([peaks, np.argsort(score_s)[-12:]]))
candidate=candidate[np.argsort(score_s[candidate])[::-1]]
peak_rows=[]
for idx in candidate[:20]:
    peak_rows.append({'rank':len(peak_rows)+1,'time_s':float(t[idx]),'drive_frequency_hz_assuming_linear_sweep':float(true_f[idx]),'observed_alias_hz':float(alias[idx]),'strongest_bin_near_alias_hz':float(alias_bin[idx]),'track_power_density':float(resp[idx]),'track_over_local_median':float(score[idx]),'smoothed_score':float(score_s[idx])})
with open(out/'sweep_resonance_candidates.csv','w',newline='') as fcsv:
    fields=list(peak_rows[0].keys()) if peak_rows else ['rank']
    w=csv.DictWriter(fcsv,fieldnames=fields); w.writeheader(); w.writerows(peak_rows)

# Full spectral peaks on on-vs-off ratio.
# interpolate off PSD to on f grid
p_off_i=np.interp(f_on, f_off, p_off)
ratio=p_on/(p_off_i+1e-12)
mask=(f_on>=1)&(f_on<=fs/2)
peaks2,_=find_peaks(ratio[mask], prominence=np.nanmedian(ratio[mask])*3)
valid=np.flatnonzero(mask)
idxs=np.unique(np.concatenate([valid[peaks2], valid[np.argsort(ratio[mask])[-20:]]]))
idxs=idxs[np.argsort(ratio[idxs])[::-1]]
ratio_rows=[]
for idx in idxs[:20]:
    ratio_rows.append({'rank':len(ratio_rows)+1,'observed_frequency_hz':float(f_on[idx]),'on_psd':float(p_on[idx]),'off_psd':float(p_off_i[idx]),'on_over_off':float(ratio[idx])})
with open(out/'on_vs_off_psd_ratio_peaks.csv','w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=list(ratio_rows[0].keys())); w.writeheader(); w.writerows(ratio_rows)

# Save track table.
with open(out/'sweep_alias_track_response.csv','w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=['time_s','drive_frequency_hz_assuming_linear_sweep','expected_alias_hz','strongest_bin_near_alias_hz','track_power_density','track_over_local_median','smoothed_score'])
    w.writeheader()
    for i in range(len(t)):
        w.writerow({'time_s':float(t[i]),'drive_frequency_hz_assuming_linear_sweep':float(true_f[i]),'expected_alias_hz':float(alias[i]),'strongest_bin_near_alias_hz':float(alias_bin[i]),'track_power_density':float(resp[i]),'track_over_local_median':float(score[i]),'smoothed_score':float(score_s[i])})

# Plots
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
for label,d in [('no sound',off),('sweep on',on)]:
    tt=np.arange(d['n'])/d['fps']; ax.plot(tt,d['channel'],lw=.8,label=label)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity over angle window'); ax.set_title('No sound vs frequency sweep intensity'); ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'intensity_trace_no_sound_vs_sweep.png',dpi=200); plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.semilogy(f_off,p_off,lw=1,label='no sound')
ax.semilogy(f_on,p_on,lw=1,label='sweep on')
ax.set_xlim(1,fs/2); ax.set_xlabel('Observed frequency after camera sampling (Hz)'); ax.set_ylabel('PSD'); ax.set_title('Intensity spectrum: no sound vs sweep on'); ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'spectrum_no_sound_vs_sweep.png',dpi=200); plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
pcm=ax.pcolormesh(t,f,10*np.log10(S+1e-15),shading='auto',cmap='magma')
ax.plot(t,alias,color='cyan',lw=1.2,label='expected alias of 20 Hz to 20 kHz linear sweep')
ax.set_ylim(0,fs/2); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed alias frequency (Hz)'); ax.set_title('Sweep-on spectrogram with expected aliased chirp track')
ax.legend(loc='upper right'); fig.colorbar(pcm,ax=ax,label='PSD (dB)')
fig.savefig(out/'sweep_spectrogram_with_alias_track.png',dpi=200); plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.plot(true_f,score,lw=.8,alpha=.45,label='track PSD / local median')
ax.plot(true_f,score_s,lw=1.5,label='smoothed')
if peak_rows:
    px=[r['drive_frequency_hz_assuming_linear_sweep'] for r in peak_rows[:8]]; py=[r['smoothed_score'] for r in peak_rows[:8]]
    ax.scatter(px,py,s=30,color='red',zorder=4,label='candidate resonances')
    for r in peak_rows[:8]:
        ax.annotate(f"{r['drive_frequency_hz_assuming_linear_sweep']/1000:.2f} kHz", (r['drive_frequency_hz_assuming_linear_sweep'], r['smoothed_score']), fontsize=8, xytext=(4,4), textcoords='offset points')
ax.set_xlabel('Assumed drive frequency during linear sweep (Hz)'); ax.set_ylabel('Response along aliased sweep track'); ax.set_title('Estimated resonance response from aliased sweep')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'estimated_resonance_response_vs_drive_frequency.png',dpi=200); plt.close(fig)

summary={'assumptions':{'sweep':'linear 20 Hz to 20 kHz across recording duration','off_recording':str(off_rec),'on_recording':str(on_rec),'alias_formula':'abs(((f + fs/2) % fs) - fs/2)'},'fps':fs,'nyquist_hz':fs/2,'intensity_stats':rows,'resonance_candidates':peak_rows,'on_vs_off_ratio_peaks':ratio_rows,'output_dir':str(out)}
with open(out/'analysis_summary.json','w') as fjson: json.dump(summary,fjson,indent=2)

print('OUTPUT_DIR',out)
print('fps',fs,'nyquist',fs/2,'duration on',on['duration'])
print('\nIntensity variation:')
for r in rows:
    print(r['label'],'p95-p5',round(r['p95_minus_p5'],3),'pct',round(r['p95_minus_p5_percent_mean'],3),'p99-p1 pct',round(r['p99_minus_p1_percent_mean'],3),'std',round(r['std'],3))
print('\nTop resonance candidates along expected aliased sweep track:')
for r in peak_rows[:10]:
    print(r['rank'], 't',round(r['time_s'],3),'drive Hz',round(r['drive_frequency_hz_assuming_linear_sweep'],1),'alias Hz',round(r['observed_alias_hz'],1),'score',round(r['smoothed_score'],2))
print('\nTop on/off observed PSD ratio peaks:')
for r in ratio_rows[:10]:
    print(r['rank'],round(r['observed_frequency_hz'],2),'Hz ratio',round(r['on_over_off'],2),'onPSD',f"{r['on_psd']:.3g}")
