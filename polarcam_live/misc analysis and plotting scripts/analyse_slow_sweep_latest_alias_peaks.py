import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks, savgol_filter

rec = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\rod_x1466_y1493_20260722-131836_1784722716940986800")
out = rec / "slow_frequency_sweep_alias_peak_analysis"
out.mkdir(exist_ok=True)

def strip_marker(arr):
    if arr.ndim == 3 and arr.shape[0] > 1:
        last=arr[-1]; finite=np.isfinite(last)
        if finite.sum()==1 and np.nanmax(last)==1.0: return arr[:-1]
    return arr

def channel_images(raw, roi):
    x=int(roi.get('x',0)); y=int(roi.get('y',0)); px=x%2; py=y%2
    g=raw[:,:14,:14].astype(np.float64)
    return {'I0':g[:,py::2,px::2],'I45':g[:,py::2,(1-px)::2],'I135':g[:,(1-py)::2,px::2],'I90':g[:,(1-py)::2,(1-px)::2]}

def odd_window(n,limit):
    n=max(1,int(round(n)))
    if n%2==0: n+=1
    return max(1,min(n,limit if limit%2==1 else limit-1))

def centered_crop_mean(stack,cx,cy,w):
    h,ww=stack.shape[1:]; half=w//2
    ix=max(half,min(ww-half-1,int(round(cx)))); iy=max(half,min(h-half-1,int(round(cy))))
    patch=stack[:,iy-half:iy+half+1,ix-half:ix+half+1]
    return patch.mean(axis=(1,2)), patch.shape[1]*patch.shape[2]

def load_trace(rec):
    with open(rec/'capture_maxfps_15x15_meta.json','r') as f: meta=json.load(f)
    arr=strip_marker(np.load(rec/'capture_maxfps_15x15.npy',mmap_mode='r'))
    actual=meta.get('actual',{}); roi=actual.get('roi',{})
    fps=float(actual.get('fps') or meta.get('fps') or 1640.0709219858156)
    raw11=arr[:,:14,:14].astype(np.float64)[:,1:12,1:12].mean(axis=(1,2))
    chans=channel_images(arr,roi); req=meta.get('requested',{}); center=req.get('center',{})
    cx=(int(center.get('x',int(roi.get('x',0))+7))-int(roi.get('x',0)))/2.0
    cy=(int(center.get('y',int(roi.get('y',0))+7))-int(roi.get('y',0)))/2.0
    cw=odd_window(float(req.get('window_raw',11))/2.0,min(next(iter(chans.values())).shape[1:]))
    parts=[]; pix_total=0
    for name in ['I0','I45','I135','I90']:
        tr,npix=centered_crop_mean(chans[name],cx,cy,cw); parts.append(tr*npix); pix_total+=npix
    channel=np.sum(parts,axis=0)/pix_total
    return fps, channel, raw11, cw, pix_total

fps, channel, raw11, cw, pix_total = load_trace(rec)
y = channel - np.mean(channel)
# Two spectrograms: one for heatmap/time detail, one for peak list/frequency detail.
configs=[('time_detail',256,224),('frequency_detail',512,448)]
summary_events=[]
for label,nperseg,noverlap in configs:
    f,t,S=spectrogram(y,fs=fps,nperseg=nperseg,noverlap=noverlap,mode='psd',scaling='density')
    use=(f>=20)&(f<=fps/2-20)
    freqs=f[use]; SS=S[use,:]
    med_f=np.median(SS,axis=1,keepdims=True)+1e-15
    Z=SS/med_f
    best_i=np.argmax(Z,axis=0)
    best_freq=freqs[best_i]
    best_score=Z[best_i,np.arange(Z.shape[1])]
    sm=best_score.copy()
    win=min(31,len(sm)//2*2-1)
    if win>=7: sm=savgol_filter(sm,win,3)
    prom=max(np.std(sm)*0.55, np.percentile(sm,75)-np.percentile(sm,25), 1.0)
    peaks,_=find_peaks(sm,prominence=prom,distance=3)
    cand=np.unique(np.concatenate([peaks,np.argsort(sm)[-40:]]))
    cand=cand[np.argsort(sm[cand])[::-1]]
    selected=[]
    for idx in cand:
        if all(abs(t[idx]-t[j])>0.45 for j in selected):
            selected.append(idx)
        if len(selected)>=20: break
    rows=[]
    for idx in selected:
        rows.append({'rank':len(rows)+1,'method':label,'spectrogram_window_frames':nperseg,'time_s':float(t[idx]),'observed_alias_frequency_hz':float(best_freq[idx]),'response_score_frequency_whitened':float(best_score[idx]),'smoothed_response_score':float(sm[idx]),'raw_psd_at_peak':float(SS[best_i[idx],idx])})
    summary_events.extend(rows)
    with open(out/f'alias_peak_times_{label}.csv','w',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fig,ax=plt.subplots(figsize=(13,5.5),constrained_layout=True)
    pcm=ax.pcolormesh(t,f,10*np.log10(S+1e-15),shading='auto',cmap='magma')
    ax.scatter([r['time_s'] for r in rows[:12]],[r['observed_alias_frequency_hz'] for r in rows[:12]],s=42,c='cyan',edgecolors='k',label='marked peaks')
    for r in rows[:12]:
        ax.annotate(f"{r['time_s']:.1f}s\n{r['observed_alias_frequency_hz']:.0f}Hz",(r['time_s'],r['observed_alias_frequency_hz']),fontsize=7,xytext=(4,4),textcoords='offset points',color='white')
    ax.set_ylim(0,fps/2); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed alias frequency (Hz)')
    ax.set_title(f'Slow sweep spectrogram, {label}')
    ax.legend(loc='upper right'); fig.colorbar(pcm,ax=ax,label='PSD (dB)')
    fig.savefig(out/f'slow_sweep_spectrogram_heatmap_{label}.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True)
    ax.plot(t,best_score,lw=.75,alpha=.4,label='best frequency-whitened score')
    ax.plot(t,sm,lw=1.4,label='smoothed')
    ax.scatter([r['time_s'] for r in rows[:12]],[r['smoothed_response_score'] for r in rows[:12]],c='red',s=30,label='marked peaks')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Response score'); ax.set_title(f'Slow sweep resonance timing score, {label}')
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(out/f'slow_sweep_peak_score_{label}.png',dpi=200); plt.close(fig)

# Intensity trace and stats
qt=np.percentile(channel,[1,5,50,95,99]); mean=float(np.mean(channel))
stats={'mean':mean,'std':float(np.std(channel,ddof=1)),'p95_minus_p5':float(qt[3]-qt[1]),'p95_minus_p5_percent_mean':float((qt[3]-qt[1])/mean*100),'p99_minus_p1_percent_mean':float((qt[4]-qt[0])/mean*100),'duration_s':len(channel)/fps,'fps':fps,'nyquist_hz':fps/2}
with open(out/'intensity_stats.json','w') as fjson: json.dump(stats,fjson,indent=2)
fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True)
time=np.arange(len(channel))/fps
ax.plot(time,channel,lw=.75,label='angle-window mean')
ax.plot(time,raw11,lw=.6,alpha=.55,label='raw central 11x11 mean')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity'); ax.set_title('Slow sweep intensity trace')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'slow_sweep_intensity_trace.png',dpi=200); plt.close(fig)
with open(out/'alias_peak_summary.json','w') as fjson:
    json.dump({'recording':str(rec),'fps':fps,'nyquist_hz':fps/2,'note':'No input frequency mapping assumed. Frequencies are camera-observed alias frequencies. Peaks are selected from frequency-whitened spectrogram response.','intensity_stats':stats,'events':summary_events},fjson,indent=2)

primary=[r for r in summary_events if r['method']=='frequency_detail']
print('OUTPUT_DIR',out)
print('fps',fps,'nyquist',fps/2,'duration',len(channel)/fps)
print('intensity p95-p5 pct',round(stats['p95_minus_p5_percent_mean'],3),'p99-p1 pct',round(stats['p99_minus_p1_percent_mean'],3))
print('\nPrimary peak times, frequency_detail:')
for r in primary[:15]:
    print(r['rank'],'time_s',round(r['time_s'],3),'alias_hz',round(r['observed_alias_frequency_hz'],2),'score',round(r['smoothed_response_score'],2),'rawPSD',f"{r['raw_psd_at_peak']:.3g}")
