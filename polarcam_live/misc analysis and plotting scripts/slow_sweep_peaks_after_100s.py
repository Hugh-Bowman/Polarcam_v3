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
    chans=channel_images(arr,roi); req=meta.get('requested',{}); center=req.get('center',{})
    cx=(int(center.get('x',int(roi.get('x',0))+7))-int(roi.get('x',0)))/2.0
    cy=(int(center.get('y',int(roi.get('y',0))+7))-int(roi.get('y',0)))/2.0
    cw=odd_window(float(req.get('window_raw',11))/2.0,min(next(iter(chans.values())).shape[1:]))
    parts=[]; pix_total=0
    for name in ['I0','I45','I135','I90']:
        tr,npix=centered_crop_mean(chans[name],cx,cy,cw); parts.append(tr*npix); pix_total+=npix
    return fps, np.sum(parts,axis=0)/pix_total

fps, trace = load_trace(rec)
y=trace-np.mean(trace)
configs=[('frequency_detail',512,448),('time_detail',256,224)]
all_rows=[]
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
    after=t>=100.0
    work=sm.copy(); work[~after]=np.nanmin(sm[after]) if after.any() else np.nanmin(sm)
    prom=max(np.std(sm[after])*0.45, np.percentile(sm[after],75)-np.percentile(sm[after],25), 0.75)
    peaks,_=find_peaks(work,prominence=prom,distance=3)
    cand=np.unique(np.concatenate([peaks,np.argsort(np.where(after,work,-np.inf))[-25:]]))
    cand=cand[after[cand]]; cand=cand[np.argsort(work[cand])[::-1]]
    selected=[]
    for idx in cand:
        if all(abs(t[idx]-t[j])>0.5 for j in selected):
            selected.append(idx)
        if len(selected)>=15: break
    rows=[]
    for idx in selected:
        rows.append({'rank':len(rows)+1,'method':label,'spectrogram_window_frames':nperseg,'time_s':float(t[idx]),'observed_alias_frequency_hz':float(best_freq[idx]),'response_score_frequency_whitened':float(best_score[idx]),'smoothed_response_score':float(sm[idx]),'raw_psd_at_peak':float(SS[best_i[idx],idx])})
    all_rows.extend(rows)
    with open(out/f'alias_peak_times_after_100s_{label}.csv','w',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fig,ax=plt.subplots(figsize=(13,5.5),constrained_layout=True)
    pcm=ax.pcolormesh(t,f,10*np.log10(S+1e-15),shading='auto',cmap='magma')
    ax.axvline(100,color='white',ls='--',lw=1,label='100 s cutoff')
    ax.scatter([r['time_s'] for r in rows[:12]],[r['observed_alias_frequency_hz'] for r in rows[:12]],s=42,c='cyan',edgecolors='k',label='peaks after 100 s')
    for r in rows[:12]:
        ax.annotate(f"{r['time_s']:.1f}s\n{r['observed_alias_frequency_hz']:.0f}Hz",(r['time_s'],r['observed_alias_frequency_hz']),fontsize=7,xytext=(4,4),textcoords='offset points',color='white')
    ax.set_ylim(0,fps/2); ax.set_xlim(95,t[-1]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed alias frequency (Hz)')
    ax.set_title(f'Slow sweep peaks after 100 s, {label}')
    ax.legend(loc='upper right'); fig.colorbar(pcm,ax=ax,label='PSD (dB)')
    fig.savefig(out/f'slow_sweep_spectrogram_peaks_after_100s_{label}.png',dpi=200); plt.close(fig)

with open(out/'alias_peak_summary_after_100s.json','w') as fjson:
    json.dump({'recording':str(rec),'fps':fps,'nyquist_hz':fps/2,'note':'No input frequency mapping; only camera-observed alias frequencies after 100 s.','events':all_rows},fjson,indent=2)
primary=[r for r in all_rows if r['method']=='frequency_detail']
print('OUTPUT_DIR',out)
print('Primary peaks after 100 s, frequency_detail:')
for r in primary[:12]:
    print(r['rank'],'time_s',round(r['time_s'],3),'alias_hz',round(r['observed_alias_frequency_hz'],2),'score',round(r['smoothed_response_score'],2),'rawPSD',f"{r['raw_psd_at_peak']:.3g}")
print('\nTime detail peaks after 100 s:')
for r in [x for x in all_rows if x['method']=='time_detail'][:8]:
    print(r['rank'],'time_s',round(r['time_s'],3),'alias_hz',round(r['observed_alias_frequency_hz'],2),'score',round(r['smoothed_response_score'],2))
