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
    baseline_mask=t<100.0
    after_mask=t>=100.0
    if baseline_mask.sum()<3:
        raise RuntimeError('Not enough pre-100s baseline')
    # Baseline per frequency. Score = fold increase above that frequency's pre-100s median.
    baseline=np.median(SS[:,baseline_mask],axis=1,keepdims=True)+1e-15
    Z=SS/baseline
    # Also subtract 1 so zero means no increase.
    Inc=Z-1.0
    best_i=np.argmax(Inc,axis=0)
    best_freq=freqs[best_i]
    best_increase=Inc[best_i,np.arange(Inc.shape[1])]
    best_ratio=Z[best_i,np.arange(Z.shape[1])]
    sm=best_increase.copy()
    win=min(31,len(sm)//2*2-1)
    if win>=7: sm=savgol_filter(sm,win,3)
    work=sm.copy(); work[~after_mask]=np.nanmin(sm[after_mask])
    prom=max(np.std(sm[after_mask])*0.45, np.percentile(sm[after_mask],75)-np.percentile(sm[after_mask],25), 0.75)
    peaks,_=find_peaks(work,prominence=prom,distance=3)
    cand=np.unique(np.concatenate([peaks,np.argsort(np.where(after_mask,work,-np.inf))[-40:]]))
    cand=cand[after_mask[cand]]; cand=cand[np.argsort(work[cand])[::-1]]
    selected=[]
    for idx in cand:
        if all(abs(t[idx]-t[j])>0.5 for j in selected):
            selected.append(idx)
        if len(selected)>=20: break
    rows=[]
    for idx in selected:
        rows.append({'rank':len(rows)+1,'method':label,'spectrogram_window_frames':nperseg,'time_s':float(t[idx]),'observed_alias_frequency_hz':float(best_freq[idx]),'increase_above_pre100s_baseline_ratio_minus_1':float(best_increase[idx]),'ratio_to_pre100s_baseline':float(best_ratio[idx]),'smoothed_increase_score':float(sm[idx]),'raw_psd_at_peak':float(SS[best_i[idx],idx]),'baseline_psd_at_that_frequency':float(baseline[best_i[idx],0])})
    all_rows.extend(rows)
    with open(out/f'alias_peak_times_after_100s_ranked_by_increase_{label}.csv','w',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fig,ax=plt.subplots(figsize=(13,5.5),constrained_layout=True)
    # Plot log increase ratio heatmap. Clip for readability.
    H=10*np.log10(Z+1e-12)
    vmin=np.nanpercentile(H[:,after_mask],5); vmax=np.nanpercentile(H[:,after_mask],99.5)
    pcm=ax.pcolormesh(t,freqs,H,shading='auto',cmap='magma',vmin=vmin,vmax=vmax)
    ax.axvline(100,color='white',ls='--',lw=1,label='100 s cutoff')
    ax.scatter([r['time_s'] for r in rows[:12]],[r['observed_alias_frequency_hz'] for r in rows[:12]],s=42,c='cyan',edgecolors='k',label='largest increases')
    for r in rows[:12]:
        ax.annotate(f"{r['time_s']:.1f}s\n{r['observed_alias_frequency_hz']:.0f}Hz",(r['time_s'],r['observed_alias_frequency_hz']),fontsize=7,xytext=(4,4),textcoords='offset points',color='white')
    ax.set_xlim(95,t[-1]); ax.set_ylim(20,fps/2-20); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed alias frequency (Hz)')
    ax.set_title(f'Post-100 s increases relative to pre-100 s baseline, {label}')
    ax.legend(loc='upper right'); fig.colorbar(pcm,ax=ax,label='PSD increase vs pre-100s baseline (dB)')
    fig.savefig(out/f'slow_sweep_post100_increase_heatmap_{label}.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True)
    ax.plot(t,np.where(after_mask,best_increase,np.nan),lw=.75,alpha=.4,label='best frequency increase')
    ax.plot(t,np.where(after_mask,sm,np.nan),lw=1.4,label='smoothed')
    ax.scatter([r['time_s'] for r in rows[:12]],[r['smoothed_increase_score'] for r in rows[:12]],c='red',s=30,label='largest increases')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Increase above pre-100 s baseline, ratio-1'); ax.set_title(f'Largest post-100 s increases, {label}')
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(out/f'slow_sweep_post100_increase_score_{label}.png',dpi=200); plt.close(fig)
with open(out/'alias_peak_summary_after_100s_ranked_by_increase.json','w') as fjson:
    json.dump({'recording':str(rec),'fps':fps,'nyquist_hz':fps/2,'note':'Peaks after 100s ranked by spectral increase relative to the same alias-frequency baseline before 100s. No input-frequency mapping used.','events':all_rows},fjson,indent=2)
primary=[r for r in all_rows if r['method']=='frequency_detail']
print('OUTPUT_DIR',out)
print('Primary post-100s peaks ranked by increase above pre-100s baseline:')
for r in primary[:12]:
    print(r['rank'],'time_s',round(r['time_s'],3),'alias_hz',round(r['observed_alias_frequency_hz'],2),'ratio',round(r['ratio_to_pre100s_baseline'],2),'increase',round(r['increase_above_pre100s_baseline_ratio_minus_1'],2),'rawPSD',f"{r['raw_psd_at_peak']:.3g}")
print('\nTime-detail version:')
for r in [x for x in all_rows if x['method']=='time_detail'][:8]:
    print(r['rank'],'time_s',round(r['time_s'],3),'alias_hz',round(r['observed_alias_frequency_hz'],2),'ratio',round(r['ratio_to_pre100s_baseline'],2),'increase',round(r['increase_above_pre100s_baseline_ratio_minus_1'],2))
