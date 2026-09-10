import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, spectrogram

rec=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\rod_x1578_y502_20260722-145208_1784728328945557800")
out=rec/"drive_780Hz_intensity_analysis"
out.mkdir(exist_ok=True)
DRIVE_HZ=780.0

def strip_marker(arr):
    if arr.ndim==3 and arr.shape[0]>1:
        last=arr[-1]; finite=np.isfinite(last)
        if finite.sum()==1 and np.nanmax(last)==1.0: return arr[:-1]
    return arr

def channel_images(raw,roi):
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

def load(rec):
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
    return fps, np.sum(parts,axis=0)/pix_total, raw11, pix_total

def alias_freq(freq,fs):
    return abs(((freq+fs/2)%fs)-fs/2)

def stats(trace_name,tr,fps):
    p1,p5,p50,p95,p99=np.percentile(tr,[1,5,50,95,99]); mean=float(np.mean(tr))
    return {'trace':trace_name,'mean':mean,'std':float(np.std(tr,ddof=1)),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'minmax':float(np.max(tr)-np.min(tr)),'minmax_percent_mean':float((np.max(tr)-np.min(tr))/mean*100),'n_frames':int(len(tr)),'duration_s':float(len(tr)/fps)}

def psd(trace,fs):
    y=trace-np.mean(trace); nperseg=min(len(y),4096)
    return welch(y,fs=fs,nperseg=nperseg,noverlap=min(nperseg//2,2048),scaling='density')

def nearest(f,p,freq):
    idx=int(np.argmin(np.abs(f-freq)))
    return float(f[idx]),float(p[idx])

def top_peaks(f,p,n=12):
    m=(f>=1)&(f<=f[-1]); valid=np.flatnonzero(m)
    peaks,_=find_peaks(p[m],prominence=np.nanmedian(p[m])*5)
    idxs=np.unique(np.concatenate([valid[peaks],valid[np.argsort(p[m])[-30:]]]))
    idxs=idxs[np.argsort(p[idxs])[::-1]]
    rows=[]
    for idx in idxs:
        if all(abs(f[idx]-r['frequency_hz'])>5 for r in rows):
            rows.append({'rank':len(rows)+1,'frequency_hz':float(f[idx]),'power_density':float(p[idx])})
        if len(rows)>=n: break
    return rows

fps,channel,raw11,pix_total=load(rec)
alias=alias_freq(DRIVE_HZ,fps)
rows=[stats('channel_mean',channel,fps),stats('raw11_mean',raw11,fps)]
with open(out/'intensity_stats.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
f,p=psd(channel,fps)
peaks=top_peaks(f,p,12)
spec_rows=[]
for r in peaks:
    spec_rows.append({'kind':'top_peak','frequency_hz':r['frequency_hz'],'power_density':r['power_density'],'expected_alias_hz':alias})
nf,npow=nearest(f,p,alias)
spec_rows.append({'kind':'nearest_expected_alias','frequency_hz':nf,'power_density':npow,'expected_alias_hz':alias})
m=(f>=max(1,alias-50))&(f<=min(fps/2,alias+50)); idxs=np.flatnonzero(m); idx=idxs[np.argmax(p[m])]
spec_rows.append({'kind':'peak_within_alias_pm50Hz','frequency_hz':float(f[idx]),'power_density':float(p[idx]),'expected_alias_hz':alias})
with open(out/'frequency_peaks.csv','w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=list(spec_rows[0].keys())); w.writeheader(); w.writerows(spec_rows)
# plots
time=np.arange(len(channel))/fps
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.plot(time,channel,lw=.75,label='angle-window mean')
ax.plot(time,raw11,lw=.65,alpha=.55,label='raw central 11x11')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity'); ax.set_title('780 Hz drive intensity trace')
ax.grid(alpha=.25); ax.legend(); fig.savefig(out/'intensity_trace_780Hz.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.semilogy(f,p,lw=1)
ax.axvline(alias,color='k',ls='--',lw=1,label=f'expected alias {alias:.2f} Hz')
ax.set_xlim(1,fps/2); ax.set_xlabel('Observed frequency (Hz)'); ax.set_ylabel('PSD'); ax.set_title('780 Hz drive spectrum')
ax.grid(alpha=.25); ax.legend(); fig.savefig(out/'spectrum_780Hz.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.semilogy(f,p,lw=1)
ax.axvline(alias,color='k',ls='--',lw=1,label=f'expected alias {alias:.2f} Hz')
ax.set_xlim(max(1,alias-80),min(fps/2,alias+80)); ax.set_xlabel('Observed frequency (Hz)'); ax.set_ylabel('PSD'); ax.set_title('780 Hz alias region')
ax.grid(alpha=.25); ax.legend(); fig.savefig(out/'spectrum_780Hz_alias_region.png',dpi=200); plt.close(fig)
fsp,tsp,S=spectrogram(channel-np.mean(channel),fs=fps,nperseg=min(512,len(channel)),noverlap=min(448,max(0,min(512,len(channel))-1)),mode='psd',scaling='density')
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
pcm=ax.pcolormesh(tsp,fsp,10*np.log10(S+1e-15),shading='auto',cmap='magma')
ax.axhline(alias,color='cyan',ls='--',lw=1,label=f'expected alias {alias:.1f} Hz')
ax.set_ylim(0,fps/2); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed frequency (Hz)'); ax.set_title('780 Hz drive spectrogram')
ax.legend(); fig.colorbar(pcm,ax=ax,label='PSD (dB)'); fig.savefig(out/'spectrogram_780Hz.png',dpi=200); plt.close(fig)
summary={'recording':str(rec),'drive_hz':DRIVE_HZ,'fps':fps,'nyquist_hz':fps/2,'expected_alias_hz':alias,'duration_s':len(channel)/fps,'n_frames':len(channel),'intensity_stats':rows,'spectrum_rows':spec_rows,'output_dir':str(out)}
with open(out/'analysis_summary.json','w') as fjson: json.dump(summary,fjson,indent=2)
print('OUTPUT_DIR',out)
print('fps',fps,'duration',len(channel)/fps,'frames',len(channel),'expected_alias',alias)
print('\nIntensity:')
for r in rows:
    print(r['trace'],'p95-p5 pct',round(r['p95_minus_p5_percent_mean'],3),'p95-p5',round(r['p95_minus_p5'],3),'p99-p1 pct',round(r['p99_minus_p1_percent_mean'],3),'std',round(r['std'],3))
print('\nPeaks:')
for r in spec_rows[:10]:
    print(r['kind'],round(float(r['frequency_hz']),3),f"{float(r['power_density']):.6g}")
