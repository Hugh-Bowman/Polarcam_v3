import json, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

rec=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\rod_x1466_y1493_20260722-143113_1784727073785250500")
out=rec/"off_then_8625Hz_drive_analysis"
out.mkdir(exist_ok=True)
DRIVE_HZ=8625.0

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
    channel=np.sum(parts,axis=0)/pix_total
    return fps, channel, raw11, pix_total

def alias_freq(freq,fs):
    return abs(((freq + fs/2) % fs) - fs/2)

def stats(label,trace,fps):
    p1,p5,p50,p95,p99=np.percentile(trace,[1,5,50,95,99]); mean=float(np.mean(trace))
    return {'segment':label,'mean':mean,'std':float(np.std(trace,ddof=1)),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'minmax':float(np.max(trace)-np.min(trace)),'minmax_percent_mean':float((np.max(trace)-np.min(trace))/mean*100),'n_frames':int(len(trace)),'duration_s':float(len(trace)/fps)}

def psd(trace,fs):
    y=trace-np.mean(trace); nperseg=min(len(y),4096)
    return welch(y,fs=fs,nperseg=nperseg,noverlap=min(nperseg//2,2048),scaling='density')

def nearest(f,p,freq):
    idx=int(np.argmin(np.abs(f-freq)))
    return float(f[idx]),float(p[idx])

fps,channel,raw11,pix_total=load(rec)
n=len(channel); mid=n//2; duration=n/fps
alias=alias_freq(DRIVE_HZ,fps)
segments={'before_off':channel[:mid], 'after_drive_on':channel[mid:]}
raw_segments={'before_off_raw11':raw11[:mid], 'after_drive_on_raw11':raw11[mid:]}
rows=[stats(k,v,fps) for k,v in segments.items()] + [stats(k,v,fps) for k,v in raw_segments.items()]
with open(out/'intensity_before_after.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

spec_rows=[]
psds=[]
for label,tr in segments.items():
    f,p=psd(tr,fps); psds.append((label,f,p))
    nf,npow=nearest(f,p,alias)
    spec_rows.append({'segment':label,'kind':'nearest_expected_alias','frequency_hz':nf,'power_density':npow,'expected_alias_hz':alias})
    mask=(f>=max(1,alias-50))&(f<=min(fps/2,alias+50))
    idxs=np.flatnonzero(mask); idx=idxs[np.argmax(p[mask])]
    spec_rows.append({'segment':label,'kind':'peak_within_alias_pm50Hz','frequency_hz':float(f[idx]),'power_density':float(p[idx]),'expected_alias_hz':alias})
    mask2=(f>=1)&(f<=fps/2)
    idxs2=np.flatnonzero(mask2); idx2=idxs2[np.argmax(p[mask2])]
    spec_rows.append({'segment':label,'kind':'strongest_1_to_nyquist','frequency_hz':float(f[idx2]),'power_density':float(p[idx2]),'expected_alias_hz':alias})
with open(out/'spectrum_before_after.csv','w',newline='') as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=list(spec_rows[0].keys())); w.writeheader(); w.writerows(spec_rows)

# plots
time=np.arange(n)/fps
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.plot(time,channel,lw=.75,label='angle-window mean')
ax.axvline(time[mid],color='k',ls='--',lw=1,label='halfway split')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean intensity'); ax.set_title('Off then 8625 Hz drive: intensity trace')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'intensity_trace_off_then_8625Hz.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
for label,f,p in psds:
    ax.semilogy(f,p,lw=1,label=label)
ax.axvline(alias,color='k',ls='--',lw=1,label=f'8625 Hz alias {alias:.2f} Hz')
ax.set_xlim(1,fps/2); ax.set_xlabel('Observed frequency (Hz)'); ax.set_ylabel('PSD'); ax.set_title('Spectrum before vs after drive on')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'spectrum_before_after_8625Hz.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
for label,f,p in psds:
    ax.semilogy(f,p,lw=1,label=label)
ax.axvline(alias,color='k',ls='--',lw=1,label=f'expected alias {alias:.2f} Hz')
ax.set_xlim(max(1,alias-80),min(fps/2,alias+80)); ax.set_xlabel('Observed frequency (Hz)'); ax.set_ylabel('PSD'); ax.set_title('Expected alias region')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'spectrum_expected_alias_region_8625Hz.png',dpi=200); plt.close(fig)
# spectrogram for visual timing
f,t,S=spectrogram(channel-np.mean(channel),fs=fps,nperseg=512,noverlap=448,mode='psd',scaling='density')
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
pcm=ax.pcolormesh(t,f,10*np.log10(S+1e-15),shading='auto',cmap='magma')
ax.axvline(duration/2,color='cyan',ls='--',lw=1,label='halfway split')
ax.axhline(alias,color='cyan',ls=':',lw=1,label=f'8625 alias {alias:.2f} Hz')
ax.set_ylim(0,fps/2); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed frequency (Hz)'); ax.set_title('Off then 8625 Hz drive spectrogram')
ax.legend(); fig.colorbar(pcm,ax=ax,label='PSD (dB)')
fig.savefig(out/'spectrogram_off_then_8625Hz.png',dpi=200); plt.close(fig)
summary={'recording':str(rec),'drive_hz':DRIVE_HZ,'fps':fps,'nyquist_hz':fps/2,'expected_alias_hz':alias,'n_frames':n,'duration_s':duration,'split_frame':mid,'split_time_s':time[mid],'intensity_stats':rows,'spectrum_stats':spec_rows,'output_dir':str(out)}
with open(out/'analysis_summary.json','w') as f: json.dump(summary,f,indent=2)
print('OUTPUT_DIR',out)
print('fps',fps,'duration',duration,'frames',n,'split_time',time[mid],'expected_alias',alias)
print('\nIntensity:')
for r in rows:
    print(r['segment'],'mean',round(r['mean'],3),'p95-p5 pct',round(r['p95_minus_p5_percent_mean'],3),'p95-p5',round(r['p95_minus_p5'],3),'p99-p1 pct',round(r['p99_minus_p1_percent_mean'],3),'std',round(r['std'],3))
print('\nSpectrum:')
for r in spec_rows:
    print(r['segment'],r['kind'],round(r['frequency_hz'],3),f"{r['power_density']:.6g}")
