import json, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, spectrogram

base=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending")
recordings=[
    ('focus_pan', None, base/'rod_x1578_y502_20260722-144539_1784727939057247100'),
    ('sound_3000Hz', 3000.0, base/'rod_x1578_y502_20260722-144558_1784727958210188800'),
    ('sound_3100Hz', 3100.0, base/'rod_x1578_y502_20260722-144622_1784727982267192800'),
    ('sound_2900Hz', 2900.0, base/'rod_x1578_y502_20260722-144635_1784727995460778900'),
]
out=base/'last4_focus_3000_3100_2900_analysis'
out.mkdir(exist_ok=True)

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
    return {'fps':fps,'channel_mean':channel,'raw11_mean':raw11,'n_frames':len(channel),'duration_s':len(channel)/fps,'path':str(rec),'pix_total':pix_total}

def alias_freq(freq,fs):
    return abs(((freq+fs/2)%fs)-fs/2)

def stats(label,trace_name,tr,fps,drive_hz=None):
    p1,p5,p50,p95,p99=np.percentile(tr,[1,5,50,95,99]); mean=float(np.mean(tr))
    return {'label':label,'drive_hz':drive_hz if drive_hz is not None else '', 'trace':trace_name,'mean':mean,'std':float(np.std(tr,ddof=1)),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'minmax':float(np.max(tr)-np.min(tr)),'minmax_percent_mean':float((np.max(tr)-np.min(tr))/mean*100),'n_frames':int(len(tr)),'duration_s':float(len(tr)/fps)}

def psd(trace,fs):
    y=trace-np.mean(trace); nperseg=min(len(y),4096)
    return welch(y,fs=fs,nperseg=nperseg,noverlap=min(nperseg//2,2048),scaling='density')

def nearest(f,p,freq):
    idx=int(np.argmin(np.abs(f-freq)))
    return float(f[idx]),float(p[idx])

def top_peaks(f,p,n=10):
    m=(f>=1)&(f<=f[-1])
    valid=np.flatnonzero(m)
    peaks,_=find_peaks(p[m], prominence=np.nanmedian(p[m])*5)
    idxs=np.unique(np.concatenate([valid[peaks], valid[np.argsort(p[m])[-30:]]]))
    idxs=idxs[np.argsort(p[idxs])[::-1]]
    rows=[]
    for idx in idxs:
        if all(abs(f[idx]-r['frequency_hz'])>5 for r in rows):
            rows.append({'rank':len(rows)+1,'frequency_hz':float(f[idx]),'power_density':float(p[idx])})
        if len(rows)>=n: break
    return rows

all_data=[]; intensity_rows=[]; spec_rows=[]
for label,drive,rec in recordings:
    d=load(rec); d['label']=label; d['drive_hz']=drive; all_data.append(d)
    intensity_rows.append(stats(label,'channel_mean',d['channel_mean'],d['fps'],drive))
    intensity_rows.append(stats(label,'raw11_mean',d['raw11_mean'],d['fps'],drive))
    f,p=psd(d['channel_mean'],d['fps'])
    peaks=top_peaks(f,p,12)
    for r in peaks:
        spec_rows.append({'label':label,'drive_hz':drive if drive is not None else '', 'kind':'top_peak','frequency_hz':r['frequency_hz'],'power_density':r['power_density'],'expected_alias_hz':alias_freq(drive,d['fps']) if drive is not None else ''})
    if drive is not None:
        al=alias_freq(drive,d['fps'])
        nf,npow=nearest(f,p,al)
        spec_rows.append({'label':label,'drive_hz':drive,'kind':'nearest_expected_alias','frequency_hz':nf,'power_density':npow,'expected_alias_hz':al})
        m=(f>=max(1,al-50))&(f<=min(d['fps']/2,al+50))
        idxs=np.flatnonzero(m); idx=idxs[np.argmax(p[m])]
        spec_rows.append({'label':label,'drive_hz':drive,'kind':'peak_within_alias_pm50Hz','frequency_hz':float(f[idx]),'power_density':float(p[idx]),'expected_alias_hz':al})
    d['spectrum_f']=f; d['spectrum_p']=p

with open(out/'intensity_fluctuation_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(intensity_rows[0].keys())); w.writeheader(); w.writerows(intensity_rows)
with open(out/'frequency_peak_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(spec_rows[0].keys())); w.writeheader(); w.writerows(spec_rows)

# plots
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
for d in all_data:
    t=np.arange(d['n_frames'])/d['fps']; y=d['channel_mean']; yn=(y-np.mean(y))/np.mean(y)*100
    ax.plot(t,yn,lw=.75,label=d['label'])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Intensity deviation from mean (%)'); ax.set_title('Last 4 recordings: angle-window intensity traces')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'normalised_intensity_traces_last4.png',dpi=200); plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
for d in all_data:
    ax.semilogy(d['spectrum_f'],d['spectrum_p'],lw=1,label=d['label'])
    if d['drive_hz'] is not None:
        ax.axvline(alias_freq(d['drive_hz'],d['fps']),ls='--',lw=.8,alpha=.35)
ax.set_xlim(1,all_data[0]['fps']/2); ax.set_xlabel('Observed frequency / alias (Hz)'); ax.set_ylabel('PSD'); ax.set_title('Last 4 recordings: angle-window spectra')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'spectra_last4.png',dpi=200); plt.close(fig)

for d in all_data:
    f,t,S=spectrogram(d['channel_mean']-np.mean(d['channel_mean']),fs=d['fps'],nperseg=min(512,d['n_frames']),noverlap=min(448,max(0,min(512,d['n_frames'])-1)),mode='psd',scaling='density')
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    pcm=ax.pcolormesh(t,f,10*np.log10(S+1e-15),shading='auto',cmap='magma')
    if d['drive_hz'] is not None:
        al=alias_freq(d['drive_hz'],d['fps']); ax.axhline(al,color='cyan',ls='--',lw=1,label=f'expected alias {al:.1f} Hz'); ax.legend()
    ax.set_ylim(0,d['fps']/2); ax.set_xlabel('Time (s)'); ax.set_ylabel('Observed frequency (Hz)'); ax.set_title(f"Spectrogram: {d['label']}")
    fig.colorbar(pcm,ax=ax,label='PSD (dB)')
    fig.savefig(out/f"spectrogram_{d['label']}.png",dpi=200); plt.close(fig)

summary={'assumption':'Newest four folders labelled in timestamp order as focus_pan, 3000 Hz, 3100 Hz, 2900 Hz. The fifth recent folder was not included.', 'recordings':[{'label':d['label'],'drive_hz':d['drive_hz'],'path':d['path'],'fps':d['fps'],'duration_s':d['duration_s'],'n_frames':d['n_frames'],'expected_alias_hz':alias_freq(d['drive_hz'],d['fps']) if d['drive_hz'] is not None else None} for d in all_data], 'intensity_rows':intensity_rows, 'spectrum_rows':spec_rows, 'output_dir':str(out)}
with open(out/'analysis_summary.json','w') as f: json.dump(summary,f,indent=2)
print('OUTPUT_DIR',out)
print('\nIntensity channel_mean:')
for r in intensity_rows:
    if r['trace']=='channel_mean':
        print(r['label'],'drive',r['drive_hz'],'duration',round(r['duration_s'],3),'p95-p5 pct',round(r['p95_minus_p5_percent_mean'],3),'p95-p5',round(r['p95_minus_p5'],3),'p99-p1 pct',round(r['p99_minus_p1_percent_mean'],3),'std',round(r['std'],3))
print('\nExpected aliases and peaks:')
for d in all_data:
    print('\n',d['label'],'drive',d['drive_hz'],'alias',None if d['drive_hz'] is None else round(alias_freq(d['drive_hz'],d['fps']),3))
    rows=[r for r in spec_rows if r['label']==d['label']]
    for r in rows[:8]:
        print(r['kind'],round(float(r['frequency_hz']),3),f"{float(r['power_density']):.6g}")
