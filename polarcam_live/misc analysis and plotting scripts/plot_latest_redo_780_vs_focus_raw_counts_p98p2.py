import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

base=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending")
focus_rec=base/'rod_x1578_y502_20260722-151029_1784729429758821400'
drive_rec=base/'rod_x1578_y502_20260722-151019_1784729419566842700'
out=base/'latest_redo_780Hz_vs_focus_pan_intensity_comparison_151019_151029'
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
    return {'fps':fps,'channel':channel,'raw11':raw11,'path':str(rec),'n_frames':len(channel),'duration_s':len(channel)/fps}

def p98p2(y):
    p2,p98=np.percentile(y,[2,98])
    return float(p98-p2), float(p2), float(p98)

def psd_peak(y,fps):
    yy=y-np.mean(y); nperseg=min(len(yy),4096)
    f,p=welch(yy,fs=fps,nperseg=nperseg,noverlap=min(nperseg//2,2048),scaling='density')
    m=(f>=1)&(f<=fps/2)
    idxs=np.flatnonzero(m); idx=idxs[np.argmax(p[m])]
    alias=780.0
    nearest_idx=int(np.argmin(np.abs(f-alias)))
    return float(f[idx]),float(p[idx]),float(f[nearest_idx]),float(p[nearest_idx])

focus=load(focus_rec); drive=load(drive_rec)

def make_plot(fname,title,key):
    fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True,sharex=False,sharey=True)
    for ax,label,d,color in [(axs[0],'780 Hz drive',drive,'tab:orange'),(axs[1],'Focus pan',focus,'tab:blue')]:
        y=d[key]; t=np.arange(len(y))/d['fps']
        ax.plot(t,y,lw=.8,color=color)
        diff,_,_=p98p2(y)
        ax.text(0.01,0.95,f'p98 - p2 = {diff:.2f} counts',transform=ax.transAxes,va='top',ha='left',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ax.set_title(label); ax.set_ylabel('Raw intensity (counts)'); ax.grid(alpha=.25)
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.savefig(out/fname,dpi=200); plt.close(fig)

make_plot('raw_counts_780Hz_vs_focus_pan_angle_window_p98p2.png','780 Hz drive vs focus pan, angle-window mean', 'channel')
make_plot('raw_counts_780Hz_vs_focus_pan_raw11_p98p2.png','780 Hz drive vs focus pan, raw central 11x11 mean', 'raw11')
summary={'recordings':{'780Hz_drive':drive,'focus_pan':focus},'p98_minus_p2_counts':{'angle_window':{'780Hz_drive':p98p2(drive['channel'])[0],'focus_pan':p98p2(focus['channel'])[0]},'raw11':{'780Hz_drive':p98p2(drive['raw11'])[0],'focus_pan':p98p2(focus['raw11'])[0]}},'drive_spectrum_channel':{'dominant_peak_hz':psd_peak(drive['channel'],drive['fps'])[0],'dominant_peak_psd':psd_peak(drive['channel'],drive['fps'])[1],'nearest_780_hz_bin':psd_peak(drive['channel'],drive['fps'])[2],'nearest_780_hz_psd':psd_peak(drive['channel'],drive['fps'])[3]},'output_dir':str(out)}
# Remove arrays from json
summary_json={k:v for k,v in summary.items() if k!='recordings'}
summary_json['recordings']={'780Hz_drive':{'path':drive['path'],'fps':drive['fps'],'n_frames':drive['n_frames'],'duration_s':drive['duration_s']},'focus_pan':{'path':focus['path'],'fps':focus['fps'],'n_frames':focus['n_frames'],'duration_s':focus['duration_s']}}
with open(out/'comparison_summary.json','w') as f: json.dump(summary_json,f,indent=2)
print('OUTPUT_DIR',out)
print('angle_window_780Hz_p98p2',p98p2(drive['channel'])[0])
print('angle_window_focus_p98p2',p98p2(focus['channel'])[0])
print('raw11_780Hz_p98p2',p98p2(drive['raw11'])[0])
print('raw11_focus_p98p2',p98p2(focus['raw11'])[0])
print('780Hz spectrum dominant/nearest780',psd_peak(drive['channel'],drive['fps']))
