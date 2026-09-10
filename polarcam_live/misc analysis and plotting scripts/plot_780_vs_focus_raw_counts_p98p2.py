import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending")
focus_rec=base/'rod_x1578_y502_20260722-144539_1784727939057247100'
drive_rec=base/'rod_x1578_y502_20260722-145208_1784728328945557800'
out=base/'latest_780Hz_vs_focus_pan_intensity_comparison'
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
    return fps, channel, raw11

def p98p2(y):
    p2,p98=np.percentile(y,[2,98])
    return float(p98-p2), float(p2), float(p98)

fps_f, focus_ch, focus_raw=load(focus_rec)
fps_d, drive_ch, drive_raw=load(drive_rec)

def make_plot(fname, title, focus_y, drive_y):
    fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True,sharex=False,sharey=True)
    for ax,label,y,fps,color in [
        (axs[0],'Focus pan',focus_y,fps_f,'tab:blue'),
        (axs[1],'780 Hz drive',drive_y,fps_d,'tab:orange'),
    ]:
        t=np.arange(len(y))/fps
        ax.plot(t,y,lw=.8,color=color)
        d,p2,p98=p98p2(y)
        ax.text(0.01,0.95,f'p98 - p2 = {d:.2f} counts',transform=ax.transAxes,va='top',ha='left',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ax.set_title(label)
        ax.set_ylabel('Raw intensity (counts)')
        ax.grid(alpha=.25)
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.savefig(out/fname,dpi=200)
    plt.close(fig)

make_plot('raw_counts_focus_pan_vs_780Hz_angle_window_p98p2.png','Focus pan vs latest 780 Hz driven intensity, angle-window mean',focus_ch,drive_ch)
make_plot('raw_counts_focus_pan_vs_780Hz_raw11_p98p2.png','Focus pan vs latest 780 Hz driven intensity, raw central 11x11 mean',focus_raw,drive_raw)
print('OUTPUT_DIR',out)
print('angle_window_focus_p98p2',p98p2(focus_ch)[0])
print('angle_window_780Hz_p98p2',p98p2(drive_ch)[0])
print('raw11_focus_p98p2',p98p2(focus_raw)[0])
print('raw11_780Hz_p98p2',p98p2(drive_raw)[0])
