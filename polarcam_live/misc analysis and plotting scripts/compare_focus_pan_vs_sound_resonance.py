import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending")
focus_rec=base/"rod_x1466_y1493_20260722-140413_1784725453970566300"
sound_on_rec=base/"rod_x1466_y1493_20260722-140139_1784725299651067500"
out=base/"focus_pan_vs_sound_resonance_20260722-140139_140413_analysis"
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
    return {'fps':fps,'channel_mean':channel,'raw11_mean':raw11,'n':len(channel),'duration':len(channel)/fps,'path':str(rec)}

def metrics(label,kind,tr,fps):
    p1,p5,p50,p95,p99=np.percentile(tr,[1,5,50,95,99]); mean=float(np.mean(tr))
    mn=float(np.min(tr)); mx=float(np.max(tr))
    return {'label':label,'trace':kind,'mean':mean,'std':float(np.std(tr,ddof=1)),'min':mn,'max':mx,'minmax':mx-mn,'minmax_percent_mean':float((mx-mn)/mean*100),'p95_minus_p5':float(p95-p5),'p95_minus_p5_percent_mean':float((p95-p5)/mean*100),'p99_minus_p1':float(p99-p1),'p99_minus_p1_percent_mean':float((p99-p1)/mean*100),'n_frames':int(len(tr)),'duration_s':float(len(tr)/fps)}

data={'sound_on_resonance':load(sound_on_rec),'manual_focus_pan':load(focus_rec)}
rows=[]
for label,d in data.items():
    rows.append(metrics(label,'channel_mean',d['channel_mean'],d['fps']))
    rows.append(metrics(label,'raw11_mean',d['raw11_mean'],d['fps']))
# ratios vs sound on
sound_rows={r['trace']:r for r in rows if r['label']=='sound_on_resonance'}
for r in rows:
    if r['label']=='manual_focus_pan':
        s=sound_rows[r['trace']]
        r['sound_fraction_of_focus_p95p5']=float(s['p95_minus_p5']/r['p95_minus_p5'])
        r['focus_over_sound_p95p5']=float(r['p95_minus_p5']/s['p95_minus_p5'])
        r['sound_fraction_of_focus_minmax']=float(s['minmax']/r['minmax'])
        r['focus_over_sound_minmax']=float(r['minmax']/s['minmax'])
with open(out/'focus_vs_sound_intensity_comparison.csv','w',newline='') as f:
    fields=sorted(set().union(*[r.keys() for r in rows]))
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(out/'focus_vs_sound_intensity_comparison.json','w') as f: json.dump({'recordings':data,'stats':rows,'output_dir':str(out)},f,indent=2,default=lambda x: '<array>')
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
for label,d in data.items():
    t=np.arange(d['n'])/d['fps']; y=d['channel_mean']
    ax.plot(t,(y-np.mean(y))/np.mean(y)*100,lw=.8,label=label)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Intensity deviation from mean (%)')
ax.set_title('Sound resonance vs manual focus pan, angle-window intensity')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'normalised_intensity_trace_sound_vs_focus.png',dpi=200); plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True,sharex=False)
for ax,(label,d) in zip(axs,data.items()):
    t=np.arange(d['n'])/d['fps']; y=d['channel_mean']
    ax.plot(t,y,lw=.8); ax.set_title(label); ax.set_ylabel('Angle-window mean intensity'); ax.grid(alpha=.25)
axs[-1].set_xlabel('Time (s)')
fig.savefig(out/'raw_intensity_traces_sound_and_focus.png',dpi=200); plt.close(fig)
print('OUTPUT_DIR',out)
for r in rows:
    print(r['label'],r['trace'],'p95-p5 pct',round(r['p95_minus_p5_percent_mean'],3),'p95-p5 counts',round(r['p95_minus_p5'],3),'minmax pct',round(r['minmax_percent_mean'],3),'minmax counts',round(r['minmax'],3))
print('\nComparison:')
for r in rows:
    if r['label']=='manual_focus_pan':
        print(r['trace'],'sound is',round(r['sound_fraction_of_focus_p95p5']*100,1),'% of focus p95-p5; focus/sound',round(r['focus_over_sound_p95p5'],2),'x')
        print(r['trace'],'sound is',round(r['sound_fraction_of_focus_minmax']*100,1),'% of focus minmax; focus/sound',round(r['focus_over_sound_minmax'],2),'x')
