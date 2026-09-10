import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
out=Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\frequency_sweep_20Hz_20kHz_20260722-130559_130620_analysis")
rows=[]
with open(out/'sweep_alias_track_response.csv','r',newline='') as f:
    for r in csv.DictReader(f): rows.append({k:float(v) for k,v in r.items()})
time=np.array([r['time_s'] for r in rows]); true_f=np.array([r['drive_frequency_hz_assuming_linear_sweep'] for r in rows]); alias=np.array([r['expected_alias_hz'] for r in rows]); score=np.array([r['track_over_local_median'] for r in rows]); sm=np.array([r['smoothed_score'] for r in rows])
for cutoff in [100,150,200]:
    valid=np.isfinite(sm)&(alias>=cutoff)&(alias<=760)
    x=np.arange(len(sm)); interp=sm.copy()
    interp[~valid]=np.interp(x[~valid],x[valid],sm[valid])
    win=min(31,len(interp)//2*2-1)
    if win>=7: interp=savgol_filter(interp,win,3)
    work=interp.copy(); work[~valid]=np.nanmin(interp[valid])
    peaks,_=find_peaks(work,prominence=max(0.75,np.nanstd(work[valid])*0.45),distance=4)
    cand=np.unique(np.concatenate([peaks,np.argsort(np.where(valid,work,-np.inf))[-80:]]))
    cand=cand[valid[cand]]; cand=cand[np.argsort(work[cand])[::-1]]
    selected=[]
    for idx in cand:
        if all(abs(true_f[idx]-true_f[j])>200 for j in selected): selected.append(idx)
        if len(selected)>=15: break
    rows_out=[]
    for idx in selected:
        rows_out.append({'rank':len(rows_out)+1,'time_s':float(time[idx]),'drive_frequency_hz_assuming_linear_sweep':float(true_f[idx]),'observed_alias_hz':float(alias[idx]),'track_over_local_median':float(score[idx]),'smoothed_score':float(work[idx])})
    csv_path=out/f'sweep_resonance_candidates_excluding_alias_below_{cutoff}Hz.csv'
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    ax.plot(true_f,np.where(valid,score,np.nan),lw=.7,alpha=.35,label=f'track response, alias {cutoff}-760 Hz')
    ax.plot(true_f,np.where(valid,work,np.nan),lw=1.4,label='smoothed')
    ax.scatter([r['drive_frequency_hz_assuming_linear_sweep'] for r in rows_out[:10]],[r['smoothed_score'] for r in rows_out[:10]],s=30,color='red',zorder=3,label='candidates')
    for r in rows_out[:10]: ax.annotate(f"{r['drive_frequency_hz_assuming_linear_sweep']/1000:.2f} kHz",(r['drive_frequency_hz_assuming_linear_sweep'],r['smoothed_score']),fontsize=8,xytext=(4,4),textcoords='offset points')
    ax.set_xlabel('Assumed drive frequency (Hz)'); ax.set_ylabel('Response along aliased sweep track'); ax.set_title(f'Estimated resonance response, excluding alias <{cutoff} Hz'); ax.grid(alpha=.25); ax.legend()
    fig.savefig(out/f'estimated_resonance_response_vs_drive_frequency_alias_gt{cutoff}Hz.png',dpi=200); plt.close(fig)
    print('\ncutoff',cutoff)
    for r in rows_out[:10]: print(r['rank'],round(r['drive_frequency_hz_assuming_linear_sweep'],1),round(r['observed_alias_hz'],1),round(r['smoothed_score'],2))
