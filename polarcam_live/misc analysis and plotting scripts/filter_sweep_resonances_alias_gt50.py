import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

out = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\sound driven slide\pending\frequency_sweep_20Hz_20kHz_20260722-130559_130620_analysis")
rows=[]
with open(out/'sweep_alias_track_response.csv','r',newline='') as f:
    for r in csv.DictReader(f):
        rows.append({k:float(v) for k,v in r.items()})
true_f=np.array([r['drive_frequency_hz_assuming_linear_sweep'] for r in rows])
alias=np.array([r['expected_alias_hz'] for r in rows])
score=np.array([r['track_over_local_median'] for r in rows])
sm=np.array([r['smoothed_score'] for r in rows])
time=np.array([r['time_s'] for r in rows])
# Exclude near-DC alias where slow z drift/focus noise dominates, and near Nyquist edge.
valid=np.isfinite(sm)&(alias>=50)&(alias<=780)
score_v=sm.copy(); score_v[~valid]=np.nan
# smooth only valid sequence by interpolating invalid for plotting, but peak on valid points
x=np.arange(len(sm))
interp=sm.copy()
if valid.sum()>5:
    interp[~valid]=np.interp(x[~valid], x[valid], sm[valid])
    win=min(31, len(interp)//2*2-1)
    if win>=7:
        interp=savgol_filter(interp,win,3)
# detect peaks in valid-only response; invalid set to low
work=interp.copy(); work[~valid]=np.nanmin(interp[valid]) if valid.any() else 0
peaks,_=find_peaks(work, prominence=max(1.0,np.nanstd(work[valid])*0.5), distance=4)
# add top points, but group nearby frequencies within 150 Hz to avoid repeated same peak
candidate=np.unique(np.concatenate([peaks, np.argsort(np.where(valid, work, -np.inf))[-50:]]))
candidate=candidate[valid[candidate]]
candidate=candidate[np.argsort(work[candidate])[::-1]]
selected=[]
for idx in candidate:
    if all(abs(true_f[idx]-true_f[j])>150 for j in selected):
        selected.append(idx)
    if len(selected)>=15: break
peak_rows=[]
for idx in selected:
    peak_rows.append({'rank':len(peak_rows)+1,'time_s':float(time[idx]),'drive_frequency_hz_assuming_linear_sweep':float(true_f[idx]),'observed_alias_hz':float(alias[idx]),'track_over_local_median':float(score[idx]),'smoothed_score_excluding_low_alias':float(work[idx])})
with open(out/'sweep_resonance_candidates_excluding_alias_below_50Hz.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(peak_rows[0].keys())); w.writeheader(); w.writerows(peak_rows)
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
ax.plot(true_f, np.where(valid, score, np.nan), lw=.7, alpha=.4, label='track response, alias 50-780 Hz only')
ax.plot(true_f, np.where(valid, work, np.nan), lw=1.4, label='smoothed')
ax.scatter([r['drive_frequency_hz_assuming_linear_sweep'] for r in peak_rows[:10]],[r['smoothed_score_excluding_low_alias'] for r in peak_rows[:10]],color='red',s=30,zorder=4,label='candidate resonances')
for r in peak_rows[:10]:
    ax.annotate(f"{r['drive_frequency_hz_assuming_linear_sweep']/1000:.2f} kHz",(r['drive_frequency_hz_assuming_linear_sweep'],r['smoothed_score_excluding_low_alias']),fontsize=8,xytext=(4,4),textcoords='offset points')
ax.set_xlabel('Assumed drive frequency during linear sweep (Hz)')
ax.set_ylabel('Response along aliased sweep track')
ax.set_title('Estimated resonance response, excluding alias <50 Hz')
ax.grid(alpha=.25); ax.legend()
fig.savefig(out/'estimated_resonance_response_vs_drive_frequency_alias_gt50Hz.png',dpi=200); plt.close(fig)
print('Filtered candidates alias >50 Hz:')
for r in peak_rows[:12]:
    print(r['rank'], 'drive Hz', round(r['drive_frequency_hz_assuming_linear_sweep'],1), 'alias', round(r['observed_alias_hz'],1), 'score', round(r['smoothed_score_excluding_low_alias'],2))
