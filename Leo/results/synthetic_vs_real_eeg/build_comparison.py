"""Compare actual recorded EEG with the current synthetic reconstruction data."""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, welch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from phoom.data.synthetic import test_recordings

folder=Path('/mnt/sde2/cs163/data/ds008065/sub-exp301/eeg')
stem='sub-exp301_task-f0s2_run-01'
header=folder/f'{stem}_eeg.vhdr'
text=header.read_text(encoding='utf-8-sig')
channels=re.findall(r'^Ch\d+=([^,]+),[^,]*,([\d.]+),([^\r\n]+)',text,re.M)
assert [r[0] for r in channels]==['FCzA1','FCz','A2']
assert all(r[2]=='µV' for r in channels)
resolution=np.array([float(r[1]) for r in channels])
fs=10000
with (folder/f'{stem}_events.tsv').open() as f:
    events=list(csv.DictReader(f,delimiter='\t'))
event=next(r for r in events if r['trial_type']=='stimulus_start' and r['stimulus_type']=='wizard')
onset=int(event['sample_corrected'])
start=onset+fs  # fixed choice: 1 second into the first Wizard trial
n=5000; guard=2*fs
raw=np.memmap(folder/f'{stem}_eeg.eeg',dtype='<f4',mode='r').reshape(-1,3)
seg=raw[start-guard:start+n].astype(np.float64)*resolution[None,:]/100
# Match the repository's two-ear referencing and gain correction.
real_long=.5*(seg[:,0]+seg[:,1]-seg[:,2])
real=real_long[guard:].copy();real-=real.mean()
truth,matrix,eeg=test_recordings(count=1,waveform_seed=96401,noise_seed=812001)
synthetic=eeg[.2][0].copy();synthetic-=synthetic.mean()
clean=matrix@truth[0]
sos=[butter(1,[30,2000],btype='bandpass',fs=fs,output='sos')]
sos.extend(tf2sos(*iirnotch(f,f/5,fs=fs)) for f in np.arange(60,2000,120))
sos=np.concatenate(sos)
real_filtered=sosfilt(sos,real_long)[guard:]
synth_filtered=sosfilt(sos,synthetic)
# Common-filter view uses the central 300 ms to avoid the synthetic filter startup.
central=slice(1000,4000)
rf=real_filtered[central];rf=rf-rf.mean()
sf=synth_filtered[central];sf=sf-sf.mean()
rms=lambda v:float(np.sqrt(np.mean(v**2)))
normalized_real=rf/rms(rf);normalized_synthetic=sf/rms(sf)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#1c2a45',
    'axes.labelcolor':'#56677b','xtick.color':'#56677b','ytick.color':'#56677b',
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
    'axes.spines.bottom':False,'axes.axisbelow':True,'axes.grid':True,'grid.color':'#e9edf4'})
colors=['#5264db','#cc774c']

def export(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(HERE/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)

def waveform_figure(arrays,t,names,unit,title,foot,name):
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.1,right=.96,bottom=.18,top=.81,hspace=.44)
    fig.text(.1,.92,title,fontsize=23,weight='bold')
    lim=1.08*max(np.max(np.abs(a)) for a in arrays)
    for ax,a,label,color in zip(axes,arrays,names,colors):
        ax.plot(t,a,color=color,lw=.85)
        ax.set_ylim(-lim,lim);ax.set_xlim(t[0],t[-1])
        ax.set_ylabel(unit);ax.set_title(label,loc='left',fontsize=13,pad=10,color=color)
        ax.tick_params(length=0)
    axes[-1].set_xlabel('Time within displayed window (ms)',labelpad=12)
    export(fig,name)

waveform_figure([synthetic,real],np.arange(n)/fs*1000,
    ['Synthetic EEG', 'Real EEG'],
    'EEG (µV)','Synthetic EEG versus recorded EEG',
    'Same 500 ms duration and amplitude scale. Mean removed; no added digital filtering. Real EEG: two-ear reference average, /100 gain correction as in repository.',
    '01_original_amplitudes')
waveform_figure([normalized_synthetic,normalized_real],np.arange(len(sf))/fs*1000,
    ['Synthetic EEG','Real EEG'],
    'EEG / RMS','Scaled',
    'Both: causal 30–2000 Hz bandpass + identical odd-60-Hz harmonic notches. Central 300 ms; independently normalized for shape comparison only.',
    '02_matched_processing')
# Quantify correlation using the full-resolution data, not downsampled display samples.
def acf(x,maxlag=100):
    x=x-x.mean();v=np.dot(x,x)
    return np.array([np.dot(x[:len(x)-lag],x[lag:])/v if lag else 1. for lag in range(maxlag+1)])
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
fig.subplots_adjust(left=.09,right=.97,bottom=.21,top=.8,wspace=.26)
fig.text(.09,.93,'Where the simulator differs',fontsize=23,weight='bold')
for y,label,c in [(synthetic,'Synthetic',colors[0]),(real,'Recorded',colors[1])]:
    f,p=welch(y/rms(y),fs=fs,nperseg=1000,noverlap=500)
    axes[0].semilogy(f,np.maximum(p,1e-12),color=c,lw=1.8,label=label)
    axes[1].plot(np.arange(101)/fs*1000,acf(y),color=c,lw=1.8,label=label)
axes[0].set(xlim=(0,5000),xlabel='Frequency (Hz)',ylabel='PSD of unit-RMS EEG (1/Hz)',title='Frequency content · before added filtering')
axes[1].set(xlabel='Lag (ms)',ylabel='Autocorrelation',title='Dependence between nearby samples')
axes[1].axhline(0,color='#b1bdcb',lw=.8)
axes[0].legend(frameon=False);axes[1].legend(frameon=False)
export(fig,'03_spectrum_and_correlation')
meta={'dataset':'OpenNeuro ds008065','subject':'sub-exp301','run':1,'story':'wizard',
      'real_source':str(header),'event_sample_corrected':onset,'display_start_sample':start,
      'display_start_s':start/fs,'display_duration_s':n/fs,'real_channel':'mean(FCzA1, FCz-A2)',
      'gain_correction':100,'scaling_note':'BrainVision resolution in microvolts divided by external preamp gain 100, following repository pipeline',
      'synthetic_waveform_seed':96401,'synthetic_noise_seed':812001,'synthetic_noise_sd_uv':.2,
      'synthetic_pulses':'20–460 ms, every 10 ms, unit amplitude',
      'original_rms_uv':{'synthetic':rms(synthetic),'recorded':rms(real)},
      'original_acf_1ms':{'synthetic':float(acf(synthetic)[10]),'recorded':float(acf(real)[10])},
      'limitations':'One prespecified example, not a population summary. Real trace contains auditory responses plus background and artifacts. Its clean ABR and SNR are unknown. Hardware and preprocessing differences contribute to the comparison; no reconstruction accuracy is inferred.'}
(HERE/'metadata.json').write_text(json.dumps(meta,indent=2))
np.savez_compressed(HERE/'example_data.npz',synthetic_uv=synthetic,real_uv=real,clean_synthetic_eeg_uv=clean,
                    synthetic_common_filtered_unit_rms=normalized_synthetic,real_common_filtered_unit_rms=normalized_real,fs=fs)
print(json.dumps(meta,indent=2))
