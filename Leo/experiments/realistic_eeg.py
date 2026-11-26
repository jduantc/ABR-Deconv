"""Real pulse timings and event-defined quiet EEG for known-truth reconstruction.

One target regressor (4 kHz left); other evoked responses are not simulated.
The dataset API retains physical amplitudes, measured background, processing,
and the linear forward operator used by reconstruction.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter,iirnotch,tf2sos,sosfilt

import h5py
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legacy_phoom.data.synthetic import generate_responses,stimulus_matrix

DATA=Path('/mnt/sde2/cs163/data/ds008065')
FS=10000;N=5000;GUARD=10000
CONDITIONS=('legacy_white','regular_matched_white','real_matched_white','real_quiet_half','real_quiet_native','real_quiet_double','real_quiet_varying')
LABELS={'legacy_white':'Original simulator','regular_matched_white':'Regular pulses + matched white noise',
        'real_matched_white':'Real pulses + matched white noise','real_quiet_half':'Real pulses + quiet EEG ×0.5',
        'real_quiet_native':'Real pulses + quiet EEG ×1','real_quiet_double':'Real pulses + quiet EEG ×2',
        'real_quiet_varying':'Real pulses + varying quiet EEG'}


def rms(x,axis=None):return np.sqrt(np.mean(np.asarray(x)**2,axis=axis))


def filters():
    parts=[butter(1,[30,2000],btype='bandpass',fs=FS,output='sos')]
    parts.extend(tf2sos(*iirnotch(f,f/5,fs=FS)) for f in np.arange(60,2000,120))
    return np.concatenate(parts)


def source(subject):
    stem=f'{subject}_task-f0s2_run-01';folder=DATA/subject/'eeg'
    header=folder/f'{stem}_eeg.vhdr'
    txt=header.read_text(encoding='utf-8-sig')
    cols=re.findall(r'^Ch\d+=([^,]+),[^,]*,([\d.]+),([^\r\n]+)',txt,re.M)
    assert [c[0] for c in cols]==['FCzA1','FCz','A2'] and all(c[2]=='µV' for c in cols)
    assert 'SamplingInterval=100' in txt
    resolution=np.array([float(c[1]) for c in cols])
    raw=np.memmap(folder/f'{stem}_eeg.eeg',dtype='<f4',mode='r').reshape(-1,3)
    with (folder/f'{stem}_events.tsv').open() as f:events=list(csv.DictReader(f,delimiter='\t'))
    return {'raw':raw,'resolution':resolution,'events':events,'header':str(header)}


def eeg_segment(src,start,length=N,kind='filtered',envelope=None):
    assert start>=GUARD
    seg=src['raw'][start-GUARD:start+length].astype(np.float64)*src['resolution'][None,:]/100.
    y=.5*(seg[:,0]+seg[:,1]-seg[:,2])
    if envelope is not None:y=y*envelope
    if kind=='filtered':y=sosfilt(filters(),y)
    y=y[GUARD:];return y-y.mean()


def pulse_source(src):
    starts=[e for e in src['events'] if e['trial_type']=='stimulus_start']
    ends=[e for e in src['events'] if e['trial_type']=='stimulus_end']
    intervals=[]
    for s in starts:
        later=[int(e['sample'])-int(s['sample']) for e in ends if 9*FS<int(e['sample'])-int(s['sample'])<10.5*FS]
        if later:intervals.append(min(later))
    clock=float(np.mean(9.98*FS/np.array(intervals)))
    event=next(e for e in starts if e['stimulus_type']=='wizard')
    token=int(float(event['token']))+1
    path=DATA/'stimuli/exp3_f0s2/wizard'/f'wizard_shifted_{token:04d}.hdf5'
    with h5py.File(path) as f:
        root=f['expyfun'];fs_audio=float(root['key_fs'][()].reshape(-1)[0])
        inds=np.asarray(root['key_pulseinds_true']['idx_6'][()])
    times=np.floor(inds/fs_audio/clock*FS).astype(int)
    train=np.zeros(GUARD+N,dtype=np.float32)
    usable=times[(times>=0)&(times<len(train))];np.add.at(train,usable,1.)
    regular=np.zeros_like(train);regular[GUARD+np.arange(200,N-300,100)]=1
    mats={}
    for name,p in [('real',train),('regular',regular)]:
        context=stimulus_matrix(p).numpy().astype(np.float64)
        raw=context[GUARD:]
        processed=sosfilt(filters(),context,axis=0)[GUARD:]
        mats[name]=processed-processed.mean(0)
        if name=='regular':mats['legacy']=raw-raw.mean(0)
    return mats,train[GUARD:],regular[GUARD:],{'clock_adjust':clock,'stimulus_file':str(path),
        'stimulus_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'regressor_index':6,'regressor':'4 kHz left','window_offset_s':1.,
        'real_pulse_count':int(train[GUARD:].sum()),'regular_pulse_count':45,
        'real_interpulse_ms':(np.diff(np.flatnonzero(train[GUARD:]))/FS*1000).tolist(),
        'first_playback_sample':int(starts[0]['sample_corrected']),
        'active_sample':int(event['sample_corrected'])+GUARD}


def build(subject,count,wave_seed,noise_seed):
    src=source(subject);mats,pulses,regular,meta=pulse_source(src)
    waves,zeros=generate_responses(100,wave_seed)
    truth=waves[~zeros][:count].astype(np.float64)*.1
    truth[-1]=0 # explicit absent-response control
    quiet=[];varying=[];qmeta=[]
    for i in range(count):
        start=int((2.5+.5*i)*FS)
        if start+N>meta['first_playback_sample']-FS//2:raise ValueError('Quiet window too close to playback')
        quiet.append(eeg_segment(src,start))
        t=np.arange(GUARD+N)/FS
        envelope=1.+.5*np.sin(2*np.pi*2*t+i*.7)
        varying.append(eeg_segment(src,start,envelope=envelope))
        qmeta.append({'record':i,'start_sample':start,'end_sample':start+N,'rms_uv':float(rms(quiet[-1])),
                      'no_annotated_stimulus':True})
    quiet=np.stack(quiet);varying=np.stack(varying)
    gen=np.random.default_rng(noise_seed)
    rawwhite=gen.normal(0,.2,(count,GUARD+N))
    legacy=rawwhite[:,GUARD:];legacy-=legacy.mean(1,keepdims=True)
    white=sosfilt(filters(),rawwhite,axis=1)[:,GUARD:];white-=white.mean(1,keepdims=True)
    white*=rms(quiet,axis=1)[:,None]/rms(white,axis=1)[:,None]
    datasets={}
    for condition in CONDITIONS:
        mat=mats['legacy' if condition=='legacy_white' else 'regular' if condition=='regular_matched_white' else 'real']
        noise=legacy if condition=='legacy_white' else white if 'matched_white' in condition else (
            quiet*.5 if condition=='real_quiet_half' else quiet*2 if condition=='real_quiet_double' else
            varying if condition=='real_quiet_varying' else quiet)
        clean=truth@mat.T;eeg=clean+noise
        datasets[condition]={'matrix':mat,'truth':truth,'eeg':eeg,'noise':noise,'clean_eeg':clean}
    meta.update({'subject':subject,'header':src['header'],'quiet_windows':qmeta,
                 'gain_correction':100,'channel':'mean(FCzA1, FCz-A2)',
                 'target_units':'microvolts; synthetic ABR amplitudes multiplied by 0.1 as existing generator',
                 'filter':'causal first-order Butterworth 30–2000 Hz; 60 Hz odd-harmonic notches, width 5 Hz',
                 'note':'Quiet means before the first annotated playback, not independently acoustically verified silence. Single target band; other evoked bands omitted.'})
    active=eeg_segment(src,meta['active_sample'])
    return datasets,meta,{'real_eeg':active,'quiet_eeg':quiet,'real_pulses':pulses,'regular_pulses':regular}
