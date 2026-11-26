"""Validation-only spectral noise weighting; compare on the frozen test protocol."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.signal import welch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import realistic_eeg as data
from realistic_eeg_study import OUT,CHECKPOINT,stats,estimate,save_csv
from guidance_sweep import load_model
import torch


def transform(obs,w):
    result=dict(obs)
    result['matrix']=np.fft.irfft(np.fft.rfft(obs['matrix'],axis=0)*w[:,None],n=data.N,axis=0)
    result['eeg']=np.fft.irfft(np.fft.rfft(obs['eeg'],axis=1)*w[None,:],n=data.N,axis=1)
    return result


def main():
    torch.set_num_threads(1)
    model,scale,_=load_model(CHECKPOINT,'cuda')
    val,_,_=data.build('sub-exp302',8,97501,815001)
    obs=val['real_quiet_native']
    f,p=welch(obs['noise'],fs=data.FS,nperseg=1000,noverlap=500,axis=1)
    p=p.mean(0);target=np.fft.rfftfreq(data.N,1/data.FS)
    psd=np.interp(target,f,p);band=(target>=30)&(target<=2000)
    floor=float(np.quantile(psd[band],.1));w=np.zeros_like(psd)
    w[band]=1/np.sqrt(np.maximum(psd[band],floor))
    w[band]/=np.sqrt(np.mean(w[band]**2))
    np.savez_compressed(OUT/'validation_noise_weights.npz',frequency_hz=target,psd=psd,weights=w)
    st=stats(transform(obs,w),scale,'cuda');rows=[]
    for g in (0.,.1,.3,1.,3.,6.25,10.,30.,100.):
        pred=estimate(model,st,scale,g,4,881)
        score=float(np.sqrt(np.mean((pred-obs['truth'])**2,axis=1)).mean())
        row={'guidance':g,'validation_rmse_uv':score};rows.append(row)
        save_csv(OUT/'weighted_validation.csv',rows);print('WEIGHTED_VALIDATION',json.dumps(row),flush=True)
    best=min(rows,key=lambda r:r['validation_rmse_uv'])
    selection={'guidance':best['guidance'],'validation_rmse_uv':best['validation_rmse_uv'],
               'noise_source':'only validation subject exp302 quiet EEG',
               'definition':'inverse square root validation PSD with 10th percentile band floor; unit RMS weight within 30–2000 Hz; outside-band weights zero',
               'note':'accounts for spectral noise shape; does not use the test noise SD or test truth for tuning'}
    (OUT/'weighted_selection.json').write_text(json.dumps(selection,indent=2))
    (OUT/'WEIGHTED_SELECTION_FROZEN').write_text('Weighted loss fully specified and guidance frozen before test scoring.\n')
    test,_,_=data.build('sub-exp304',10,97512,815012);original=test['real_quiet_native']
    st=stats(transform(original,w),scale,'cuda')
    pred=estimate(model,st,scale,best['guidance'],16,991)
    records=[];tm=(np.arange(256)-50)/10;win=(tm>=2)&(tm<=9)
    for i,(p,t) in enumerate(zip(pred,original['truth'])):
        null=bool(np.all(t==0))
        records.append({'record':i,'null_control':null,'guidance':best['guidance'],
                        'rmse_uv':float(np.sqrt(np.mean((p-t)**2))),
                        'peak_amplitude_error_uv':float(abs(np.max(p[win])-np.max(t[win]))),
                        'peak_latency_error_ms':float(abs(tm[win][np.argmax(p[win])]-tm[win][np.argmax(t[win])])) if not null else ''})
    save_csv(OUT/'weighted_test_metrics.csv',records)
    np.savez_compressed(OUT/'weighted_predictions.npz',truth=original['truth'],prediction=pred)
    print('WEIGHTED_TEST',float(np.mean([r['rmse_uv'] for r in records])),flush=True)
    fixed=estimate(model,st,scale,6.25,16,991)
    fixed_rows=[{'record':i,'guidance':6.25,'rmse_uv':float(np.sqrt(np.mean((p-t)**2)))}
                for i,(p,t) in enumerate(zip(fixed,original['truth']))]
    save_csv(OUT/'weighted_fixed_test_metrics.csv',fixed_rows)


if __name__=='__main__':main()
