"""Known-truth, frozen-MLP reconstruction study with realistic acquisition noise."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('WANDB_BASE_URL','https://api.forge.coreweave.com')
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import realistic_eeg as data
from guidance_sweep import load_model,sample

ROOT=data.ROOT;OUT=ROOT/'Leo/results/realistic_eeg'
CHECKPOINT=ROOT/'Leo/runs/phoom_matched_10m/step_2000000_ema_0.005.pt'
GRID=[0.,.1,.3,1.,3.,6.25,10.,30.]


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(obs,scale,device):
    X=obs['matrix'];y=obs['eeg']/scale
    return {'A':torch.tensor(X.T@X/len(X),dtype=torch.float32,device=device),
            'b':torch.tensor(y@X/len(X),dtype=torch.float32,device=device),
            'yy':torch.tensor(np.mean(y*y,axis=1),dtype=torch.float32,device=device)}


def estimate(model,st,scale,g,draws,seed):
    predictions=sample(model,st,g,draws,seed,'cuda').mean(1)*scale
    return predictions


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Study output exists')
    started=time.monotonic();torch.set_num_threads(1)
    model,scale,cp=load_model(CHECKPOINT,'cuda')
    protocol={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              'model_updates':2000000,'ema_rate':.005,'scale_uv':scale,'validation_subject':'sub-exp302',
              'test_subject':'sub-exp304','validation_count':8,'test_count':10,'validation_draws':4,'test_draws':16,
              'guidance_grid':GRID,'validation_sampling_seed':881,'test_sampling_seed':991,
              'validation_wave_seed':97501,'test_wave_seed':97512,'conditions':list(data.CONDITIONS),
              'loss':'EEG residual RMS in model units, no noise-SD normalization',
              'new_training':False,'target':'one synthetic ABR, 4 kHz left stimulus train; final case is a null ABR',
              'unit_note':'microvolts; header resolution and original-author /100 preamp correction. No plot RMS normalization in amplitude figures.',
              'quiet_note':'pre-first-annotated-playback EEG; no independent acoustic verification',
              'selection':'select guidance separately for each known simulation condition using validation subject only; test subject untouched until freeze'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'STATUS.txt').write_text('RUNNING\n')
    import wandb
    run=wandb.init(entity='abr-deconv',project='abr-deconv-realistic-eeg',name='Real timings and quiet EEG · frozen 2m MLP',config=protocol,settings=wandb.Settings(disable_code=True))
    (OUT/'forge_url.txt').write_text(run.url);print('FORGE',run.url,flush=True)
    val,vmeta,_=data.build('sub-exp302',8,97501,815001)
    (OUT/'validation_sources.json').write_text(json.dumps(vmeta,indent=2))
    choices={};rows=[]
    for condition in data.CONDITIONS:
        obs=val[condition];st=stats(obs,scale,'cuda');scores=[]
        for g in GRID:
            try:
                prediction=estimate(model,st,scale,g,4,881)
                err=np.sqrt(np.mean((prediction-obs['truth'])**2,axis=1))
                score=float(err.mean())
            except RuntimeError:
                score=float('inf')
            row={'condition':condition,'guidance':g,'mean_rmse_uv':score};rows.append(row);scores.append(row)
            save_csv(OUT/'guidance_validation.csv',rows)
            print('VALIDATION',json.dumps(row),flush=True)
            run.log({'validation/condition':condition,'validation/guidance':g,'validation/rmse_uv':score})
        winner=min(scores,key=lambda r:r['mean_rmse_uv'])
        choices[condition]={'guidance':winner['guidance'],'validation_rmse_uv':winner['mean_rmse_uv'],
                            'boundary':winner['guidance'] in (min(GRID),max(GRID))}
    (OUT/'selection.json').write_text(json.dumps(choices,indent=2))
    (OUT/'SELECTION_FROZEN').write_text('Frozen before test subject generation and scoring.\n')
    test,tmeta,extra=data.build('sub-exp304',10,97512,815012)
    (OUT/'test_sources.json').write_text(json.dumps(tmeta,indent=2))
    records=[];physics=[];predictions={};examples={**extra,'truth':test['legacy_white']['truth']}
    time_ms=(np.arange(256)-50)/10;peak=(time_ms>=2)&(time_ms<=9)
    for condition in data.CONDITIONS:
        obs=test[condition];st=stats(obs,scale,'cuda')
        for label,g in [('fixed_6_25',6.25),('validation_selected',choices[condition]['guidance'])]:
            pred=estimate(model,st,scale,g,16,991)
            predictions[condition+'__'+label]=pred
            for i in range(len(pred)):
                true=obs['truth'][i];p=pred[i]
                null=bool(np.all(true==0))
                ti=np.argmax(true[peak]);pi=np.argmax(p[peak])
                records.append({'condition':condition,'method':label,'guidance':g,'record':i,'null_control':null,
                                'rmse_uv':float(np.sqrt(np.mean((p-true)**2))),
                                'peak_amplitude_error_uv':float(abs(np.max(p[peak])-np.max(true[peak]))),
                                'peak_latency_error_ms':float(abs(time_ms[peak][pi]-time_ms[peak][ti])) if not null else '',
                                'noise_rms_uv':float(data.rms(obs['noise'][i]))})
            save_csv(OUT/'test_metrics.csv',records)
            print('TEST',condition,label,float(np.sqrt(np.mean((pred-obs['truth'])**2,axis=1)).mean()),flush=True)
        for i in range(len(obs['truth'])):
            physics.append({'condition':condition,'record':i,'eeg_rms_uv':float(data.rms(obs['eeg'][i])),
                            'noise_rms_uv':float(data.rms(obs['noise'][i])),
                            'clean_eeg_rms_uv':float(data.rms(obs['clean_eeg'][i]))})
        examples[condition]=obs['eeg'][0];examples[condition+'__clean']=obs['clean_eeg'][0]
        run.log({f'test/{condition}/rmse_uv':float(np.mean([r['rmse_uv'] for r in records if r['condition']==condition and r['method']=='validation_selected']))})
    save_csv(OUT/'signal_amplitudes.csv',physics)
    np.savez_compressed(OUT/'examples.npz',**examples)
    np.savez_compressed(OUT/'predictions.npz',truth=test['legacy_white']['truth'],**predictions)
    (OUT/'runtime.json').write_text(json.dumps({'seconds':time.monotonic()-started},indent=2))
    from realistic_eeg_report import render
    report=render(OUT)
    for p in sorted((OUT/'figures').glob('*.png')):run.log({f'figures/{p.stem}':wandb.Image(str(p))})
    for p in OUT.glob('*'):
        if p.suffix in ('.html','.csv','.json'):run.save(str(p),base_path=str(OUT),policy='now')
    run.finish();(OUT/'STATUS.txt').write_text('COMPLETE\n');print('COMPLETE',report,flush=True)


if __name__=='__main__':main()
