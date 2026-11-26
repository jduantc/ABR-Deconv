"""Reconstruction-selected MLP/U-Net study with resumable 2m-update runs.

Same data, v-prediction target, 200-step cosine schedule, batch size and
unnormalized EEG residual. Short LR screens are continued, not restarted.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('WANDB_BASE_URL', 'https://api.forge.coreweave.com')
os.environ.setdefault('WANDB_INIT_TIMEOUT', '60')

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Leo'))
sys.path.insert(0, str(ROOT / 'Leo/experiments'))
sys.path.insert(0, str(ROOT))
from model import Denoiser, UNetDenoiser
from diffusion import cosine_schedule, make_training_pair
from legacy_phoom.data.synthetic import waveforms_uv, NOISE_UV
from guidance_sweep import make_observations, sample, errors

LRS = (1e-4, 2e-4, 4e-4)
GUIDANCES = (0., 1., 3., 6.25, 12., 24.)
PROJECT = 'abr-deconv-mlp-unet-2m'
BASELINE = ROOT / 'Leo/runs/phoom_matched_10m/step_2000000_ema_0.005.pt'


def save_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def model_for(name, device):
    return {'mlp':Denoiser,'unet':UNetDenoiser}[name]().to(device)


def checkpoint_save(path, model, ema, opt, gen, step, architecture, lr, scale, rows):
    tmp=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                'generator':gen.get_state(),'step':step,'architecture':architecture,
                'lr':lr,'scale':scale,'rows':rows},tmp)
    tmp.replace(path)


def evaluate(model, obs, scale, guidance, draws, seed=311, chunk=6):
    """Chunk recordings to bound memory; all methods share corresponding RNG seeds."""
    outputs=[]
    for i in range(0,len(obs['truth']),chunk):
        sub={'A':obs['A'],'b':obs['b'][i:i+chunk],'yy':obs['yy'][i:i+chunk]}
        outputs.append(sample(model,sub,guidance,draws,seed+i,next(model.parameters()).device))
    values=np.concatenate(outputs)
    err,pred=errors(values,obs['truth'],scale)
    return err,pred


def train_to(name, lr, end, cfg, out, x, scale, alpha, run, validation=None):
    dest=out/name/f'lr_{lr:.0e}'
    dest.mkdir(parents=True,exist_ok=True)
    state_path=dest/'last.pt'
    torch.manual_seed(cfg['model_seed'])
    model=model_for(name,cfg['device'])
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    gen=torch.Generator(device=cfg['device']).manual_seed(cfg['batch_seed'])
    first=0; rows=[]
    if state_path.exists():
        saved=torch.load(state_path,map_location=cfg['device'],weights_only=False)
        if saved['architecture']!=name or saved['lr']!=lr or saved['scale']!=scale:
            raise ValueError('Existing checkpoint does not match experiment')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']);gen.set_state(saved['generator'].cpu())
        first=int(saved['step']); rows=saved['rows']
    if first>end:
        raise ValueError('Cannot train an existing checkpoint backwards')
    loss_sum=0.; count=0; started=time.monotonic()
    for step in range(first+1,end+1):
        opt.param_groups[0]['lr']=lr*min(1.,step/500)
        batch=x[torch.randint(len(x),(cfg['batch_size'],),generator=gen,device=cfg['device'])]
        t=torch.randint(200,(len(batch),),generator=gen,device=cfg['device'])
        noise=torch.randn(batch.shape,generator=gen,device=cfg['device'])
        noisy,target=make_training_pair(batch,t,noise,alpha)
        loss=(model(noisy,t)-target).square().mean()
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {name}, {lr}, {step}')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()): e.lerp_(p,.005)
        loss_sum += loss.detach();count+=1
        if step % cfg['log_every']==0 or step==end:
            v=float(loss_sum/count); loss_sum=0.;count=0
            row={'step':step,'v_mse':v,'elapsed_since_resume_s':time.monotonic()-started}
            rows.append(row);save_csv(dest/'training.csv',rows)
            print('TRAIN',name,lr,step,'v_mse',v,flush=True)
            (out/'PROGRESS.txt').write_text(f'Training {name} lr={lr:g}: {step:,}/{end:,} updates; last resumable state may be up to {cfg["save_every"]:,} updates earlier.\n')
            if run: run.log({'update':step,f'{name}/lr_{lr:.0e}/v_mse':v})
        if step % cfg['save_every']==0 or step==end:
            checkpoint_save(state_path,model,ema,opt,gen,step,name,lr,scale,rows)
            (out/'PROGRESS.txt').write_text(f'Training {name} lr={lr:g}: {step:,}/{end:,} updates; resumable state saved.\n')
        if validation is not None and step in cfg['milestones']:
            torch.save({'state':ema.state_dict(),'step':step,'architecture':name,'lr':lr,'scale':scale},dest/f'step_{step}_ema.pt')
            err,_=evaluate(ema,validation,scale,cfg['fixed_screen_guidance'],cfg['tune_draws'])
            row={'model':name,'lr':lr,'step':step,'guidance':cfg['fixed_screen_guidance'],'validation_rmse_uv':float(err.mean())}
            save_json(dest/f'validation_{step}.json',row)
            print('CHECKPOINT_VALIDATION',json.dumps(row),flush=True)
            if run:run.log({'update':step,f'{name}/validation_rmse_uv':row['validation_rmse_uv']})
    return ema


def guide_search(model,obs,scale,candidates,draws,stage,model_name,out,run):
    rows=[]
    for g in candidates:
        err,_=evaluate(model,obs,scale,g,draws)
        row={'stage':stage,'model':model_name,'guidance':g,'rmse_uv':float(err.mean()),
             **{f'rmse_{s}_uv':float(err[obs['sigmas']==s].mean()) for s in NOISE_UV}}
        rows.append(row)
        print('GUIDANCE',json.dumps(row),flush=True)
        if run: run.log({f'{model_name}/{stage}/guidance':g,f'{model_name}/{stage}/rmse_uv':row['rmse_uv']})
    save_csv(out/f'{model_name}_{stage}_guidance.csv',rows)
    return min(rows,key=lambda r:r['rmse_uv'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'Leo/runs/mlp_unet_2m')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--no-wandb',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg={'architectures':['mlp','unet'],'lr_grid':LRS,'tune_updates':2 if args.smoke else 20000,
         'refine_updates':3 if args.smoke else 100000,
         'total_updates':4 if args.smoke else 2000000,'batch_size':8 if args.smoke else 64,
         'train_count':20000,'train_seed':5000,'model_seed':0,'batch_seed':1000,
         'val_wave_seed':95111,'val_noise_seed':711001,
         'test_wave_seed':95112 if args.smoke else 95122,
         'test_noise_seed':711201 if args.smoke else 711301,
         'validation_count':1 if args.smoke else 24,'test_count':1 if args.smoke else 24,
         'tune_draws':1 if args.smoke else 4,'final_val_draws':1 if args.smoke else 8,
         'test_draws':1 if args.smoke else 32,'fixed_screen_guidance':6.25,'guidance_grid':[0.,6.25,12.] if args.smoke else GUIDANCES,
         'ema_rate':.005,'weight_decay':0.,'warmup':500,'cosine_diffusion_steps':200,
         'loss':'v MSE; reconstruction EEG residual RMS without noise SD normalization',
         'save_every':2 if args.smoke else 25000,'log_every':2 if args.smoke else 2000,
         'milestones':[4] if args.smoke else [250000,500000,1000000,1500000,2000000],
         'device':args.device,'test_selection':'freeze all choices before test construction',
         'baseline_checkpoint':str(BASELINE),'baseline_sha256':hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
         'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
             Path(__file__), ROOT/'Leo/experiments/mlp_unet_report.py',
             ROOT/'Leo/experiments/guidance_sweep.py',ROOT/'Leo/model.py',
             ROOT/'Leo/diffusion.py',ROOT/'Leo/experiments/legacy_phoom/data/synthetic.py')},
         'note':'Screen three LRs at 20k, advance top two to 100k, continue best same state to 2m; early selection can favor short training. MLP baseline used batch 256.'}
    cfg=json.loads(json.dumps(cfg)); config_path=out/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:
        raise ValueError('Protocol differs from existing output; choose a new output')
    save_json(config_path,cfg)
    (out/'STATUS.txt').write_text('RUNNING\n')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    run=None
    if not args.no_wandb:
        import wandb
        rid_path=out/'wandb_id.txt'
        rid=rid_path.read_text().strip() if rid_path.exists() else os.urandom(8).hex()
        rid_path.write_text(rid)
        run=wandb.init(entity='abr-deconv',project=PROJECT,id=rid,resume='allow',name='MLP vs U-Net · reconstruction',config=cfg,settings=wandb.Settings(disable_code=True))
        run.define_metric('update')
        for name in cfg['architectures']:
            for lr in LRS:
                run.define_metric(f'{name}/lr_{lr:.0e}/v_mse',step_metric='update')
            run.define_metric(f'{name}/validation_rmse_uv',step_metric='update')
        print('FORGE',run.url,flush=True)
    x_uv=waveforms_uv(20000,5000);scale=float(x_uv.std())
    x=torch.tensor(x_uv/scale,dtype=torch.float32,device=args.device)
    alpha=cosine_schedule().to(args.device)
    val=make_observations(cfg['validation_count'],cfg['val_wave_seed'],cfg['val_noise_seed'],scale,args.device)
    selections={}
    selection_path=out/'training_selection.json'
    if selection_path.exists():selections=json.loads(selection_path.read_text())
    for name in cfg['architectures']:
        pilot=[]
        for lr in LRS:
            result_path=out/name/f'lr_{lr:.0e}'/'screen.json'
            if result_path.exists():
                row=json.loads(result_path.read_text())
            else:
                ema=train_to(name,lr,cfg['tune_updates'],cfg,out,x,scale,alpha,run)
                err,_=evaluate(ema,val,scale,cfg['fixed_screen_guidance'],cfg['tune_draws'])
                row={'architecture':name,'lr':lr,'updates':cfg['tune_updates'],'guidance':cfg['fixed_screen_guidance'],
                     'validation_rmse_uv':float(err.mean())}
                save_json(result_path,row)
                print('SCREEN',json.dumps(row),flush=True)
                if run:run.log({f'{name}/screen/lr':lr,f'{name}/screen/rmse_uv':row['validation_rmse_uv']})
                del ema
            pilot.append(row)
        save_csv(out/f'{name}_screen.csv',pilot)
        finalists=sorted(pilot,key=lambda r:r['validation_rmse_uv'])[:2]
        refined=[]
        for candidate in finalists:
            lr=candidate['lr']; result_path=out/name/f'lr_{lr:.0e}'/'refined_screen.json'
            if result_path.exists():
                row=json.loads(result_path.read_text())
            else:
                ema=train_to(name,lr,cfg['refine_updates'],cfg,out,x,scale,alpha,run)
                err,_=evaluate(ema,val,scale,cfg['fixed_screen_guidance'],cfg['tune_draws'])
                row={'architecture':name,'lr':lr,'updates':cfg['refine_updates'],'guidance':cfg['fixed_screen_guidance'],
                     'validation_rmse_uv':float(err.mean())}
                save_json(result_path,row)
                print('REFINED_SCREEN',json.dumps(row),flush=True)
                if run:run.log({f'{name}/refined_screen/lr':lr,f'{name}/refined_screen/rmse_uv':row['validation_rmse_uv']})
                del ema
            refined.append(row)
        save_csv(out/f'{name}_refined_screen.csv',refined)
        winner=min(refined,key=lambda r:r['validation_rmse_uv'])
        if name in selections and selections[name]!=winner:raise ValueError('Saved choice does not match screening')
        selections[name]=winner;save_json(selection_path,selections)
        print('LR_SELECTED',json.dumps(winner),flush=True)
        (out/'PROGRESS.txt').write_text(f'Selected {name} LR {winner["lr"]:g}; continuing to {cfg["total_updates"]:,} updates.\n')
        train_to(name,winner['lr'],cfg['total_updates'],cfg,out,x,scale,alpha,run,validation=val)
    baseline=torch.load(BASELINE,map_location=args.device,weights_only=True)
    old=model_for('mlp',args.device).eval().requires_grad_(False);old.load_state_dict(baseline['state'])
    baseline_scale=float(baseline['scale'])
    final_path=out/'final_selection.json'
    if final_path.exists():
        final_choices=json.loads(final_path.read_text())
    else:
        final_choices={}
        for name in cfg['architectures']:
            row=selections[name];path=out/name/f'lr_{row["lr"]:.0e}'/'last.pt'
            saved=torch.load(path,map_location=args.device,weights_only=False)
            model=model_for(name,args.device).eval().requires_grad_(False);model.load_state_dict(saved['ema'])
            final_choices[name]=guide_search(model,val,scale,cfg['guidance_grid'],cfg['final_val_draws'],'final',name,out,run)
            del model
        base_val=make_observations(cfg['validation_count'],cfg['val_wave_seed'],cfg['val_noise_seed'],baseline_scale,args.device)
        final_choices['existing_mlp_2m']=guide_search(old,base_val,baseline_scale,cfg['guidance_grid'],cfg['final_val_draws'],'final','existing_mlp_2m',out,run)
        save_json(final_path,final_choices)
    (out/'SELECTION_FROZEN').write_text('Training and guidance choices frozen before test generation.\n')
    print('SELECTION_FROZEN',flush=True)
    test_rows=[];preds={}
    for name in ['existing_mlp_2m']+cfg['architectures']:
        if name=='existing_mlp_2m': model=old;current_scale=baseline_scale
        else:
            row=selections[name];saved=torch.load(out/name/f'lr_{row["lr"]:.0e}'/'last.pt',map_location=args.device,weights_only=False)
            model=model_for(name,args.device).eval().requires_grad_(False);model.load_state_dict(saved['ema']);current_scale=scale
        test=make_observations(cfg['test_count'],cfg['test_wave_seed'],cfg['test_noise_seed'],current_scale,args.device)
        err,pred=evaluate(model,test,current_scale,final_choices[name]['guidance'],cfg['test_draws'],seed=479)
        preds['truth']=test['truth'];preds[name]=pred
        for i,(s,e) in enumerate(zip(test['sigmas'],err)):
            test_rows.append({'model':name,'guidance':final_choices[name]['guidance'],'noise_uv':float(s),'record':i%cfg['test_count'],'rmse_uv':float(e)})
        save_csv(out/'test_per_record.csv',test_rows)
        print('TEST',name,float(err.mean()),flush=True)
        if run:run.log({f'{name}/test/mean_rmse_uv':float(err.mean())})
    np.savez_compressed(out/'test_predictions.npz',**preds)
    from mlp_unet_report import render
    report=render(out)
    if run:
        for image_path in sorted((out/'figures').glob('*.png')):
            run.log({f'figures/{image_path.stem}':wandb.Image(str(image_path))})
        run.save(str(report),base_path=str(out),policy='now')
        for f in ('protocol.json','training_selection.json','final_selection.json','test_per_record.csv','mlp_screen.csv','unet_screen.csv','mlp_refined_screen.csv','unet_refined_screen.csv'):
            run.save(str(out/f),base_path=str(out),policy='now')
        run.finish()
    (out/'STATUS.txt').write_text('COMPLETE\n')
    print('COMPLETE',report,flush=True)


if __name__=='__main__':main()
