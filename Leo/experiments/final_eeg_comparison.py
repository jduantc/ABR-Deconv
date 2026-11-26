"""Evaluate completed models on realistic hybrid and actual EEG; produce one folder."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('WANDB_BASE_URL','https://api.forge.coreweave.com')
import csv,json,sys,base64,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Leo/experiments'))
import realistic_eeg as data
from realistic_eeg_study import stats,estimate
from mlp_unet_comparison import model_for
RUN=ROOT/'Leo/runs/mlp_unet_2m';OUT=ROOT/'Leo/results/mlp_unet_eeg_comparison'
GRID=(0.,.1,.3,1.,3.,6.25,12.)
CONDITIONS=('real_quiet_half','real_quiet_native','real_quiet_double')
LABELS=('0.5×','1×','2×')


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def save(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_model(name,sel):
    lr=sel[name]['lr'];path=RUN/name/f'lr_{lr:.0e}'/'last.pt'
    c=torch.load(path,map_location='cuda',weights_only=False)
    if c['step']!=2000000:raise ValueError(f'{name} checkpoint not finished')
    m=model_for(name,'cuda').eval().requires_grad_(False);m.load_state_dict(c['ema'])
    return m,float(c['scale']),{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    torch.set_num_threads(1)
    if (RUN/'STATUS.txt').read_text().strip()!='COMPLETE':raise RuntimeError('Training and its frozen test must finish first')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Output exists; avoid overwriting the test results')
    selected=json.loads((RUN/'training_selection.json').read_text());old=read(RUN/'test_per_record.csv')
    models={};scales={};sources={}
    for n in ('mlp','unet'):models[n],scales[n],sources[n]=load_model(n,selected)
    protocol={'checkpoint_sources':sources,'validation_subject':'sub-exp302','test_subject':'sub-exp303',
      'validation_count':8,'test_count':10,'validation_wave_seed':98531,'test_wave_seed':98532,
      'guidance_grid':GRID,'validation_draws':4,'test_draws':16,'validation_sampling_seed':881,
      'test_sampling_seed':991,'conditions':CONDITIONS,'note':'New data is known synthetic ABR + real pulse times + pre-playback EEG; 9 nonzero and 1 null target. Real speech EEG has no true ABR; its exploratory reconstruction has no RMSE.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'STATUS.txt').write_text('RUNNING\n')
    import wandb
    wandb_run=wandb.init(entity='abr-deconv',project='abr-deconv-eeg-comparison',name='MLP and U-Net · old and realistic EEG',config=protocol,settings=wandb.Settings(disable_code=True))
    (OUT/'forge_url.txt').write_text(wandb_run.url)
    val,vmeta,_=data.build('sub-exp302',8,98531,825001)
    (OUT/'validation_sources.json').write_text(json.dumps(vmeta,indent=2))
    choices={};valrows=[]
    for name in ('mlp','unet'):
        choices[name]={}
        for c in CONDITIONS:
            st=stats(val[c],scales[name],'cuda');scores=[]
            for g in GRID:
                p=estimate(models[name],st,scales[name],g,4,881)
                err=np.sqrt(np.mean((p-val[c]['truth'])**2,axis=1))
                row={'model':name,'condition':c,'guidance':g,'rmse_uv':float(err.mean())}
                valrows.append(row);scores.append(row)
                print('VALIDATION',json.dumps(row),flush=True)
                wandb_run.log({'validation/guidance':g,'validation/rmse_uv':row['rmse_uv'],'validation/model':name,'validation/condition':c})
            choices[name][c]=min(scores,key=lambda r:r['rmse_uv'])
            save(OUT/'validation_grid.csv',valrows)
    (OUT/'selection.json').write_text(json.dumps(choices,indent=2))
    (OUT/'SELECTION_FROZEN').write_text('Choices frozen before held-out test generation.\n')
    test,tmeta,extras=data.build('sub-exp303',10,98532,825002)
    (OUT/'test_sources.json').write_text(json.dumps(tmeta,indent=2))
    rows=[];predictions={'truth':test['real_quiet_native']['truth'],'real_eeg':extras['real_eeg']}
    for name in ('mlp','unet'):
        for c in CONDITIONS:
            obs=test[c];g=choices[name][c]['guidance'];st=stats(obs,scales[name],'cuda')
            for mode,strength in [('selected',g),('fixed_6_25',6.25)]:
                p=estimate(models[name],st,scales[name],strength,16,991)
                err=np.sqrt(np.mean((p-obs['truth'])**2,axis=1))
                for i,e in enumerate(err):rows.append({'model':name,'condition':c,'method':mode,'guidance':strength,'record':i,'null_control':i==9,'rmse_uv':float(e),'noise_rms_uv':float(data.rms(obs['noise'][i]))})
                predictions[f'{name}__{c}__{mode}']=p
                print('TEST',name,c,mode,float(err.mean()),flush=True)
                wandb_run.log({f'test/{name}/{c}/{mode}_rmse_uv':float(err.mean())})
        # Exploratory application to real speech EEG, using the 4 kHz-left-only model.
        # It cannot explain the other nine responses; the true ABR is unknown.
        real={'matrix':test['real_quiet_native']['matrix'],'eeg':extras['real_eeg'][None,:]}
        g=choices[name]['real_quiet_native']['guidance']
        predictions[f'{name}__real_eeg']=estimate(models[name],stats(real,scales[name],'cuda'),scales[name],g,16,991)[0]
    save(OUT/'new_test_per_record.csv',rows);np.savez_compressed(OUT/'predictions.npz',**predictions)
    (OUT/'old_test_per_record.csv').write_text((RUN/'test_per_record.csv').read_text())
    render(OUT,old,rows,choices,predictions,protocol)
    for p in sorted((OUT/'figures').glob('*.png')):wandb_run.log({f'figures/{p.stem}':wandb.Image(str(p))})
    for p in OUT.glob('*'):
        if p.suffix in ('.html','.json','.csv'):wandb_run.save(str(p),base_path=str(OUT),policy='now')
    wandb_run.finish();(OUT/'STATUS.txt').write_text('COMPLETE\n');print('COMPLETE',OUT/'REPORT.html',flush=True)


def render(out,old,new,choices,pred,protocol):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'axes.grid':True,'grid.color':'#e9edf4','text.color':'#192b45'})
    palette={'mlp':'#5264db','unet':'#cc774c'};figures=[]
    def savefig(fig,n,title,caption):
        fig.set_layout_engine('constrained')
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'{n}.{ext}',dpi=200,facecolor='white')
        plt.close(fig);img=base64.b64encode((figdir/f'{n}.png').read_bytes()).decode()
        figures.append(f'<section><h2>{title}</h2><img src="data:image/png;base64,{img}" alt="{title}"><p>{caption}</p></section>')
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(3);w=.32
    for i,n in enumerate(('mlp','unet')):
        v=[np.mean([float(r['rmse_uv']) for r in old if r['model']==n and float(r['noise_uv'])==s]) for s in (.05,.2,.5)]
        bars=ax.bar(x+(i-.5)*w,v,w,label=n.upper() if n=='mlp' else 'U-Net',color=palette[n]);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
    ax.set_xticks(x,['0.05','0.20','0.50']);ax.set(xlabel='White-noise SD (µV)',ylabel='ABR RMSE (µV)',title='Original simulated EEG');ax.legend(frameon=False);ax.set_ylim(0,ax.get_ylim()[1]*1.2)
    savefig(fig,'01_old_data_rmse','Original simulated EEG','Fresh old-simulator test set: 24 nonzero synthetic ABRs at each noise level; 32 reconstructions averaged per EEG. Guidance was selected per model on the original validation set.')
    fig,ax=plt.subplots(figsize=(11,4.8))
    for i,n in enumerate(('mlp','unet')):
        v=[np.mean([float(r['rmse_uv']) for r in new if r['model']==n and r['condition']==c and r['method']=='selected']) for c in CONDITIONS]
        bars=ax.bar(x+(i-.5)*w,v,w,label=n.upper() if n=='mlp' else 'U-Net',color=palette[n]);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
    ax.set_xticks(x,LABELS);ax.set(xlabel='Recorded quiet-background multiplier',ylabel='ABR RMSE (µV)',title='Real pulse timings + recorded background');ax.legend(frameon=False);ax.set_ylim(0,ax.get_ylim()[1]*1.2)
    savefig(fig,'02_new_data_rmse','New simulated EEG with recorded background','Fresh held-out subject: 9 nonzero synthetic ABRs and one null response per noise level; 16 reconstructions averaged per EEG. Guidance is validation-selected separately per model and condition. Zero guidance means the reconstruction ignores EEG.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True);tm=(np.arange(256)-50)/10
    for ax,n in zip(axes,('mlp','unet')):
        ax.plot(tm,pred['truth'][0],label='Clean ABR',color='#182a43',lw=2)
        ax.plot(tm,pred[f'{n}__real_quiet_native__selected'][0],label='Reconstructed ABR',color=palette[n],lw=1.8)
        ax.set(title=n.upper() if n=='mlp' else 'U-Net',xlabel='Time relative to stimulus (ms)')
    axes[0].set_ylabel('ABR (µV)');axes[1].legend(frameon=False)
    savefig(fig,'03_hybrid_example','Example: known ABR and reconstructions','Same prespecified first test ABR for both models, native quiet EEG. This is an example, not an average. The test ABR is known because it was simulated.')
    fig,ax=plt.subplots(figsize=(11,4))
    for n in ('mlp','unet'):ax.plot(tm,pred[f'{n}__real_eeg'],label=n.upper() if n=='mlp' else 'U-Net',color=palette[n],lw=1.8)
    ax.set(xlabel='Time relative to stimulus (ms)',ylabel='Estimated ABR (µV)',title='Exploratory estimates from actual speech EEG');ax.legend(frameon=False)
    savefig(fig,'04_real_eeg_estimates','Actual speech EEG: exploratory output','No clean reference exists, so RMSE cannot be computed. The input contains multiple overlapping responses but this exploratory reconstruction models only the 4 kHz left stimulus train. If selected guidance is zero, that output ignores the EEG.')
    zero=', '.join(f'{n} {c}' for n in choices for c in choices[n] if choices[n][c]['guidance']==0) or 'none'
    table=''.join(f'<tr><td>{n}</td><td>{c}</td><td>{choices[n][c]["guidance"]:g}</td><td>{np.mean([float(r["rmse_uv"]) for r in new if r["model"]==n and r["condition"]==c and r["method"]=="selected"]):.6f}</td></tr>' for n in ('mlp','unet') for c in CONDITIONS)
    h=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLP vs U-Net reconstruction</title><style>body{{max-width:1100px;margin:auto;padding:42px 24px;background:#f3f7fb;color:#1b2b43;font:16px/1.65 system-ui}}h1{{font-size:clamp(2.4rem,5vw,3.5rem);line-height:1.08}}section{{background:white;border:1px solid #e0e7ef;border-radius:14px;padding:25px;margin:20px 0}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #e5eaf1}}.note{{background:#fff6e8;padding:15px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>MLP versus U-Net: EEG reconstruction</h1><section><p>Both models were trained to two million updates and evaluated on two EEG simulation settings. Lower ABR RMSE is better. The realistic set combines known synthetic ABRs with actual speech pulse timings and quiet EEG from a held-out subject. This is a simulation containing real recorded background, with known ABR truth.</p><p class="note">Zero-guidance selections: {zero}. Such reconstructions ignore the EEG. Actual speech EEG has no known clean ABR, so a true real-EEG reconstruction RMSE cannot be calculated here.</p><p><a href="{(out/'forge_url.txt').read_text().strip()}">CoreWeave Forge run</a></p></section>{''.join(figures)}<section><h2>New-data results</h2><table><tr><th>Model</th><th>Condition</th><th>Selected guidance</th><th>Mean RMSE µV</th></tr>{table}</table><p>Guidance was chosen on a separate subject before evaluating the test subject. One seed per model, one validation subject and one test subject limit generalization. The 10 test cases include one null control. The original and new tests differ in recordings, number of draws, and noise; compare models within each graph rather than interpreting the difference between graphs as a single-factor experiment.</p></section><section><h2>Method and limitations</h2><p>The new data uses 500 ms, one 4 kHz-left pulse train, synthetic ABRs, and recorded quiet EEG before first annotated playback, using the original 100× gain correction and 30–2000 Hz/notch preprocessing. Quiet is event-defined, not independently verified. Real speech EEG includes other band/ear responses that the single-regressor exploratory run does not model. Sample size is small and no clinical conclusion follows.</p><p>Training, selection, per-recording CSV files and PNG/PDF/SVG figures are stored beside this report.</p><pre>{json.dumps(protocol,indent=2)}</pre></section></html>'''
    (out/'REPORT.html').write_text(h,encoding='utf-8')


if __name__=='__main__':main()
