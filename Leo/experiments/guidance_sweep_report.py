"""Create the self-contained report and publication-quality sweep figures."""
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def render(out):
    out = Path(out)
    config = json.loads((out/'protocol.json').read_text())
    choice = json.loads((out/'selection.json').read_text())
    elapsed = json.loads((out/'runtime.json').read_text())['seconds']
    val = sorted(read(out/'validation_grid.csv'), key=lambda r: float(r['guidance']))
    test = read(out/'test_per_record.csv')
    pred = np.load(out/'test_predictions.npz')
    sigmas = config['noise_uv']
    selected = choice['guidance']
    colors = ['#127b80', '#3155a4', '#bd6839']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.axisbelow':True, 'axes.grid':True, 'grid.color':'#e5ecf0',
        'text.color':'#193044', 'axes.labelcolor':'#193044', 'figure.facecolor':'white',
        'axes.titleweight':'bold', 'savefig.facecolor':'white'})
    figdir = out/'figures'
    figdir.mkdir(exist_ok=True)
    images, sections = [], []
    def save(fig, name, heading, caption):
        fig.set_layout_engine('constrained')
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(figdir/f'{name}.{ext}', dpi=200)
        plt.close(fig)
        path = figdir/f'{name}.png'
        images.append(path)
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        sections.append(f'<section><h2>{heading}</h2><img src="data:image/png;base64,{data}" alt="{heading}"><p class="caption">{caption}</p></section>')
    x = [float(r['guidance']) for r in val]
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    for sigma, color in zip(sigmas, colors):
        ax.plot(x, [float(r[f'rmse_{sigma}_uv']) for r in val], 'o-', color=color, markersize=4,
                label=f'EEG noise SD {sigma:g} µV')
    ax.axvline(selected, color='#bd4d4a', ls='--', lw=1.5, label=f'Selected: {selected:g}')
    ax.set(xscale='symlog', xlabel='Guidance strength (symmetric log scale)', ylabel='Validation ABR RMSE (µV)',
           title='Guidance sweep | one shared strength across all noise levels')
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, '01_validation_grid', '1 · Where does guidance help?',
         'Each point uses the same 32 reference waveforms at each of three noise levels and an average of eight reconstructions per EEG. The dashed line minimizes the equally weighted mean RMSE across all three noise levels. Zero is included on the symmetric log axis.')

    def vals(method, sigma):
        return np.array([float(r['rmse_uv']) for r in test if r['method']==method and float(r['noise_uv'])==sigma])
    methods = [('unguided','No guidance','#9baab4'), ('original_default','Original 0.1','#d59656'), ('selected',f'Selected {selected:g}','#127b80')]
    fig, ax = plt.subplots(figsize=(10.5,4.8))
    xx = np.arange(3); width = .23
    for i,(method,label,color) in enumerate(methods):
        means = [vals(method,s).mean() for s in sigmas]
        bars = ax.bar(xx+(i-1)*width, means, width, label=label, color=color)
        ax.bar_label(bars, labels=[f'{z:.4f}' for z in means], padding=3, fontsize=8, rotation=0)
    ax.set_xticks(xx, [f'{s:g}' for s in sigmas])
    ax.set(xlabel='Simulated EEG noise SD (µV)', ylabel='Test ABR RMSE (µV)',
           title='Fresh test recordings | lower is better')
    ax.set_ylim(0, ax.get_ylim()[1]*1.15)
    ax.legend(frameon=False, ncol=3)
    save(fig, '02_test_comparison', '2 · Does the selected value generalize?',
         'The guidance value was frozen before these 32 new ABRs were evaluated. Bars are the mean of per-recording RMSEs after averaging 32 reconstructions per EEG. All methods use the same recordings and random draws.')

    fig, axes = plt.subplots(1,3,figsize=(11.5,4),sharey=True)
    for ax,sigma,color in zip(axes,sigmas,colors):
        before, after = vals('original_default',sigma), vals('selected',sigma)
        diff = (before-after)*1000
        ax.axhline(0,color='#536b7b',lw=1)
        ax.scatter(np.arange(len(diff)), np.sort(diff), s=18, color=color, alpha=.8)
        ax.set(title=f'Noise SD {sigma:g} µV', xlabel='Test recordings, sorted')
    axes[0].set_ylabel('RMSE reduction vs 0.1 (nV)')
    fig.suptitle('Per-recording improvement | points above zero favor selected guidance',fontweight='bold')
    save(fig, '03_paired_changes', '3 · Is the improvement consistent?',
         'Each point compares the same test EEG at the default and selected guidance values. A point below zero means the selected setting performed worse on that recording.')

    sigma = .2; idx0 = sigmas.index(sigma)*config['test_count']
    delta = vals('original_default',sigma)-vals('selected',sigma)
    idx = idx0 + int(np.argsort(delta)[len(delta)//2])
    t = (np.arange(256)-50)/10
    fig, ax = plt.subplots(figsize=(10.5,4.4))
    for key,label,color,style in [('truth','Clean reference','#172e42','-'),('original_default','Default 0.1','#d59656','--'),('selected',f'Selected {selected:g}','#127b80','-')]:
        ax.plot(t,pred[key][idx], label=label, color=color, lw=2 if key=='truth' else 1.6, ls=style)
    ax.axvline(0, color='#9baab4', lw=1, ls=':')
    ax.set(xlabel='Time relative to stimulus (ms)', ylabel='ABR amplitude (µV)', title='Example reconstruction | 0.2 µV simulated EEG noise SD')
    ax.legend(frameon=False, ncol=3)
    save(fig,'04_example_waveform','4 · What does a reconstruction look like?',
         'Example chosen by the median per-recording improvement at the middle noise level; it is illustrative, not the basis of the selection. The dark curve is the known synthetic reference.')

    sensitivity_path = out/'checkpoint_sensitivity.csv'
    if sensitivity_path.exists():
        sensitivity = read(sensitivity_path)
        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        for update,color in [(4000,'#bd6839'),(500000,'#3155a4')]:
            subset=sorted((r for r in sensitivity if int(r['checkpoint_updates'])==update),key=lambda r:float(r['guidance']))
            ax.plot([float(r['guidance']) for r in subset], [float(r['mean_rmse_uv']) for r in subset], 'o-', color=color, label=f'{update:,} updates')
        subset=[r for r in val if float(r['guidance']) in (3.,4.,5.5,6.25,7.,8.5,10.)]
        ax.plot([float(r['guidance']) for r in subset], [float(r['mean_rmse_uv']) for r in subset], 'o-', color='#127b80',label='1,000,000 updates')
        ax.axvline(selected,color='#738b93',ls='--',lw=1.5)
        ax.set(xlabel='Guidance strength',ylabel='Mean validation ABR RMSE (µV)',title='Checkpoint sensitivity | the same validation recordings')
        ax.legend(frameon=False,ncol=3)
        save(fig,'05_checkpoint_sensitivity','5 · Does the guidance range depend on training duration?',
             'Exploratory check on the same 32 validation ABRs, each at all three noise levels, with eight-sample averages. These checkpoints follow the same training trajectory and EMA rate. The selected value was not changed after the fresh-test evaluation.')

    sample_path = out/'test_samples_per_record.csv'
    if sample_path.exists():
        sample_data = read(sample_path)
        fig, ax = plt.subplots(figsize=(10.5,4.4))
        for method,label,color in [('original_default','Default 0.1','#d59656'),('selected',f'Selected {selected:g}','#127b80')]:
            means=[]
            for count in (1,4,8,16,32):
                entries=[float(r['rmse_uv']) for r in sample_data if r['method']==method and int(r['samples'])==count]
                means.append(float(np.mean(entries)))
            ax.plot((1,4,8,16,32),means,'o-',color=color,label=label)
        ax.set(xscale='log',xticks=(1,4,8,16,32),xticklabels=('1','4','8','16','32'),
               xlabel='Number of reconstructions averaged',ylabel='Mean test ABR RMSE (µV)',
               title='Effect of averaging | fixed guidance, fresh test set')
        ax.legend(frameon=False)
        save(fig,'06_sample_averaging','6 · What if you use one reconstruction?',
             'Exploratory evaluation of the already selected guidance on the test set; no choice was changed. Each point uses the first N of the same 32 draws per EEG and averages error equally across the three noise levels. One draw corresponds to the ordinary single-call reconstruction workflow.')

    table = []
    for s in sigmas:
        a,b=vals('original_default',s),vals('selected',s)
        rng=np.random.default_rng(841)
        d=a-b
        boot=d[rng.integers(0,len(d),size=(3000,len(d)))].mean(1)
        low,high=np.quantile(boot,[.025,.975])
        table.append((s,float(a.mean()),float(b.mean()),float(d.mean()),float(low),float(high),int((d>0).sum())))
    body=''.join(f'<tr><td>{s:g}</td><td>{a:.6f}</td><td>{b:.6f}</td><td>{d:+.6f}</td><td>[{lo:+.6f}, {hi:+.6f}]</td><td>{wins}/32</td></tr>' for s,a,b,d,lo,hi,wins in table)
    overall_default=np.mean([vals('original_default',s).mean() for s in sigmas])
    overall_selected=np.mean([vals('selected',s).mean() for s in sigmas])
    relative=100*(overall_default-overall_selected)/overall_default
    grid_rows=''.join(f'<tr><td>{float(r["guidance"]):g}</td><td>{r["stage"]}</td><td>{float(r["mean_rmse_uv"]):.6f}</td><td>{float(r["rmse_0.05_uv"]):.6f}</td><td>{float(r["rmse_0.2_uv"]):.6f}</td><td>{float(r["rmse_0.5_uv"]):.6f}</td></tr>' for r in val)
    boundary = '<p class="note">The selected value lies at an edge of the tested range, so the optimum is not bracketed.</p>' if choice['boundary_warning'] else ''
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Guidance strength · ABR reconstruction</title><style>
    :root{{--ink:#193044;--teal:#127b80;--line:#dbe7ea}}*{{box-sizing:border-box}}body{{margin:0;background:#f2f7f7;color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:48px 24px 80px}}header{{padding:12px 0 20px}}h1{{font-size:clamp(2.3rem,5vw,3.7rem);letter-spacing:-.04em;line-height:1.06;margin:.25em 0}}h2{{line-height:1.2;margin-top:0}}.eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:.76rem;font-weight:750;color:var(--teal)}}.lead{{font-size:1.1rem;max-width:75ch}}section,.hero{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:28px;margin:20px 0;box-shadow:0 7px 24px #173d3d08}}.hero{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.stat strong{{display:block;font-size:2rem;line-height:1.2;color:var(--teal)}}.stat small,.caption{{color:#536b7b}}img{{width:100%;height:auto;display:block}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{color:#536b7b}}.scroll{{overflow:auto}}code,pre{{font-family:ui-monospace,monospace}}pre{{white-space:pre-wrap;background:#edf5f4;padding:18px;border-radius:8px;overflow-wrap:anywhere}}.note{{border-left:4px solid #d59656;padding:10px 16px;background:#fff8ed}}a{{color:#176a83}}@media(max-width:700px){{main{{padding:25px 14px}}.hero{{grid-template-columns:1fr}}section{{padding:18px}}}}
    </style></head><body><main><header><div class="eyebrow">CS163 · synthetic reconstruction study</div><h1>How much should EEG guide the reconstruction?</h1><p class="lead">A two-stage grid search using your EEG-guided diffusion update, a frozen one-million-update MLP, and fresh validation and test recordings.</p></header>
    <div class="hero"><div class="stat"><strong>{selected:g}</strong><small>one shared guidance value selected on validation data</small></div><div class="stat"><strong>{overall_selected:.5f} µV</strong><small>mean fresh-test reconstruction RMSE across noise levels</small></div><div class="stat"><strong>{relative:+.1f}%</strong><small>test RMSE reduction relative to the original guidance of 0.1</small></div></div>
    <section><h2>What was trained?</h2><p>No new model was trained for this search. We reused the MLP trained for 1,000,000 optimizer updates on 20,000 synthetic clean ABRs (one training seed), with batch size 256, AdamW learning rate 0.0002, a 500-update warmup, and zero weight decay. The checkpoint uses EMA rate 0.005. Its denoising training objective is mean squared error in predicted <em>v</em>; it is not an ABR reconstruction loss.</p><p>For this experiment the checkpoint is frozen. We vary only guidance during the 200-step reconstruction. The model works in training-standard-deviation-scaled waveform units; EEG values are converted to those same units. This waveform scaling is separate from EEG noise normalization.</p></section>
    <section><h2>How the experiment works</h2><p>For each clean simulated ABR, we build an EEG by convolving it with known stimulus pulses and adding white noise. Starting from noise, the frozen model estimates an ABR. The pulse matrix predicts the EEG from that estimate; the RMS difference from observed EEG gives a gradient, and guidance determines the size of the update. The EEG residual is divided by √N to make it an RMS, <strong>not divided by noise standard deviation</strong>.</p><pre>EEG residual RMS = √mean((observed EEG − predicted EEG)²)
next waveform = diffusion update − guidance × residual gradient
ABR RMSE = √mean((reconstructed ABR − clean reference ABR)²)</pre><p>All reported RMSE values are in µV. Each clean waveform has a separate simulated EEG at noise SD 0.05, 0.2 and 0.5 µV. Noise SD is used to create and label simulations, not as an input to the reconstruction loss or to choose among different guidance strengths per recording.</p></section>
    {''.join(sections)}
    <section><h2>Exact fresh-test results</h2><div class="scroll"><table><thead><tr><th>Noise SD (µV)</th><th>Default 0.1 RMSE (µV)</th><th>Selected RMSE (µV)</th><th>Mean reduction (µV)</th><th>95% paired bootstrap interval (µV)</th><th>Selected better</th></tr></thead><tbody>{body}</tbody></table></div><p>Positive reduction favors the selected value. Bootstrap intervals resample the 32 paired recordings within each noise level; they capture test-recording variability, not uncertainty across training seeds.</p>{boundary}</section>
    <section><h2>How to use this result with the checkpoint</h2><p>The chosen value applies to this particular checkpoint and its waveform scaling. The checkpoint stores weights under <code>state</code> and a scaling factor under <code>scale</code>; the standalone <code>load_model</code> helper expects a different checkpoint format. Run the following from the <code>Leo</code> directory, with <code>eeg_uv</code> and the equally long pulse train already loaded. Averaging 32 samples matches the test protocol.</p><pre>import numpy as np
import torch
from model import Denoiser
from dps.reconstruct import reconstruct

checkpoint = torch.load(
    "runs/phoom_matched_10m/step_1000000_ema_0.005.pt",
    map_location="cpu", weights_only=True,
)
model = Denoiser().eval().requires_grad_(False)
model.load_state_dict(checkpoint["state"])
scale_uv = checkpoint["scale"]
draws = [reconstruct(model, eeg_uv / scale_uv, pulses,
                     guidance={selected:g}, seed=seed)
         for seed in range(32)]
reconstructed_abr_uv = np.mean(draws, axis=0) * scale_uv</pre><p>The EEG is divided by the model's waveform scale, not by an estimated EEG noise standard deviation. The example uses different random seeds to create 32 draws; exact draws can differ from the batched GPU experiment. Guidance may need retuning for a checkpoint trained with different scaling or for a single reconstruction.</p></section>
    <section><h2>Validation grid and selection</h2><p>The coarse grid was {html.escape(str(config['coarse_grid']))}. After finding its best value, the neighboring interval was refined using a nine-point linear grid, excluding already tested points. One global guidance value was selected by minimizing the equally weighted mean of the three noise-level RMSEs. The 32 validation waveforms and 32 test waveforms have distinct generation seeds. Each setting sees the same EEGs and sampling seed within a split.</p><div class="scroll"><table><thead><tr><th>Guidance</th><th>Stage</th><th>Mean RMSE (µV)</th><th>0.05 µV noise</th><th>0.2 µV noise</th><th>0.5 µV noise</th></tr></thead><tbody>{grid_rows}</tbody></table></div></section>
    <section><h2>Interpretation and limitations</h2><p>These results establish a comparison for this checkpoint and this simulator. They do not prove an optimum for other checkpoints, pulse patterns, recording lengths, real EEG, or a different number of averaged reconstructions. Selection used eight samples per validation EEG; primary test results use 32, so the optimum may shift with sample count. Only one model training seed was evaluated. The generator uses filtered analytic waveforms and simulated white noise; real EEG has different noise and unknown ground truth. Test results cannot prove clinical or real-recording accuracy.</p><p>The accelerated batched sampler uses the same residual and diffusion update as the standalone code. A complete 200-step CPU trajectory was compared numerically against that function (maximum absolute difference {config['sampler_max_abs_difference_cpu']:.2g} model units). The main search and primary test used {elapsed/60:.1f} minutes; exploratory checks took additional time. No retraining was required. <a href="https://forge.coreweave.com/wandb/abr-deconv/abr-deconv-guidance-sweep">Open the CoreWeave Forge project</a>.</p><p>For follow-up, verify the broad guidance range with other training seeds, longer and varied pulse sequences, and realistic correlated EEG noise. Keep a new final test set separate if using these results to tune further.</p></section>
    <section><h2>Reproducibility</h2><p>The adjacent CSV files contain the full validation grid and per-recording test errors. Figures are available as PNG, PDF and SVG. The checkpoint and its SHA-256 are recorded below.</p><pre>{html.escape(json.dumps(config,indent=2))}</pre></section></main></body></html>'''
    report=out/'GUIDANCE_REPORT.html'
    report.write_text(doc,encoding='utf-8')
    return report, images
