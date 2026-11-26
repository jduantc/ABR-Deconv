"""Make a dated, static report of the active matched-training experiment."""
import base64
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'Leo/runs/phoom_matched_10m'
GUIDANCE = ROOT / 'Leo/results/guidance_sweep'
OUT = ROOT / 'Leo/results/training_progress_2026_11_20'


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = OUT / 'figures'
    figs.mkdir(exist_ok=True)
    train, val = read(RUN/'training.csv'), read(RUN/'validation.csv')
    interim, tests = read(RUN/'interim_1m_metrics.csv'), read(GUIDANCE/'test_per_record.csv')
    latest = int(train[-1]['update'])
    stamp = datetime.now(ZoneInfo('Europe/Zagreb')).strftime('%d %B %Y, %H:%M CET')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.axisbelow':True, 'axes.grid':True,
        'grid.color':'#e1e9ef', 'axes.titleweight':'bold', 'figure.facecolor':'white'})
    cards = []

    def save(fig, name, title, caption):
        fig.set_layout_engine('constrained')
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(figs / f'{name}.{ext}', dpi=200, facecolor='white')
        plt.close(fig)
        b64 = base64.b64encode((figs / f'{name}.png').read_bytes()).decode()
        cards.append(f'<section><h2>{title}</h2><img src="data:image/png;base64,{b64}" alt="{title}"><p class="caption">{caption}</p></section>')

    fig, ax = plt.subplots(figsize=(10.5,4))
    x = [int(r['update']) for r in train]
    y = [float(r['v_mse']) for r in train]
    ax.plot(x, y, color='#146f79', lw=1.1)
    ax.set(xscale='log', xlabel='Training updates (log scale)', ylabel='Training v-MSE', title='Denoising training loss over time')
    save(fig,'01_training_loss','1 · Is the model still learning to predict v?',
         'This is training loss, not ABR reconstruction RMSE. Points before 50,000 updates summarize 200 updates; subsequent points summarize 2,000 updates. Each batch and its diffusion noise are randomly sampled.')

    fig, ax = plt.subplots(figsize=(10.5,4.5))
    for rate, color in [(.001,'#8596a5'),(.005,'#146f79')]:
        rows = sorted((r for r in val if float(r['ema_rate']) == rate), key=lambda r:int(r['update']))
        ax.plot([int(r['update']) for r in rows],[float(r['rmse_uv']) for r in rows],'-o',color=color,label=f'EMA update rate {rate:g}')
    ax.set(xscale='log',xlabel='Checkpoint updates (log scale)',ylabel='Validation ABR RMSE (µV)',title='Reconstruction quality at saved checkpoints')
    ax.legend(frameon=False)
    save(fig,'02_validation_reconstruction','2 · Does longer training improve reconstruction?',
         'The same 24 validation ABRs at three noise levels, eight reconstructions averaged per EEG, and fixed noise-normalized guidance. Both EMA curves follow the same training run.')

    sigmas = [.05,.2,.5]
    def mean(rows, method, sigma):
        return np.mean([float(r['rmse_uv']) for r in rows if r['method']==method and float(r['noise_uv'])==sigma])
    a = [mean(interim,'ema_0.005',s) for s in sigmas]
    b = [mean(interim,'ema_0.005_refined',s) for s in sigmas]
    a2 = [mean(tests,'original_default',s) for s in sigmas]
    b2 = [mean(tests,'selected',s) for s in sigmas]
    def bars(first, second, labels, name, title, caption):
        fig, ax = plt.subplots(figsize=(10.5,4.5))
        xx = np.arange(3)
        for offset, values, label, color in [(-.16,first,labels[0],'#d99556'),(.16,second,labels[1],'#146f79')]:
            artist=ax.bar(xx+offset,values,.32,label=label,color=color)
            ax.bar_label(artist,labels=[f'{v:.5f}' for v in values],padding=3,fontsize=9)
        ax.set_xticks(xx,[str(s) for s in sigmas])
        ax.set_ylim(0,max(first+second)*1.25)
        ax.set(xlabel='Simulated EEG noise standard deviation (µV)',ylabel='Test ABR RMSE (µV)',title=title)
        ax.legend(frameon=False,ncol=2)
        save(fig,name,title,caption)
    bars(a,b,('Without correction','With correction'),'03_interim_test','3 · One-million-update interim test',
         '24 fresh nonzero synthetic ABRs at each noise level. Each estimate averages 32 reconstructions. This Phoom-style evaluation uses the known simulated noise level in guidance; the optional correction also uses it.')
    bars(a2,b2,('Guidance 0.1','Guidance 6.25'),'04_guidance_test','4 · Separate guidance sweep',
         'A different set of 32 test ABRs and a different reconstruction protocol. Guidance 6.25 was chosen on separate validation ABRs; each test estimate averages 32 reconstructions. The EEG residual is not divided by noise standard deviation in this experiment. Do not treat the difference between this figure and the preceding figure as an isolated model improvement.')

    vr={(int(r['update']),float(r['ema_rate'])):float(r['rmse_uv']) for r in val}
    tr={int(r['update']):float(r['v_mse']) for r in train}
    checkpoints=[4000,50000,100000,250000,500000,1000000,2000000,5000000]
    table=''.join(f'<tr><td>{n:,}</td><td>{tr[n]:.5f}</td><td>{vr[(n,.001)]:.6f}</td><td>{vr[(n,.005)]:.6f}</td></tr>' for n in checkpoints if n in tr and (n,.005) in vr)
    test_table=''.join(f'<tr><td>{s:g}</td><td>{a[i]:.6f}</td><td>{b[i]:.6f}</td><td>{a2[i]:.6f}</td><td>{b2[i]:.6f}</td></tr>' for i,s in enumerate(sigmas))
    one, two, five = vr[(1000000,.005)], vr[(2000000,.005)], vr[(5000000,.005)]
    reduction=100*(one-two)/one
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ABR training progress</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f6f6;color:#173044;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:48px 24px 80px}}h1{{font-size:clamp(2.4rem,5vw,3.7rem);line-height:1.08;letter-spacing:-.04em;margin:.25em 0}}h2{{line-height:1.2}}.eyebrow{{text-transform:uppercase;letter-spacing:.16em;color:#0e777c;font-size:.76rem;font-weight:800}}.lead{{max-width:78ch;font-size:1.08rem}}section,.hero{{background:white;border:1px solid #dce8e9;border-radius:14px;padding:27px;margin:20px 0;box-shadow:0 7px 24px #173d3d08}}.hero{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.stat strong{{display:block;color:#0e777c;font-size:1.9rem;line-height:1.15}}.stat small,.caption{{color:#526c78}}img{{width:100%;height:auto;display:block}}table{{width:100%;border-collapse:collapse;font-size:.91rem}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #e1eaec}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;background:#f2f7f7;padding:16px;border-radius:8px}}a{{color:#126c83}}@media(max-width:700px){{main{{padding:25px 14px}}.hero{{grid-template-columns:1fr}}section{{padding:18px}}}}
</style></head><body><main><header><div class="eyebrow">CS163 · training progress snapshot</div><h1>How is the ABR reconstruction model progressing?</h1><p class="lead">A report of the active long MLP training run and the separate guidance experiment. Snapshot: {stamp}. Figures and numbers do not update automatically.</p></header>
<div class="hero"><div class="stat"><strong>{latest/1e6:.3f} million</strong><small>latest logged training updates; 10 million planned</small></div><div class="stat"><strong>{five:.5f} µV</strong><small>validation reconstruction RMSE at the latest saved 5-million checkpoint, EMA 0.005</small></div><div class="stat"><strong>{two:.5f} µV</strong><small>best observed validation checkpoint: 2 million, EMA 0.005</small></div></div>
<section><h2>What is the main finding?</h2><p>Reconstruction validation RMSE improved from {one:.6f} µV at one million updates to {two:.6f} µV at two million, a {reduction:.1f}% reduction. At five million it was {five:.6f} µV: slightly higher than at two million under the same fixed evaluation. This suggests diminishing returns so far; the small difference is not proof that the two-million checkpoint will be better on new recordings.</p><p>The current training process continues beyond five million, but no reconstruction result exists yet for its latest updates. The next scheduled saved checkpoint and validation assessment is at ten million. The most recent resumable training state is at five million.</p><p><a href="https://forge.coreweave.com/wandb/abr-deconv/abr-deconv-phoom-matched-training/runs/b3ac809bd4f791a5">Follow the active training run in CoreWeave Forge</a></p></section>
<section><h2>What exactly is being trained?</h2><p>A 544,768-parameter residual MLP learns to predict <em>v</em> from a clean synthetic ABR after diffusion noise is added. It is trained on 20,000 filtered synthetic ABRs, with batches of 256, AdamW at learning rate 0.0002, a 500-update initial warmup, and no weight decay. The run continues the same optimizer trajectory started in the earlier 50,000-update experiment. Two exponential moving averages (EMA rates 0.001 and 0.005) follow the same weights; they are not independent training runs.</p><p>During reconstruction the weights are frozen. Starting from noise, the model repeatedly estimates a clean ABR and guidance adjusts the next waveform according to how well the estimate explains the EEG.</p></section>
<section><h2>Three different quantities</h2><p><strong>Training v-MSE:</strong> error between predicted v and the v target for artificially noised clean ABRs. It measures denoising training performance, in scaled model units.</p><p><strong>EEG residual RMS:</strong> difference between observed EEG and EEG predicted from the current estimated ABR. Its gradient guides reconstruction. The long run evaluation divides it by the known simulated noise standard deviation; this differs from your updated standalone reconstruction and the later guidance sweep.</p><p><strong>ABR reconstruction RMSE:</strong> difference between the final reconstructed ABR and the known clean synthetic ABR, in µV. This is the quantity we want to minimize. It requires ground truth, which generally is unavailable for real EEG.</p></section>
{''.join(cards)}
<section><h2>Checkpoint results in detail</h2><div class="scroll"><table><thead><tr><th>Updates</th><th>Training v-MSE</th><th>Validation RMSE: EMA 0.001 (µV)</th><th>Validation RMSE: EMA 0.005 (µV)</th></tr></thead><tbody>{table}</tbody></table></div><p>Validation RMSE is the mean of individual waveform RMSEs across three equally represented noise conditions. The 4,000-update checkpoint is part of the same trajectory. EMA 0.005 has the lowest recorded validation score at two million, but later EMA differences are small.</p></section>
<section><h2>Test results in detail</h2><div class="scroll"><table><thead><tr><th>Noise SD (µV)</th><th>1m interim, no correction (µV)</th><th>1m interim, correction (µV)</th><th>Guidance sweep, 0.1 (µV)</th><th>Guidance sweep, 6.25 (µV)</th></tr></thead><tbody>{test_table}</tbody></table></div><p>The first two result columns use 24 test ABRs and noise-normalized guidance. The last two use 32 other test ABRs and unnormalized guidance. Both use EMA 0.005 and 32 draws per EEG, but they are different experiments. The correction helps at medium and high noise levels and slightly hurts at the low noise level in this interim test.</p></section>
<section><h2>What we can conclude—and what remains unknown</h2><ul><li>The one-million-update model reconstructs the tested synthetic EEGs, and additional training to two million reduced validation RMSE under this protocol.</li><li>The EMA choices are close at long durations. The two-million versus five-million difference is small and comes from one training seed and a modest synthetic validation set.</li><li>The five-million checkpoint has a validation score, not the same independent interim test as the one-million checkpoint. The final test is planned at ten million and has not been completed.</li><li>Guidance 6.25 is promising for a different, unnormalized reconstruction protocol and the one-million checkpoint. It cannot be directly transferred into the noise-normalized evaluation.</li><li>Synthetic ABRs come from a restricted waveform family; EEG noise is white and its simulated level is known. Real EEG may behave differently. The interim and guidance test results have been inspected, so a fresh holdout is needed for a final comparison after further tuning.</li></ul><p>Future work can compare training-set sizes while holding the training budget and evaluation protocol fixed, and include realistic waveforms and correlated noise.</p></section>
<section><h2>Reproducibility and links</h2><p>This is a static snapshot derived from the run's live CSV files. Figures are saved separately as PNG, PDF and SVG. The full earlier <a href="../../runs/phoom_matched_10m/INTERIM_1M.html">one-million-update interim report</a> and <a href="../guidance_sweep/GUIDANCE_REPORT.html">guidance sweep report</a> contain further details.</p><pre>Validation: 24 nonzero ABRs × 3 noise levels; 8 draws per EEG
1m interim test: 24 nonzero ABRs × 3 noise levels; 32 draws per EEG
Guidance test: 32 nonzero ABRs × 3 noise levels; 32 draws per EEG
Next scheduled checkpoint: 10 million updates
Synthetic EEG: 0.5 seconds; regular pulses; white noise
Source: Leo/experiments/phoom_matched_training.py</pre></section></main></body></html>'''
    report=OUT/'TRAINING_PROGRESS_REPORT.html'
    report.write_text(doc,encoding='utf-8')
    (OUT/'snapshot.json').write_text(json.dumps({'as_of':stamp,'latest_logged_update':latest,'latest_validation_checkpoint':5000000,'status':'RUNNING','validation':val},indent=2))
    print(report)


if __name__ == '__main__':
    main()
