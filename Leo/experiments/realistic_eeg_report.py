"""Minimal scientific figures and a self-contained realism-study report."""
import base64
import csv
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def render(out):
    out=Path(out)
    cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'test_sources.json').read_text())
    choices=json.loads((out/'selection.json').read_text());runtime=json.loads((out/'runtime.json').read_text())
    metrics=read(out/'test_metrics.csv');amp=read(out/'signal_amplitudes.csv')
    example=np.load(out/'examples.npz');preds=np.load(out/'predictions.npz')
    from realistic_eeg import LABELS,CONDITIONS
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#1c2a45',
        'axes.labelcolor':'#596b80','axes.spines.top':False,'axes.spines.right':False,
        'axes.spines.left':False,'axes.spines.bottom':False,'axes.grid':True,'axes.axisbelow':True,
        'grid.color':'#e9edf4','xtick.color':'#596b80','ytick.color':'#596b80'})
    blocks=[]
    def save(fig,name,title,caption):
        fig.set_layout_engine('constrained')
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'{name}.{ext}',dpi=200,facecolor='white')
        plt.close(fig)
        image=base64.b64encode((figdir/f'{name}.png').read_bytes()).decode()
        blocks.append(f'<section><h2>{title}</h2><img src="data:image/png;base64,{image}" alt="{title}"><p>{caption}</p></section>')
    t=np.arange(5000)/10
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,sharey=True)
    series=[('legacy_white','Original simulator','#8796b4'),('real_quiet_native','Real pulses + quiet EEG','#5264db'),('real_eeg','Real speech EEG','#ca774c')]
    for ax,(key,label,color) in zip(axes,series):
        ax.plot(t,example[key],color=color,lw=.8)
        ax.set(title=label,ylabel='EEG (µV)',xlim=(0,500));ax.tick_params(length=0)
    axes[-1].set_xlabel('Time (ms)')
    save(fig,'01_new_comparison','1 · The new comparison',
         'Every trace uses the same physical amplitude and time axes. No RMS normalization is applied. The original-style simulation uses regular pulses and 0.2 µV white noise; the new example uses real stimulus timings plus unscaled quiet EEG. The bottom trace is the recorded speech EEG used for visual reference, whose clean ABR is unknown. The original-style trace has mean removal for consistency.')
    fig,axes=plt.subplots(2,1,figsize=(12,4.5),sharex=True)
    for ax,key,label,color in zip(axes,['regular_pulses','real_pulses'],['Regular pulses','Real speech pulses'],['#8796b4','#5264db']):
        times=np.flatnonzero(example[key])/10
        ax.vlines(times,0,1,color=color,lw=1.4);ax.set(yticks=[],title=label,xlim=(0,500));ax.grid(False)
    axes[-1].set_xlabel('Time (ms)')
    save(fig,'02_real_pulses','2 · What changed in pulse timing?',
         f'Regular pulses have 45 events, exactly 10 ms apart. The selected real 4 kHz left pulse train contains {meta["real_pulse_count"]} events in this 500 ms window. Stimulus timings are mapped from 48 kHz to the EEG clock, using the measured clock correction and corrected trial onset. Source: first Wizard stimulus file, window 1–1.5 s into the trial.')
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,sharey=True)
    for ax,key,label in zip(axes,['real_quiet_half','real_quiet_native','real_quiet_double'],['Quiet EEG ×0.5','Quiet EEG ×1','Quiet EEG ×2']):
        ax.plot(t,example[key],color='#5264db',lw=.8);ax.set(title=label,ylabel='EEG (µV)',xlim=(0,500))
    axes[-1].set_xlabel('Time (ms)')
    save(fig,'03_noise_strength','3 · Noise strength in physical units',
         'These examples reuse the same recorded quiet-EEG segment and clean response, multiplying the background by 0.5, 1 or 2. The native 1× case retains its recorded amplitude. A separate varying-strength condition modulates the background by a smooth envelope between 0.5 and 1.5 before filtering.')
    def rows(c,method):return [r for r in metrics if r['condition']==c and r['method']==method]
    short=['Original','Regular + white','Real + white','Quiet ×0.5','Quiet ×1','Quiet ×2','Quiet varying']
    fig,ax=plt.subplots(figsize=(12,4.7));x=np.arange(len(CONDITIONS));w=.35
    for offset,method,label,color in [(-w/2,'fixed_6_25','Fixed guidance 6.25','#b1bdd7'),(w/2,'validation_selected','Validation-selected guidance','#5264db')]:
        values=[np.mean([float(r['rmse_uv']) for r in rows(c,method)]) for c in CONDITIONS]
        bars=ax.bar(x+offset,values,w,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3,fontsize=8)
    ax.set_xticks(x,short,rotation=18,ha='right');ax.set_ylabel('ABR reconstruction RMSE (µV)')
    ax.legend(frameon=False,ncol=2);ax.set_ylim(0,ax.get_ylim()[1]*1.18)
    save(fig,'04_reconstruction_rmse','4 · Can the frozen MLP reconstruct these examples?',
         'Known-truth reconstruction test using the same existing 2-million-update MLP for every condition, with 16 reconstructions averaged per EEG. Guidance was chosen separately per simulation condition on another subject, then frozen. The gray bars show the earlier unnormalized guidance value 6.25. Scores include one explicit absent-response control among 10 cases.')
    fig,ax=plt.subplots(figsize=(12,4.5))
    for key,label,color in [('legacy_white','Original simulator','#8796b4'),('real_quiet_native','New simulator','#5264db'),('real_eeg','Recorded speech EEG','#ca774c')]:
        f,p=welch(example[key],fs=10000,nperseg=1000,noverlap=500)
        keep=(f>=30)&(f<=2000)
        ax.semilogy(f[keep],p[keep],label=label,color=color,lw=1.7)
    ax.set(xlabel='Frequency (Hz)',ylabel='EEG power density (µV²/Hz)');ax.legend(frameon=False)
    save(fig,'05_spectrum','5 · Frequency content and amplitude',
         'Absolute power spectra for the displayed examples. The new simulator and recorded speech EEG share the same processing pipeline; the original white-noise simulator did not have this EEG preprocessing. Similarity here is illustrative for one window, not proof of a calibrated population-wide model.')
    tm=(np.arange(256)-50)/10
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,c,title in zip(axes,['legacy_white','real_quiet_native','real_quiet_double'],['Original','Quiet ×1','Quiet ×2']):
        ax.plot(tm,preds['truth'][0],color='#1c2a45',lw=2,label='Clean ABR')
        ax.plot(tm,preds[c+'__validation_selected'][0],color='#5264db',lw=1.7,label='Reconstructed')
        ax.set(title=title,xlabel='Time (ms)')
    axes[0].set_ylabel('ABR (µV)');axes[-1].legend(frameon=False,fontsize=9)
    save(fig,'06_reconstructed_abrs','6 · Actual reconstructed waveforms',
         'The first nonzero test ABR is shown for every condition, without selecting an unusually good example. The clean synthetic waveform is exactly known. These are single-band simulations; they do not yet include ten simultaneous unknown ABRs.')
    table=[];summary=[]
    for c in CONDITIONS:
        selected=rows(c,'validation_selected');fixed=rows(c,'fixed_6_25');nonzero=[r for r in selected if r['null_control']=='False']
        noise=float(np.mean([float(r['noise_rms_uv']) for r in amp if r['condition']==c]))
        total=float(np.mean([float(r['rmse_uv']) for r in selected]));nz=float(np.mean([float(r['rmse_uv']) for r in nonzero]))
        pe=float(np.mean([float(r['peak_amplitude_error_uv']) for r in nonzero]))
        le=float(np.mean([float(r['peak_latency_error_ms']) for r in nonzero]))
        null=next(float(r['peak_amplitude_error_uv']) for r in selected if r['null_control']=='True')
        row={'condition':c,'noise_rms_uv':noise,'guidance':choices[c]['guidance'],'test_rmse_uv':total,
             'nonnull_rmse_uv':nz,'peak_amplitude_mae_uv':pe,'peak_latency_mae_ms':le,'null_false_peak_uv':null}
        summary.append(row)
        table.append(f'<tr><td>{LABELS[c]}</td><td>{noise:.3f}</td><td>{choices[c]["guidance"]:g}</td><td>{total:.5f}</td><td>{nz:.5f}</td><td>{pe:.5f}</td><td>{le:.2f}</td><td>{null:.5f}</td></tr>')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    native=next(r for r in summary if r['condition']=='real_quiet_native')
    old=next(r for r in summary if r['condition']=='legacy_white')
    boundaries=[LABELS[c] for c in CONDITIONS if choices[c]['boundary']]
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Realistic simulated EEG</title><style>body{{max-width:1150px;margin:auto;padding:44px 24px;background:#f3f7fb;color:#1c2a45;font:16px/1.65 system-ui}}h1{{font-size:clamp(2.2rem,5vw,3.5rem);line-height:1.1}}section{{background:white;padding:25px;border-radius:14px;border:1px solid #dfe7ef;margin:20px 0}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:.86rem;width:100%}}td,th{{padding:8px;border-bottom:1px solid #e2eaf0;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#435fc6}}.note{{border-left:4px solid #ca774c;padding:12px;background:#fff7ee}}</style><h1>Simulated EEG with real timings and recorded background</h1><section><p>The simulator now uses a real speech stimulus pulse train, calibrated recording amplitudes following the original-author gain convention, and contiguous quiet-EEG segments extracted before any annotated playback. The existing 2-million-update MLP was frozen; no denoiser was retrained.</p><p>In this test, native quiet EEG has mean RMS <strong>{native['noise_rms_uv']:.2f} µV</strong>, versus <strong>{old['noise_rms_uv']:.2f} µV</strong> in the old-style white-noise simulation. The new native-amplitude reconstruction RMSE is <strong>{native['test_rmse_uv']:.5f} µV</strong>, versus <strong>{old['test_rmse_uv']:.5f} µV</strong> for the old-style simulation, after validation-selected guidance in each case. These conditions have different noise levels and timing; their difference does not isolate one factor.</p><p><a href="{(out/'forge_url.txt').read_text().strip()}">Open the experiment in CoreWeave Forge</a></p></section>{''.join(blocks)}<section><h2>Exact test results</h2><div class="scroll"><table><tr><th>Condition</th><th>Noise RMS µV</th><th>Guidance</th><th>All-case RMSE µV</th><th>Nonzero RMSE µV</th><th>Peak amplitude MAE µV</th><th>Peak latency MAE ms</th><th>Null false peak µV</th></tr>{''.join(table)}</table></div><p>Peak measurements use the 2–9 ms interval. One null-response example is insufficient to establish a false-positive rate. The 10 test windows come from one recording, so these are exploratory examples rather than independent-subject performance estimates.</p></section><section><h2>How realism and amplitude were handled</h2><p>BrainVision header resolutions convert stored values to µV. We then divide by 100 for the external EP preamplifier, matching the original authors' analysis, and average FCz–A1 and FCz–A2. No scaling to RMS = 1 is used for simulation or amplitude plots. Native quiet EEG is reused at its measured strength; the other cases use explicitly declared multipliers.</p><p>The regular/real matched-white conditions use the same post-filter noise RMS as their corresponding quiet-EEG segments. This provides a way to separate changes in pulse timing and noise structure from a simple increase in amplitude. Both simulated signal and noise use the same causal 30–2000 Hz and notch filters. The forward matrix is filtered too, so reconstruction predicts the processed EEG consistently without filtering the ABR target twice.</p><p>Pulse times are mapped from the actual 48 kHz stimulus file to the measured EEG clock. Corrected trial onsets include the tubing delay. The selected regressor is 4 kHz left; we simulate one unknown ABR at a time. All ten overlapping band/ear responses and trial-dependent ABR changes remain future extensions.</p></section><section><h2>Quiet EEG, truth, and limitations</h2><p>Validation uses subject exp302; test uses exp304. Background windows start at 2.5 s and end at least 0.5 s before the first annotated playback. The first second of each extracted context is used to settle the filter. Experimental playback code was inspected to verify the stimulus starts after the instruction screen, but no independent microphone record was available. These are event-defined quiet backgrounds, not certified absence of all auditory or biological activity.</p><p>The clean targets are generated synthetic ABRs with declared µV scaling, not recovered ground truth from the real recording. We preserve the established target generator to isolate acquisition changes; this does not establish that its amplitudes or morphology represent all real subjects. Quiet EEG can differ from EEG during speech. Reusing one subject's nearby quiet windows also gives correlated examples.</p><p>Guidance is selected separately for each known simulation condition on eight validation cases. This is an optimistic condition-aware evaluation, not an automatic noise-adaptation algorithm for unseen recordings. Boundary-selected grid conditions: {html.escape(', '.join(boundaries) or 'none')}. A boundary choice is not a proven optimum. Each test estimate averages 16 draws; the residual loss is not divided by EEG noise SD.</p><p>More realistic simulation can produce worse reconstruction scores because it removes favorable assumptions. The next steps are longer recording durations, more subjects with verified quiet intervals, realistic clean ABR references, and joint multiband reconstruction. Elapsed main experiment time: {runtime['seconds']/60:.1f} minutes.</p></section><section><h2>Protocol</h2><pre>{html.escape(json.dumps(cfg,indent=2))}</pre><p>CSV files contain every guidance score, per-recording reconstruction error, physical amplitude, and the exact quiet-window sample positions. PNG/PDF/SVG plots have minimal labels; this report contains their context.</p></section></html>'''
    if choices['real_quiet_native']['guidance']==0:
        warning='<p class="note"><strong>Reconstruction limitation:</strong> all recorded-quiet-background conditions selected guidance zero. Their identical selected RMSEs come from the same unguided model samples, which ignore EEG. This is not evidence that we successfully recovered ABRs or became robust to the noise. Fixed guidance 6.25 performs worse on these conditions.</p>'
        body=body.replace('</section>',warning+'</section>',1)
    if (out/'weighted_selection.json').exists() and (out/'weighted_test_metrics.csv').exists():
        selection=json.loads((out/'weighted_selection.json').read_text())
        weighted=float(np.mean([float(r['rmse_uv']) for r in read(out/'weighted_test_metrics.csv')]))
        weighted_fixed=''
        if (out/'weighted_fixed_test_metrics.csv').exists():
            fixed=float(np.mean([float(r['rmse_uv']) for r in read(out/'weighted_fixed_test_metrics.csv')]))
            ordinary=float(np.mean([float(r["rmse_uv"]) for r in rows("real_quiet_native","fixed_6_25")]))
            weighted_fixed=f'<p>At the same previously specified nonzero guidance 6.25, spectral weighting gave RMSE {fixed:.5f} µV, versus {ordinary:.5f} µV with the ordinary loss at native quiet noise. This descriptive fixed-guidance comparison does not change the validation-selected choice.</p>'
        section=f'<section><h2>Follow-up: account for correlated background</h2><p>We estimated a frequency-dependent weight from quiet EEG in validation subject exp302 only, using inverse square-root spectral power with a floor to prevent extreme weights. Both the forward matrix and measured EEG were transformed identically. The test recording and clean test ABRs were not used to choose weights or guidance.</p><p>The selected guidance was {selection["guidance"]:g}, with test RMSE {weighted:.5f} µV. The selected zero value again means the sampler ignored the EEG; this initial spectral-weighting attempt did not establish successful recovery from native background in 500 ms.</p>{weighted_fixed}<p>This weights the frequency structure of the residual rather than dividing by a known test noise standard deviation. A longer recording and a better validated noise model are needed before claiming accurate reconstruction at these noise levels.</p></section>'
        body=body.replace('<section><h2>Exact test results</h2>',section+'<section><h2>Exact test results</h2>')
    report=out/'REALISTIC_EEG_REPORT.html';report.write_text(body,encoding='utf-8');return report
