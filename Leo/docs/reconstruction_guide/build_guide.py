"""Build a self-contained explanation of Leo/dps/reconstruct.py."""

import base64
import html
from pathlib import Path
from string import Template
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EQUATIONS = {
    "01_forward": (
        r"y = Mh + \varepsilon",
        "The EEG is modeled as the sum of overlapping ABRs plus noise.",
    ),
    "02_noising": (
        r"h_t = \sqrt{\bar{\alpha}_t}\,h_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon",
        "Training creates a noisy waveform from a clean waveform and random noise.",
    ),
    "03_v_target": (
        r"v = \sqrt{\bar{\alpha}_t}\,\varepsilon - \sqrt{1-\bar{\alpha}_t}\,h_0",
        "The network is trained to predict this combination, called v.",
    ),
    "04_clean": (
        r"\hat{h}_0 = \sqrt{\bar{\alpha}_t}\,h_t - \sqrt{1-\bar{\alpha}_t}\,\hat{v}_{\theta}(h_t,t)",
        "The model’s v prediction and current waveform give an estimate of the clean ABR.",
    ),
    "05_eeg_loss": (
        r"\hat{y}=M\hat{h}_0, \qquad L_t=\frac{\|y-\hat{y}\|_2}{\sqrt{N}}",
        "The guidance loss is the RMS mismatch between observed and predicted EEG.",
    ),
    "06_gradient": (
        r"g_t=\nabla_{h_t}L_t",
        "The gradient says how the current waveform affects the EEG mismatch.",
    ),
    "07_schedule": (
        r"\alpha_t=\frac{\bar{\alpha}_t}{\bar{\alpha}_{\mathrm{prev}}}, \qquad \beta_t=1-\alpha_t",
        "The current and previous noise levels determine the step coefficients.",
    ),
    "08_mean": (
        r"\mu_t=\frac{\sqrt{\bar{\alpha}_{\mathrm{prev}}}\,\beta_t}{1-\bar{\alpha}_t}\hat{h}_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{\mathrm{prev}})}{1-\bar{\alpha}_t}h_t",
        "The diffusion mean combines the estimated clean waveform with the current waveform.",
    ),
    "09_variance": (
        r"\tilde{\beta}_t=\frac{\beta_t(1-\bar{\alpha}_{\mathrm{prev}})}{1-\bar{\alpha}_t}",
        "The variance controls random noise added at the next step.",
    ),
    "10_guidance": (
        r"h_{\mathrm{next}}=\mu_t+\sqrt{\tilde{\beta}_t}\,z-\lambda g_t",
        "The update combines the diffusion step, scheduled noise, and EEG-guidance correction.",
    ),
    "11_rmse": (
        r"\operatorname{RMSE}=\sqrt{\frac{1}{256}\sum_{i=1}^{256}(\hat{h}_i-h_i)^2}",
        "When a known reference ABR is available, RMSE measures the reconstructed waveform error.",
    ),
}


def main():
    folder = HERE / "equations"
    folder.mkdir(parents=True, exist_ok=True)
    figures = {}
    for name, (formula, caption) in EQUATIONS.items():
        fig = plt.figure(figsize=(11, 1.1), facecolor="white")
        fig.text(
            0.5,
            0.5,
            "$" + formula + "$",
            ha="center",
            va="center",
            fontsize=25,
            color="#11233b",
        )
        path = folder / f"{name}.png"
        fig.savefig(
            path, dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white"
        )
        plt.close(fig)
        encoded = base64.b64encode(path.read_bytes()).decode()
        figures[name] = (
            f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)} <a href="equations/{name}.png" download>Download PNG</a></figcaption></figure>'
        )
    with zipfile.ZipFile(HERE / "equation_pngs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.glob("*.png")):
            z.write(p, p.name)
    source = html.escape((ROOT / "dps/reconstruct.py").read_text())
    template = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>How my ABR reconstruction works</title><style>
:root{color-scheme:light;--ink:#172940;--muted:#52677c;--blue:#17669b;--rule:#dbe5ec;--bg:#f2f6f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,-apple-system,sans-serif}main{max-width:1050px;margin:auto;padding:52px 24px 84px}header{padding:8px 0 22px}h1{font-size:clamp(2.3rem,5vw,3.6rem);line-height:1.1;letter-spacing:-.035em;margin:12px 0 20px}h2{font-size:1.55rem;line-height:1.25;margin:0 0 15px}h3{font-size:1.05rem;margin:24px 0 7px}.eyebrow{font-size:.75rem;letter-spacing:.16em;font-weight:700;text-transform:uppercase;color:var(--blue)}.lead{max-width:750px;font-size:1.12rem;color:var(--muted)}section{background:#fff;border:1px solid var(--rule);border-radius:14px;padding:28px 32px;margin:20px 0;box-shadow:0 4px 20px #243f5310}p{margin:10px 0 16px}a{color:#0b689b}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;background:#edf3f6;padding:2px 5px;border-radius:4px}pre{overflow:auto;background:#142a40;color:#e6f0f7;padding:20px;border-radius:10px;font-size:.85rem;line-height:1.6}pre code{background:none;color:inherit;padding:0}figure{margin:22px 0;background:#f8fbfc;border:1px solid var(--rule);border-radius:10px;padding:10px 14px}figure img{display:block;width:auto;max-width:100%;max-height:105px;height:auto;margin:4px auto}figcaption{color:var(--muted);font-size:.87rem;line-height:1.45;padding:3px 5px 7px}figcaption a{white-space:nowrap;margin-left:5px}.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:20px 0}.flow span{flex:1;min-width:135px;padding:12px;background:#e9f3f7;border-radius:8px;text-align:center;font-weight:650;font-size:.87rem}.note{background:#eaf4f8;border-left:4px solid #2684ad;padding:14px 18px;border-radius:3px}.tag{display:inline-block;padding:3px 9px;border-radius:30px;background:#e6f2f6;color:#14648a;font-size:.8rem;font-weight:700}table{border-collapse:collapse;width:100%;font-size:.93rem}td,th{padding:10px 8px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}th{color:#355269}details summary{cursor:pointer;font-weight:700;color:var(--blue)}@media(max-width:650px){main{padding:30px 14px}section{padding:22px 17px}table{font-size:.82rem}td,th{padding:8px 4px}}
</style></head><body><main><header><div class="eyebrow">CS163 · A guide to your code</div><h1>How my ABR reconstruction works</h1><p class="lead">A step-by-step guide to <code>Leo/dps/reconstruct.py</code>: what goes in, how the model and EEG work together, and what each loss actually measures.</p><p><a href="equation_pngs.zip">Download all equation PNGs</a></p></header>
<section><h2>Start with the goal</h2><p>Given an EEG recording and the times at which sound pulses occurred, estimate the small auditory brainstem response (ABR) that each pulse produced. The code assumes one shared ABR shape and adds the responses from all pulses together.</p>$E01<div class="flow"><span>Known pulses + EEG</span><span>Start with random noise</span><span>Model proposes an ABR</span><span>EEG guides the estimate</span><span>Repeat → reconstructed ABR</span></div><p>The ε in this first equation represents EEG measurement noise. The model has learned what ABR-like waveforms look like. The EEG-guidance term checks whether a proposed waveform also explains this recording.</p></section>
<section><h2>1 · Inputs and output</h2><pre><code>reconstruct(model, eeg, pulses, *, guidance=0.1, seed=1)</code></pre><table><tr><th>Input</th><th>Meaning in this function</th></tr><tr><td><code>model</code></td><td>A trained v-prediction network. <code>load_model()</code> sets evaluation mode and freezes its weights.</td></tr><tr><td><code>eeg</code></td><td>A one-dimensional recording of N samples, in the model’s amplitude units.</td></tr><tr><td><code>pulses</code></td><td>N event weights marking stimulus times. These are event weights, not EEG voltages.</td></tr><tr><td><code>guidance</code></td><td>The strength of the EEG-based adjustment. The default is 0.1.</td></tr><tr><td><code>seed</code></td><td>Controls random draws, so a run can be repeated.</td></tr></table><p>The function returns one 256-sample estimated ABR in model units. In this project, one model unit corresponds to 0.1 µV; <code>run_eeg.py</code> divides EEG by 0.1 before reconstruction and multiplies the output by 0.1 to obtain µV.</p></section>
<section><h2>2 · The stimulus matrix: ABR → predicted EEG</h2><p><code>stimulus_matrix(pulses)</code> builds a matrix <strong>M</strong> with N rows and 256 columns. Each column shifts the pulse train to represent one point of a candidate ABR. Multiplying M by an ABR adds up responses at every pulse time, including overlaps.</p><pre><code>matrix = stimulus_matrix(pulses)
predicted_eeg = matrix @ clean[0]</code></pre><p><code>clean</code> has shape <code>(1, 256)</code>; <code>clean[0]</code> selects its waveform. The result has N samples, matching <code>eeg</code>. Index 50 of the ABR corresponds to stimulus onset: at 10,000 samples per second, the first 50 samples cover 5 ms before onset. The matrix masks shifts that would fall outside the recording.</p><p class="note"><strong>Model assumption:</strong> each pulse produces the same waveform and overlapping responses add linearly. Whether that is adequate for a real recording is a scientific question.</p></section>
<section><h2>3 · Why the network predicts “v”</h2><p>During training, the experiment knows a reference waveform <strong>h₀</strong> and adds generated Gaussian noise <strong>ε</strong>. This diffusion noise is separate from the EEG measurement noise in the first equation. At diffusion step t, the schedule <code>alpha_bar</code> determines the mixture:</p>$E02<p>The model’s target is another combination of the same waveform and noise:</p>$E03<p>The network receives the noisy waveform and t, and predicts <code>v</code>. Its weights are learned beforehand; they are not updated during reconstruction.</p><h3>What <code>clean_from_v</code> does</h3><p>Given the noisy waveform and predicted v, the following identity estimates the clean waveform. It would recover the training waveform exactly if v were predicted exactly; in practice it is an estimate.</p>$E04<pre><code>predicted_v = model(current, step)
clean = clean_from_v(current, predicted_v, step, alpha_bar)</code></pre><p>Here, <code>current</code> is hₜ and <code>clean</code> is the estimate ĥ₀. The helper’s implementation is <code>signal_weight * noisy - noise_weight * predicted_v</code>.</p></section>
<section><h2>4 · Initialization</h2><pre><code>alpha_bar = cosine_schedule()
generator = torch.Generator().manual_seed(seed)
current = torch.randn((1, WAVEFORM_LENGTH), generator=generator)</code></pre><p>The schedule has 200 diffusion steps. The reconstruction starts as random noise, not as an EEG waveform. The seed controls this initial noise and the additional diffusion noise drawn later.</p></section>
<section><h2>5 · The loop, one step at a time</h2><p>The loop counts backward from t = 199 to t = 0. Each pass makes a candidate clean ABR, checks it against the EEG, and computes the next waveform.</p><h3>5.1 · Track how the current waveform affects the result</h3><pre><code>current = current.detach().requires_grad_(True)
step = torch.tensor([t], dtype=torch.long)</code></pre><p><code>detach()</code> separates the current iteration from earlier calculation histories. <code>requires_grad_(True)</code> lets PyTorch calculate how the current waveform affects this iteration’s mismatch. With a model obtained from <code>load_model()</code>, its parameters are frozen. The reconstruction function itself performs no optimizer update, even if a caller passes an unfrozen model.</p><h3>5.2 · Estimate a clean ABR</h3><pre><code>predicted_v = model(current, step)
clean = clean_from_v(current, predicted_v, step, alpha_bar)</code></pre><p>The model uses t to know the current noise level. The helper applies the clean-estimate equation above.</p><h3>5.3 · Predict the recording and measure its mismatch</h3><pre><code>predicted_eeg = matrix @ clean[0]
residual = eeg - predicted_eeg
loss = torch.linalg.vector_norm(residual) / np.sqrt(len(eeg))</code></pre>$E05<p>Here y is the observed EEG, M is the stimulus matrix, and N is the number of EEG samples. The result is the <strong>root mean square of the EEG residual</strong>. It measures how well the candidate explains the recording; it is not the reconstruction RMSE against a known ABR.</p><h3>5.4 · Get the guidance direction</h3><pre><code>gradient = torch.autograd.grad(loss, current)[0]</code></pre>$E06<p>PyTorch differentiates through the predicted EEG, clean estimate, and network back to <code>current</code>. The gradient points toward increasing mismatch; the update subtracts it to nudge the waveform toward a lower mismatch.</p><h3>5.5 · What are we trying to do next?</h3>
<p>At the start of this block, <code>current</code> is the noisy ABR waveform at step <em>t</em>, <code>clean</code> is the network’s estimate of the final clean ABR, and <code>gradient</code> tells us how the EEG mismatch changes with <code>current</code>. Each is a waveform with 256 values. The loop moves backward from <em>t</em> = 199 toward 0, making a new <code>current</code> at each step.</p>
<div class="flow"><span>Current noisy waveform</span><span>Calculate a cleaner diffusion step</span><span>Add scheduled randomness</span><span>Nudge using the EEG</span></div>
<p>Here is the entire block. The explanations below follow its lines in order:</p>
<pre><code>a = alpha_bar[t]
previous_a = alpha_bar[t - 1] if t &gt; 0 else torch.tensor(1.0)
alpha = a / previous_a
beta = 1 - alpha
mean = (previous_a.sqrt() * beta / (1 - a)) * clean
mean = mean + (alpha.sqrt() * (1 - previous_a) / (1 - a)) * current
variance = beta * (1 - previous_a) / (1 - a)
noise = torch.randn(current.shape, generator=generator) if t &gt; 0 else 0
current = mean + variance.sqrt() * noise - guidance * gradient</code></pre>
<h3>5.6 · Look up the current and next noise levels</h3>
<pre><code>a = alpha_bar[t]
previous_a = alpha_bar[t - 1] if t &gt; 0 else torch.tensor(1.0)</code></pre>
<p><code>alpha_bar</code> is a precomputed schedule describing how much of the original signal is retained after forward diffusion. In the noising equation in section 3, the clean waveform is multiplied by the square root of this value, and the noise by the square root of one minus this value. Values near 1 correspond to less noise; values near 0 correspond to more noise.</p>
<p><code>a</code> is the schedule value at the current step <em>t</em>. <code>previous_a</code> is the value at <em>t</em> − 1: the next, cleaner stage during reconstruction. At <em>t</em> = 0 there is no earlier array entry, so the code uses 1 to represent the clean endpoint.</p>
<h3>5.7 · Find the coefficients for this one step</h3>
<pre><code>alpha = a / previous_a
beta = 1 - alpha</code></pre>
$E07
<p><code>a</code> and <code>previous_a</code> are cumulative schedule values. Their ratio, <code>alpha</code>, describes signal retention for the single transition between these two stages; <code>beta</code> describes the corresponding forward-step noise variance. The code uses them to calculate the reverse step.</p>
<h3>5.8 · Find the center of the next step</h3>
<pre><code>mean = (previous_a.sqrt() * beta / (1 - a)) * clean
mean = mean + (alpha.sqrt() * (1 - previous_a) / (1 - a)) * current</code></pre>
$E08
<p>Read the two lines together as <code>mean = clean_weight * clean + current_weight * current</code>. The weights come from the noise schedule. The first line takes part of the network’s predicted clean ABR; the second adds part of the current noisy waveform. They are calculated for every step and applied to every waveform point.</p>
<p><code>mean</code> is a 256-point waveform, not one average number. It is the center of the reverse-diffusion distribution: the default next waveform before the random and EEG-guidance terms are added. The network’s <code>clean</code> is an estimate, so it need not be the true ABR.</p>
<h3>5.9 · Set the random part</h3>
<pre><code>variance = beta * (1 - previous_a) / (1 - a)
noise = torch.randn(current.shape, generator=generator) if t &gt; 0 else 0</code></pre>
$E09
<p><code>variance</code> controls how widely the next sample can vary around <code>mean</code>. For <em>t</em> &gt; 0, <code>torch.randn</code> draws Gaussian noise with one value for each point of the waveform. A fixed seed makes these draws reproducible. At <em>t</em> = 0, the code draws no new noise and sets this term to zero.</p>
<h3>5.10 · Combine denoising, randomness, and EEG guidance</h3>
<pre><code>current = mean + variance.sqrt() * noise - guidance * gradient</code></pre>
$E10
<table><tr><th>Term</th><th>What it does</th></tr><tr><td><code>mean</code></td><td>Provides the center of the next, cleaner diffusion step.</td></tr><tr><td><code>variance.sqrt() * noise</code></td><td>Adds random variation at the scheduled scale. The square root converts variance to the standard deviation used to scale Gaussian noise.</td></tr><tr><td><code>- guidance * gradient</code></td><td>Uses the EEG mismatch to nudge the waveform. The gradient points locally toward increasing the loss with respect to <code>current</code>; subtracting it points toward decreasing that loss.</td></tr></table>
<p>These terms act on all 256 waveform values. The gradient was calculated through the predicted EEG, <code>clean</code>, and the network back to the old <code>current</code>. It is combined with the diffusion step; because the mean and random term also move the waveform, the EEG loss is not guaranteed to decrease at every iteration.</p>
<p>λ is <code>guidance</code>. If guidance is zero, the EEG does not steer the output. Larger values strengthen the correction; in this implementation the correction is not multiplied by the diffusion variance. Choose guidance on separate validation data. To reproduce an earlier update for a fixed recording, divide the old guidance value by the old noise SD in model units.</p>
<h3>5.11 · A worked example for one waveform point</h3>
<p>These are illustrative values, not measured results. In a real iteration the same schedule coefficients apply to all waveform points, while the clean estimate, current value, noise and gradient vary by point. Suppose <code>a = 0.60</code>, <code>previous_a = 0.75</code>, <code>clean = 0.80</code>, <code>current = 0.40</code>, <code>noise = -0.20</code>, <code>gradient = 0.10</code> and <code>guidance = 0.20</code>. Then:</p>
<table><tr><th>Calculation</th><th>Approximate result</th></tr><tr><td><code>alpha = 0.60 / 0.75</code>; <code>beta = 1 - alpha</code></td><td>0.80 and 0.20</td></tr><tr><td><code>mean = 0.433 * 0.80 + 0.559 * 0.40</code></td><td>0.570</td></tr><tr><td><code>variance = 0.20 * (1 - 0.75) / (1 - 0.60)</code></td><td>0.125</td></tr><tr><td><code>new current = 0.570 + sqrt(0.125) * (-0.20) - 0.20 * 0.10</code></td><td>0.479</td></tr></table>
<p>In this example the diffusion mean moves the point from 0.40 to about 0.570; the random term subtracts about 0.071, and the EEG-guidance term subtracts 0.020. The model and the EEG are therefore both involved in the next value.</p>
<h3>5.12 · The last step</h3>
<p>At <em>t</em> = 0, <code>previous_a = 1</code>. Substituting that value makes <code>mean = clean</code> and <code>variance = 0</code>. No new random noise is drawn, so the last update simplifies to:</p>
<pre><code>current = clean - guidance * gradient</code></pre>
<p>After this last correction, the function returns <code>current.detach().numpy()[0]</code> as the estimated ABR. The model weights are not trained or changed by this loop.</p></section>
<section><h2>6 · Three different errors</h2><table><tr><th>Quantity</th><th>What is compared?</th><th>Purpose</th></tr><tr><td>Training v-MSE</td><td>Predicted v vs the known training target v</td><td>Train and validate the network’s diffusion prediction.</td></tr><tr><td>Guidance loss</td><td>Observed EEG vs EEG predicted by the current ABR estimate; RMS across EEG samples</td><td>Calculate the gradient that steers each reconstruction step.</td></tr><tr><td>Reconstruction RMSE</td><td>Final reconstructed ABR vs a known reference ABR</td><td>Evaluate recovery in a simulation or other setting with a reference.</td></tr></table>$E11<p>For real EEG the true ABR is generally unknown, so this exact reference-based RMSE is not directly available. The simulated experiments start from a known reference, use it to generate noisy EEG, reconstruct, then calculate this error. If the reference and reconstruction are in model units, multiply the RMSE by 0.1 to report µV.</p></section>
<section><h2>7 · One sample versus many</h2><p>This particular <code>reconstruct()</code> call returns <strong>one</strong> sample. Repeating it with different seeds can produce different waveforms. An experiment may average several samples before calculating RMSE, or calculate RMSE for each sample and average those errors. Those are different measurements.</p><p>The separate batched experiment code runs the same kind of guidance calculation for multiple observations. The existing batched experiment implementations still use their original noise-normalized loss, so their historical results are unchanged. Later experiments may also apply a covariance-based correction after averaging; that correction is not part of this function.</p></section>
<section><h2>Symbols at a glance</h2><table><tr><th>Symbol</th><th>In your code</th><th>Meaning</th></tr><tr><td>y</td><td><code>eeg</code></td><td>Observed recording.</td></tr><tr><td>M</td><td><code>matrix</code></td><td>Pulse-timing matrix.</td></tr><tr><td>hₜ</td><td><code>current</code></td><td>Current noisy ABR estimate.</td></tr><tr><td>v̂</td><td><code>predicted_v</code></td><td>Network prediction at step t.</td></tr><tr><td>ĥ₀</td><td><code>clean</code></td><td>Current estimate of the clean ABR.</td></tr><tr><td>gₜ</td><td><code>gradient</code></td><td>How EEG mismatch changes with the current waveform.</td></tr><tr><td>λ</td><td><code>guidance</code></td><td>Strength of the EEG correction.</td></tr></table></section>
<section><details><summary>Show the complete source used for this explanation</summary><pre><code>$SOURCE</code></pre></details><p>Equations are embedded in this HTML for offline reading. Their standalone PNG files are in the accompanying <code>equations</code> folder and ZIP archive.</p></section></main></body></html>""")
    values = {f"E{i:02d}": figures[name] for i, name in enumerate(EQUATIONS, 1)}
    values["SOURCE"] = source
    (HERE / "RECONSTRUCTION_GUIDE.html").write_text(
        template.substitute(values), encoding="utf-8"
    )
    print(f"Wrote guide and {len(EQUATIONS)} equation PNGs to {HERE}")


if __name__ == "__main__":
    main()
