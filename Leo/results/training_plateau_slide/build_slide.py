"""A single graph slide from the fixed-setting checkpoint validation results."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
with (ROOT/'Leo/runs/phoom_matched_10m/validation.csv').open() as f:
    records = list(csv.DictReader(f))
steps = [500000, 1000000, 2000000, 5000000]
lookup = {int(r['update']): float(r['rmse_uv']) for r in records if float(r['ema_rate']) == .005}
x = np.array(steps) / 1e6
y = np.array([lookup[n] for n in steps]) * 1000  # µV → nV

plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':'#18223B',
    'axes.labelcolor':'#637087', 'xtick.color':'#637087', 'ytick.color':'#637087'})
fig = plt.figure(figsize=(16, 9), facecolor='white')
ax = fig.add_axes([.14, .23, .79, .53])
fig.text(.14, .875, 'Reconstruction plateaus after 2M updates', fontsize=29, weight='bold')
ax.axvspan(2, 5.35, facecolor='#F5F6FF', zorder=0)
ax.axvline(2, color='#C6CFF5', lw=1.3, linestyle=(0, (3, 5)), zorder=1)
ax.plot(x, y, color='#4B5FE3', lw=3.4, marker='o', markersize=9,
        markerfacecolor='white', markeredgewidth=2.8, zorder=3)
ax.scatter([x[2]], [y[2]], s=125, color='#4B5FE3', edgecolor='white', linewidth=2.2, zorder=4)
for xi, yi in zip(x, y):
    ax.annotate(f'{yi:.2f}', (xi, yi), xytext=(0, 17), textcoords='offset points',
                ha='center', color='#243254', fontsize=20, weight='bold')
ax.set_xlim(.15, 5.35)
ax.set_ylim(5.0, 5.95)
ax.set_xticks(x, ['0.5M', '1M', '2M', '5M'])
ax.set_yticks([5.0, 5.2, 5.4, 5.6, 5.8])
ax.set_xlabel('MLP training updates', fontsize=17, labelpad=22)
ax.set_ylabel('ABR reconstruction RMSE (nV)', fontsize=17, labelpad=19)
ax.tick_params(axis='both', labelsize=15, length=0, pad=12)
ax.yaxis.grid(True, color='#E9ECF3', linewidth=.9)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(HERE / f'training_plateau.{suffix}', dpi=200, facecolor='white')
plt.close(fig)
with (HERE/'plotted_data.csv').open('w', newline='') as f:
    writer=csv.writer(f);writer.writerow(['training_updates','validation_rmse_uv','validation_rmse_nv'])
    writer.writerows((s,lookup[s],lookup[s]*1000) for s in steps)

sys.path.insert(0, '/tmp/abr-slide-deps')
from pptx import Presentation
from pptx.util import Inches
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(16), Inches(9)
slide=prs.slides.add_slide(prs.slide_layouts[6])
slide.shapes.add_picture(str(HERE/'training_plateau.png'), 0, 0, width=prs.slide_width, height=prs.slide_height)
slide.notes_slide.notes_text_frame.text = (
    'Fixed-setting validation reconstruction results, EMA update rate 0.005. '
    '24 synthetic ABRs at each of three EEG noise levels; eight reconstructions averaged per EEG. '
    'The 2M→5M change is approximately +0.54% RMSE, and is not evidence of a statistically significant deterioration. '
    'Source: Leo/runs/phoom_matched_10m/validation.csv. '
    'nV values are µV values multiplied by 1000. '
    'The line chart uses a nonzero vertical-axis lower limit.'
    ' Reconstruction: cosine schedule, 200 reverse steps, v prediction, Gaussian initial noise, '
    'eight samples averaged per EEG, sampling seed 0; no post-reconstruction correction. '
    'Guidance [10, 31.622776601683793, 100] corresponds respectively to simulated EEG '
    'noise standard deviations [0.05, 0.2, 0.5] microvolts. Guidance loss is EEG residual RMS '
    'divided by the corresponding known simulated noise standard deviation. '
    'Validation waveform seed 90426; validation noise seed 5001. '
    'This is the historical noise-normalized evaluation, separate from the later guidance 6.25 sweep.'
)
prs.save(HERE/'training_plateau.pptx')
print(HERE/'training_plateau.pptx')
