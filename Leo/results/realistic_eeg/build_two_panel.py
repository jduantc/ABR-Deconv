"""Two-panel comparison of new simulated EEG and recorded speech EEG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent
with np.load(out/'examples.npz') as saved:
    synthetic=saved['real_quiet_native'].copy()
    recorded=saved['real_eeg'].copy()
t=np.arange(len(recorded))/10
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#1c2a45',
    'axes.labelcolor':'#637087','xtick.color':'#637087','ytick.color':'#637087',
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
    'axes.spines.bottom':False,'axes.axisbelow':True,'axes.grid':True,'grid.color':'#e9edf4'})
fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,sharey=True)
fig.subplots_adjust(left=.10,right=.97,bottom=.14,top=.84,hspace=.40)
fig.text(.10,.93,'Simulated and real EEG',fontsize=25,weight='bold')
limit=1.08*max(np.max(np.abs(synthetic)),np.max(np.abs(recorded)))
for ax,y,label,color in zip(axes,[synthetic,recorded],['New simulated EEG','Real EEG'],['#5264db','#ca774c']):
    ax.plot(t,y,lw=1,color=color)
    ax.set_title(label,loc='left',color=color,fontsize=15,pad=12)
    ax.set_ylabel('EEG (µV)',labelpad=12)
    ax.set_xlim(0,500);ax.set_ylim(-limit,limit);ax.tick_params(length=0,pad=8)
axes[-1].set_xlabel('Time (ms)',labelpad=12)
for extension in ('png','pdf','svg'):
    fig.savefig(out/f'new_simulated_vs_real.{extension}',dpi=200,facecolor='white')
plt.close(fig)
print(out/'new_simulated_vs_real.png')
