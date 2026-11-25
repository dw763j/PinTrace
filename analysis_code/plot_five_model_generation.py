"""Render three separate outcome panels from the verified five-model summary."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    s=json.loads(a.summary.read_text());assert s['planned_responses']==2400
    order=list(s['models']);labels=[s['models'][x] for x in order]
    labels=[x.replace('Gemini 3.1 Pro Preview','Gemini 3.1 Pro') for x in labels]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(6.9,3.4),sharey=True)
    panels=[('static_pass','A  Static pass','PinTrace: 60 tasks','#335F9C'),('dynamic_pass','B  Test pass','BigCodeBench: 20 tasks','#A95A38'),('exact','C  Exact advice','PinTrace: 60 tasks','#277F8E')]
    for ax,(metric,title,sub,color) in zip(axes,panels):
        data=[next(x for x in s['paired'] if x['model']==m and x['metric']==metric) for m in order]
        lo=min(x['ci_low'] for x in data);hi=max(x['ci_high'] for x in data)
        left=10*np.floor((min(0,lo)-8)/10);right=10*np.ceil((max(0,hi)+8)/10)
        for i in range(5):
            if i%2==0:ax.axhspan(i-.47,i+.47,color='#F2F5F8',zorder=0)
        ax.axvline(0,color='#778390',linewidth=1,linestyle=(0,(3,3)),zorder=1)
        for i,x in enumerate(data):
            ax.plot([x['ci_low'],x['ci_high']],[i,i],color=color,linewidth=2.1,zorder=2,solid_capstyle='round')
            ax.scatter(x['delta_pp'],i,s=48,color=color,edgecolor='white',linewidth=.7,zorder=3)
            ax.text(x['delta_pp'],i-.22,f"{x['delta_pp']:+.1f}",ha='center',va='bottom',fontsize=8,fontweight='bold',color=color)
        ax.set_xlim(left,right);ax.set_ylim(4.6,-.72)
        ax.set_title(title,loc='left',fontweight='bold',pad=23,color='#1A3348')
        ax.text(0,1.035,sub,transform=ax.transAxes,color='#56697B',fontsize=8)
        ax.set_xlabel(r'$\widehat{\Delta}_{\theta}$ (VA - MR, pp)',labelpad=6)
        ax.set_yticks(range(5));ax.set_yticklabels(labels)
        ax.tick_params(axis='y',length=0,pad=8);ax.tick_params(axis='x',color='#8B9CAB')
        ax.grid(axis='x',color='#DDE5EB',linewidth=.55,zorder=0)
        for name in ['top','right','left']:ax.spines[name].set_visible(False)
        ax.spines['bottom'].set_color('#B0BDCA')
    fig.subplots_adjust(left=.235,right=.985,top=.80,bottom=.30,wspace=.35)
    fig.text(.03,.07,'Three calls per task and condition. Bars: 95% paired task-cluster bootstrap intervals.',fontsize=8,color='#42566A')
    fig.text(.03,.025,'Positive values indicate higher VA rates.',fontsize=8,color='#42566A')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(a.out.with_suffix('.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    print(str(a.out))
if __name__=='__main__':main()
