import os
"""Publication figures with consistent visual semantics and explicit denominators."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from restore_overview_figure import build_overview
from plot_rq1_version_choices import build_version_choices
from plot_release_age import build_release_age
from plot_advisory_scopes import build_advisory_scopes
from plot_rq3_results import build_construction_stages, build_execution_outcomes
from plot_neighbor_budget import build_neighbor_budget
import numpy as np
import pandas as pd
NAMES={'gpt-5.4':'GPT-5.4','claude-sonnet-4-6':'Claude Sonnet 4.6','gemini-3.1-pro-preview':'Gemini 3.1 Pro','deepseek-v3.2':'DeepSeek V3.2','kimi-k2.5':'Kimi K2.5','qwen3.5-397b-a17b':'Qwen3.5 397B','qwen3-235b-a22b-instruct-2507':'Qwen3 235B','qwen3-30b-a3b-instruct-2507':'Qwen3 30B','minimax-m2.5':'MiniMax M2.5','meta-llama/llama-4-scout':'Llama 4 Scout','All observations':'All observations'}
INK='#20364A';MUTED='#627487';GRID='#E2E8EE';BLUE='#326AA8';TEAL='#147D75';AMBER='#A56120';CORAL='#BD6257';PURPLE='#76609D'
LB='#ECF2F9';LT='#EAF5F2';LA='#FCF3E7';GREY='#CAD4DF'
def main():
 root=Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]));ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,default=root/'analysis');ap.add_argument('--output',type=Path,default=root/'figures');ap.add_argument('--poppler-bin',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':GRID,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titlesize':11,'axes.titleweight':'bold','axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9.5,'xtick.major.size':0,'ytick.major.size':0,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':240})
 used={}
 def read(name):
  p=a.analysis/name;used[name]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
 def save(fig,name):
  for ext in ['pdf','svg','png']:fig.savefig(a.output/(name+'.'+ext),bbox_inches='tight',pad_inches=.12,facecolor='white')
  plt.close(fig)
 def order(z):
  z=z.copy();z['order']=z.model.map({m:i for i,m in enumerate(NAMES)});return z.sort_values('order')
 def grid(ax):ax.set_axisbelow(True);ax.grid(axis='x',color=GRID,lw=.7)
 def band(ax,n):
  for y in range(n):
   if y%2==0:ax.axhspan(y-.45,y+.45,color='#F4F7FA',zorder=-3,lw=0)
 def note(fig,text,y=.98):fig.text(.02,y,text,color=MUTED,fontsize=9,ha='left',va='top')

 # 1. Retain the submitted five-module overview and update its labels in place.
 overview=build_overview(a.output,poppler_bin=a.poppler_bin)

 # 2. Shared package rows with paired, compact version-string distributions.
 concentration=build_version_choices(a.analysis,a.output)
 used['rq1_concentration_checked.csv']=concentration['input_sha256']

 # 3. Compact quantile summaries with shading extended through both numeric columns.
 release_age=build_release_age(a.analysis,a.output)
 used['version_age_summary.csv']=release_age['input_sha256']

 # 4. Compact horizontal comparisons preserve the two nested publication scopes.
 advisory_scopes=build_advisory_scopes(a.analysis,a.output)
 used['security_funnels.csv']=advisory_scopes['input_sha256']

 # 5. Recorded stopping stages with full-population and reached-stage denominators.
 construction = build_construction_stages(a.analysis, a.output)
 used.update(construction['inputs_sha256'])

 # 6. Execution evidence among static passes and in the complete response population.
 execution = build_execution_outcomes(a.analysis, a.output)
 used.update(execution['inputs_sha256'])

 # 7. Compact cumulative and marginal yield preserve the same response population.
 neighbor = build_neighbor_budget(a.analysis, a.output)
 used.update(neighbor['inputs_sha256'])
 (a.output/'figure_manifest.json').write_text(json.dumps({'input_sha256':used,'style':'Submitted study overview and coordinated evidence plots','semantics':{'MR':BLUE,'VA':AMBER,'engineering_or_security_stage':TEAL},'formats':['pdf','svg','png'],'figure_formats':{'fig1_evaluation_framework':['pdf','png']},'overview_source':overview['source'],'figures':7},indent=2));print('Generated seven figures; the study overview is supplied as vector PDF and PNG preview.')
if __name__=='__main__':main()
