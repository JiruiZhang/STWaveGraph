"""Generate numerical summaries and plots. Dependencies: numpy, pandas, matplotlib.
Run: python scripts/rebuild_main_results.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'generated';O.mkdir(exist_ok=True)
methods=['STWaveGraph','DeepGFT','GraphST','STAGATE','SpaGCN','SEDR','DeepST','GAAEST','SpaCAE','3dOT']
families=['DLPFC','Breast','MouseBrain','osmFISH','MERFISH','STARmap']
raw=pd.read_csv(R/'data/benchmark_seed_metrics.csv');p=raw[raw.stage=='common_external']
metrics=['ARI','NMI','boundary_precision','boundary_recall','boundary_f1','boundary_assd_spacing']
result=[]
for fam in families:
 for method in methods:
  sub=p[(p.family==fam)&(p.method==method)]
  vals=sub.groupby('sample')[metrics].mean() if fam=='DLPFC' else sub[metrics]
  assert len(vals)==(12 if fam=='DLPFC' else 5)
  for metric in metrics:result.append(dict(family=fam,method=method,metric=metric,mean=vals[metric].mean(),sd=vals[metric].std(ddof=1),n=len(vals)))
summary=pd.DataFrame(result);summary.to_csv(O/'benchmark_mean_sd.csv',index=False)
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(O/(name+'.'+ext),dpi=160,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
for ax,metric in zip(axes,['ARI','boundary_f1']):
 v=summary[summary.metric==metric].pivot(index='method',columns='family',values='mean').reindex(index=methods,columns=families)
 im=ax.imshow(v,aspect='auto',cmap='YlGnBu');ax.set_xticks(range(6),families,rotation=40,ha='right');ax.set_yticks(range(10),methods);ax.set_title(metric)
 for i in range(10):
  for j in range(6):ax.text(j,i,f'{v.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if v.iloc[i,j]>(v.min().min()+v.max().max())/2 else 'black')
 fig.colorbar(im,ax=ax,shrink=.65)
save(fig,'benchmark_ARI_F1')
a=pd.read_csv(R/'data/ablation_paired_effects.csv');fig,axes=plt.subplots(1,2,figsize=(10,8),layout='constrained')
for ax,metric in zip(axes,['ARI','boundary_f1']):
 v=a[(a.stage=='external')&(a.metric==metric)].pivot(index='variant',columns='dataset',values='delta_mean').reindex(columns=['DLPFC','Breast','MouseBrain']);limit=np.nanmax(np.abs(v.to_numpy()));im=ax.imshow(v,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
 ax.set_xticks(range(3),v.columns,rotation=30,ha='right');ax.set_yticks(range(len(v)),v.index);ax.set_title('Alternative minus full: '+metric);fig.colorbar(im,ax=ax,shrink=.6)
save(fig,'ablation_effects')
s=pd.read_csv(R/'data/sensitivity.csv');s=s[s.stage=='external'];fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
for ax,(name,g) in zip(axes.flat,s.groupby('axis',sort=False)):
 g=g.sort_values('axis_value');ax.errorbar(g.axis_value,g.ARI_mean,yerr=g.ARI_std,fmt='o-',capsize=3);ax.set_title(name);ax.set_ylabel('ARI mean ± SD')
save(fig,'sensitivity_ARI')
print('Generated',len(summary),'benchmark summaries and 3 quantitative plots (PNG + SVG).')
