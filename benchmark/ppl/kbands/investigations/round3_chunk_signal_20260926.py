import json,numpy as np
from pathlib import Path
root=Path('/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/prc2')
def rank(x):
 out=np.empty(x.size,float); out[np.argsort(x,kind='stable')]=np.arange(x.size); return out
def rho(a,b):
 if np.ptp(a)==0 or np.ptp(b)==0: return None
 return float(np.corrcoef(rank(a),rank(b))[0,1])
for model in ['4B','llama8B']:
 d=np.load(root/f'{model}_t32_c8w6.npz'); measured=list(d['measured']); ref=measured.index(24); top=measured.index(128)
 keys=sorted({tuple(k.split('|')[1:3]) for k in d.files if k.startswith('calib|')})
 summary={}
 for op,q in keys:
  data={}
  for phase in ['calib','hold']:
   pre=f'{phase}|{op}|{q}|'
   f=d[pre+'fis']; f=np.maximum(f[:,ref].astype(float)-f[:,top],0)
   r=d[pre+'raw']; r=np.maximum(r[:,ref].astype(float)-r[:,top],0)
   data[phase]={'f':f,'raw':r,**{k:d[pre+k] for k in ['blk','cidx','mn','rpos','win']}}
  for blk in np.unique(data['calib']['blk']):
   scores={}
   for phase,v in data.items():
    sel=v['blk']==blk; c=v['cidx'][sel].astype(int); f=v['f'][sel]
    # cap row mass at 100 times median, matching prior robust safeguard
    unit=v['win'][sel].astype(int)*100000+v['rpos'][sel].astype(int)
    _,inv=np.unique(unit,return_inverse=True); mass=np.bincount(inv,weights=f)
    cap=100*np.median(mass); clipped=f*np.minimum(1,cap/np.maximum(mass[inv],1e-300))
    n=int(c.max())+1
    scores[phase]={}
    for cur,values in [('fis',f),('clip100',clipped)]:
     for denom,weights in [('mn2',v['mn'][sel].astype(float)**2),('raw',v['raw'][sel])]:
      num=np.bincount(c,weights=values,minlength=n); den=np.bincount(c,weights=weights,minlength=n)
      scores[phase][f'{cur}/{denom}']=num/np.maximum(den,1e-30)
   for cur,a in scores['calib'].items():
    b=scores['hold'][cur]; n=a.size; order=np.argsort(a); lo=order[:max(1,n//4)]; hi=order[-max(1,n//4):]
    r=rho(a,b)
    if r is not None:
     summary.setdefault((op,cur),[]).append({'rho':r,'hold_hi_lo':float(np.mean(b[hi])/max(np.mean(b[lo]),1e-30)),'train_hi_lo':float(np.mean(a[hi])/max(np.mean(a[lo]),1e-30))})
  del data
 for (op,cur),v in summary.items():
  print(json.dumps({'model':model,'op':op,'signal':cur,'blocks':len(v),'median_rank_rho':round(float(np.median([x['rho'] for x in v])),3),'positive_rho_blocks':sum(x['rho']>0 for x in v),'median_holdout_hi_lo':round(float(np.median([x['hold_hi_lo'] for x in v])),3),'median_train_hi_lo':round(float(np.median([x['train_hi_lo'] for x in v])),3)}),flush=True)
