"""Part B of the sorting cost study: tournament and sorting-tree totals from the price curve C(eps).

Reads results/price_curve.json (made by src/price_curve.py, which imports core.py) and interpolates
log-log between the computed eps values (0.001 to 0.3). Error budget: total failure probability 0.01
by the union bound over the comparisons on the executed path.
  B1 single elimination tournament with N entrants (N-1 comparisons), optimized eps split
  B2 sorting trees for N = 3 and 4, worst path photons
  B3 fits C(eps) = a ln(1/eps) + b and c (1/eps)^p on eps 0.001..0.1, extrapolation to N = 8, 16, 32
Writes results/tournament.txt (or the path in TOURN_OUT).

Usage: python3 src/tournament.py
"""
import os
import json, numpy as np, math
from scipy.optimize import minimize
R=json.load(open(os.environ.get('PRICE_JSON','results/price_curve.json')))
E=np.array(sorted(float(k) for k in R))
cur={'banded':np.array([R[str(e)]['banded'] for e in E]),'twochan':np.array([R[str(e)]['twochan'] for e in E])}
def mk(name):
    lx=np.log(E);ly=np.log(cur[name])
    return lambda e: float(np.exp(np.interp(np.log(e),lx,ly)))   # log-log linear interpolation, valid for 0.001<=e<=0.3
C={k:mk(k) for k in cur}
B=0.01
out=[]
def P(*a):
    s=' '.join(str(x) for x in a);print(s);out.append(s)
# convexity check of interpolant in eps
for k in cur:
    g=np.geomspace(0.001,0.3,400);v=np.array([C[k](x) for x in g])
    d2=np.diff(v,2)/np.diff(g)[:-1]**2
    P(k,'min second diff (convexity in eps, >=0 is convex):',d2.min())
def opt_sum(c,n,budget=B,paths=None):
    # minimize sum c(e_i), sum e_i<=budget, e_i>=0.001/ ... bounded to [1e-3,0.3]
    best=None
    rng=np.random.default_rng(0)
    for t in range(30):
        x0=rng.dirichlet(np.ones(n))*budget if t else np.full(n,budget/n)
        x0=np.clip(x0,1.0001e-3,None)
        r=minimize(lambda e:sum(c(v) for v in e),x0,method='SLSQP',bounds=[(1e-3,0.3)]*n,
                   constraints=[{'type':'ineq','fun':lambda e:budget-e.sum()}])
        if r.success and r.x.sum()<=budget*(1+1e-9) and (best is None or r.fun<best.fun): best=r
    return best
P('B1 single elimination, sum eps<=0.01')
for k in cur:
    for N in (2,3,4,8):
        n=N-1;eq=B/n
        if eq<1e-3: P(k,N,'eps_eq',eq,'below computed range 0.001, interpolation clamps (np.interp flat extrapolation)');
        tot=n*C[k](eq)
        r=opt_sum(C[k],n)
        P(k,'N',N,'comps',n,'eps_each',round(eq,6),'photons/comp',round(C[k](eq),2),'total',round(tot,2),'| optimizer best',round(r.fun,2) if r is not None else None,'eps',np.round(r.x,5).tolist() if r is not None else None)
# B2 trees: list of paths as lists of stage indices, stage count
trees={3:(3,[[0,1],[0,1,2]]),4:(5,[[0,1,2,3],[0,1,2,3,4]])}
P('B2 sorting trees (worst-path photons = max over executed paths)')
for k in cur:
    for N,(ns,paths) in trees.items():
        L=max(len(p) for p in paths)
        eq=B/L;uni=max(sum(C[k](eq) for _ in p) for p in paths)
        P(k,'N',N,'uniform eps',round(eq,6),'worst-path photons',round(uni,2),'(all-nodes eps=0.01/maxdepth)')
        best=None
        rng=np.random.default_rng(1)
        for t in range(60):
            x0=np.concatenate([[rng.uniform(1.5e-3,B/2) for _ in range(ns)],[0.0]])
            x0=np.clip(x0,1.5e-3,None);x0[-1]=max(sum(C[k](v) for v in x0[:ns]),1)
            cons=[{'type':'ineq','fun':(lambda z,p=p:B-sum(z[i] for i in p))} for p in paths]
            cons+=[{'type':'ineq','fun':(lambda z,p=p:z[-1]-sum(C[k](z[i]) for i in p))} for p in paths]
            r=minimize(lambda z:z[-1],x0,method='SLSQP',bounds=[(1e-3,0.3)]*ns+[(0,1e5)],constraints=cons,options={'maxiter':500})
            if r.success and all(c['fun'](r.x)>=-1e-9 for c in cons) and (best is None or r.fun<best.fun): best=r
        P(k,'N',N,'optimized per-stage worst-path photons',round(best.fun,2),'eps stages',np.round(best.x[:ns],5).tolist(),'path sums',[round(sum(best.x[i] for i in p),5) for p in paths])
# B3 fits on 0.001..0.1
m=(E>=0.001)&(E<=0.1)
P('B3 fits on eps in',E[m].tolist())
fits={}
for k in cur:
    y=cur[k][m];e=E[m]
    A=np.c_[np.log(1/e),np.ones(m.sum())];co=np.linalg.lstsq(A,y,rcond=None)[0]
    rl=np.sqrt(np.mean((np.log(A@co)-np.log(y))**2))  # compare in log space rms
    A2=np.c_[np.log(1/e),np.ones(m.sum())];p=np.polyfit(np.log(1/e),np.log(y),1)
    pred=np.exp(np.polyval(p,np.log(1/e)));rp=np.sqrt(np.mean((np.log(pred)-np.log(y))**2))
    P(k,'log fit C=a ln(1/e)+b: a=%.3f b=%.3f rms log-resid=%.4f maxrel=%.4f'%(co[0],co[1],rl,np.max(np.abs(A@co/y-1))))
    P(k,'power fit C=c*(1/e)^p: c=%.3f p=%.4f rms log-resid=%.4f maxrel=%.4f'%(np.exp(p[1]),p[0],rp,np.max(np.abs(pred/y-1))))
    fits[k]=(co,p)
def FJ(n): return sum(math.ceil(math.log2(3*k/4)) for k in range(1,n+1))
P('Ford-Johnson counts',{n:FJ(n) for n in (3,4,8,16,32)},'ceil(log2 N!)',{n:math.ceil(math.log2(math.factorial(n))) for n in (3,4,8,16,32)})
for k in cur:
    co,p=fits[k]
    for N in (8,16,32):
        for lab,cnt in (('FJ',FJ(N)),('infobound',math.ceil(math.log2(math.factorial(N))))):
            e=B/cnt
            tl=cnt*(co[0]*math.log(1/e)+co[1]);tp=cnt*math.exp(np.polyval(p,math.log(1/e)))
            P('EXTRAPOLATION',k,'N',N,lab,'comps',cnt,'eps',round(e,6),'log-fit total',round(tl,1),'power-fit total',round(tp,1),'in-range interp' if e>=1e-3 else 'eps below 0.001 (extrapolated)')
open(os.environ.get('TOURN_OUT','results/tournament.txt'),'w').write('\n'.join(out) + '\n')
